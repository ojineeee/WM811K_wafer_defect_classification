"""Grad-CAM: CNN이 웨이퍼의 어느 부분을 보고 판단했는지 시각화.

SECOM 프로젝트의 SHAP과 대칭되는 설명력 분석. SHAP은 "어떤 센서가
기여했는가"를 숫자로 분해했다면, Grad-CAM은 "이미지의 어느 영역이
판단에 기여했는가"를 히트맵으로 보여준다.

가장 성능이 좋았던 조합(파생변수 + 증강, ablation_with_ci.py의 "both")
모델을 다시 학습시킨 뒤, 마지막 conv 레이어(공간 정보가 남아있는 마지막
지점)의 활성값과 그레디언트로 클래스별 관심 영역을 계산한다.
"""
import json
import time
from pathlib import Path

from plot_style import plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data import DEFECT_CLASSES, load_labeled
from derived_features import FEATURE_NAMES, HybridCNN, HybridDataset, extract_features
from train_cnn import BATCH_SIZE, EPOCHS, RANDOM_STATE, build_balanced_subset, resize_wafer
from augmentation import AUGMENT_TARGETS, augment_variants

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def build_augmented(subset, idx):
    maps, labels = [], []
    for i in idx:
        m = subset["waferMap"].iloc[i]
        cls = subset["failureType"].iloc[i]
        variants = augment_variants(m) if cls in AUGMENT_TARGETS else [m]
        for v in variants:
            maps.append(v)
            labels.append(cls)
    return maps, labels


def train_best_model():
    """ablation_with_ci.py의 "both"(파생변수+증강) 조건과 동일하게 재학습."""
    df = load_labeled()
    subset = build_balanced_subset(df)
    labels_arr = subset["failureType"].values
    idx = np.arange(len(subset))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=labels_arr, random_state=RANDOM_STATE)

    test_maps = [subset["waferMap"].iloc[i] for i in te_idx]
    test_labels = [subset["failureType"].iloc[i] for i in te_idx]
    aug_maps, aug_labels = build_augmented(subset, tr_idx)

    label_to_idx = {c: i for i, c in enumerate(DEFECT_CLASSES)}
    X_tr = np.stack([resize_wafer(m) for m in aug_maps])
    F_tr_raw = np.array([extract_features(m) for m in aug_maps], dtype=np.float32)
    y_tr = np.array([label_to_idx[c] for c in aug_labels])
    X_te = np.stack([resize_wafer(m) for m in test_maps])
    F_te_raw = np.array([extract_features(m) for m in test_maps], dtype=np.float32)
    y_te = np.array([label_to_idx[c] for c in test_labels])

    scaler = StandardScaler().fit(F_tr_raw)
    F_tr = scaler.transform(F_tr_raw).astype(np.float32)
    F_te = scaler.transform(F_te_raw).astype(np.float32)

    train_loader = DataLoader(HybridDataset(X_tr, F_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    counts = np.bincount(y_tr, minlength=len(DEFECT_CLASSES))
    weights = torch.tensor(1.0 / np.sqrt(counts + 1), dtype=torch.float32)
    weights = weights / weights.sum() * len(DEFECT_CLASSES)

    torch.manual_seed(RANDOM_STATE)
    model = HybridCNN(len(DEFECT_CLASSES), F_tr.shape[1], use_feats=True)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for xb, fb, yb in train_loader:
            optimizer.zero_grad()
            criterion(model(xb, fb), yb).backward()
            optimizer.step()
        print(f"epoch {epoch+1}/{EPOCHS} done, elapsed={time.time()-t0:.0f}s")

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_te).unsqueeze(1),
                       torch.from_numpy(F_te)).argmax(1).numpy()
    print("test macro F1:", round(float(f1_score(y_te, preds, average="macro")), 4))

    return model, X_te, F_te, y_te, test_maps, test_labels, label_to_idx


class GradCAM:
    """마지막 conv 블록(features[7] = 두 번째 Conv2d 다음 ReLU)의 활성값/그레디언트로 CAM 계산."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, f, class_idx):
        self.model.zero_grad()
        out = self.model(x, f)
        score = out[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, torch.softmax(out, dim=1)[0, class_idx].item()


def main():
    model, X_te, F_te, y_te, test_maps, test_labels, label_to_idx = train_best_model()

    # 마지막 conv 블록: features[6]=Conv2d, [7]=ReLU (여기서 hook, AdaptiveAvgPool 이전 -> 16x16 공간정보 보존)
    target_layer = model.features[7]
    cam_engine = GradCAM(model, target_layer)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    report = {}
    for ax, cls in zip(axes.flat, DEFECT_CLASSES):
        cls_idx = label_to_idx[cls]
        candidates = np.where((y_te == cls_idx))[0]
        # 정확히 맞춘 사례 중 하나를 선택
        chosen = None
        for i in candidates:
            x = torch.from_numpy(X_te[i:i+1]).unsqueeze(1)
            f = torch.from_numpy(F_te[i:i+1])
            with torch.no_grad():
                pred = model(x, f).argmax(1).item()
            if pred == cls_idx:
                chosen = i
                break
        if chosen is None:
            chosen = candidates[0] if len(candidates) else None
        if chosen is None:
            ax.axis("off")
            ax.set_title(f"{cls} (테스트 샘플 없음)", fontsize=10)
            continue

        x = torch.from_numpy(X_te[chosen:chosen+1]).unsqueeze(1).requires_grad_(False)
        f = torch.from_numpy(F_te[chosen:chosen+1])
        cam, prob = cam_engine(x, f, cls_idx)

        orig = X_te[chosen]
        ax.imshow(orig, cmap="gray", vmin=0, vmax=1)
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(f"{cls}  (p={prob:.2f})", fontsize=10)
        ax.axis("off")
        report[cls] = {"test_index": int(chosen), "predicted_correctly": True, "probability": round(float(prob), 4)}

    fig.suptitle("Grad-CAM: 클래스별 모델이 주목한 영역 (정확히 맞춘 사례)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "21_grad_cam_by_class.png", dpi=130)
    plt.close(fig)

    # Scratch 심층 분석: 가장 어려웠던 클래스가 실제로 선을 보고 있는지 확인
    scratch_idx = label_to_idx["Scratch"]
    scratch_candidates = np.where(y_te == scratch_idx)[0][:4]
    if len(scratch_candidates) > 0:
        fig, axes = plt.subplots(1, len(scratch_candidates), figsize=(4 * len(scratch_candidates), 4))
        if len(scratch_candidates) == 1:
            axes = [axes]
        for ax, i in zip(axes, scratch_candidates):
            x = torch.from_numpy(X_te[i:i+1]).unsqueeze(1)
            f = torch.from_numpy(F_te[i:i+1])
            with torch.no_grad():
                pred = model(x, f).argmax(1).item()
            cam, prob = cam_engine(x, f, scratch_idx)
            ax.imshow(X_te[i], cmap="gray", vmin=0, vmax=1)
            ax.imshow(cam, cmap="jet", alpha=0.45)
            correct = "O" if pred == scratch_idx else f"X (예측:{DEFECT_CLASSES[pred]})"
            ax.set_title(f"Scratch #{i}  {correct}", fontsize=9)
            ax.axis("off")
        fig.suptitle("Grad-CAM: Scratch 사례 4개 — 모델이 실제로 선을 보고 있는가")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "22_grad_cam_scratch_detail.png", dpi=130)
        plt.close(fig)

    with open(ROOT / "results" / "grad_cam_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n결과 저장 완료: results/grad_cam_summary.json, figures/21~22")


if __name__ == "__main__":
    main()
