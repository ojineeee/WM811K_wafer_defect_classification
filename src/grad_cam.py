"""Grad-CAM: CNN이 웨이퍼의 어느 부분을 보고 판단했는지 시각화.

SECOM 프로젝트의 SHAP과 대칭되는 설명력 분석. SHAP은 "어떤 센서가
기여했는가"를 숫자로 분해했다면, Grad-CAM은 "이미지의 어느 영역이
판단에 기여했는가"를 히트맵으로 보여준다.

lot_final_evaluation.py(lot 기준 train/val/test 분리)가 선택하고 test에서
딱 한 번 평가한 최종 모델(results/models/final_model.pt)과 그때 쓴 동일한
test split을 그대로 불러와 재사용한다 — 재학습하면서 다시 시드가 갈라지거나
test set이 또 바뀌는 걸 막기 위함이다. (구버전 final_evaluation.py가 저장한
웨이퍼 단위 무작위 분할 모델도 `split_type` 필드로 구분해 계속 로드할 수
있다.)

한계: Grad-CAM은 이 모델의 CNN(이미지) 경로에 걸린 hook만 설명한다.
최종 조합이 파생변수(derived features)를 함께 쓰는 하이브리드 모델이라면,
분류기 직전에 concat되는 파생변수 경로는 Grad-CAM으로 설명되지 않는
"보이지 않는 두 번째 경로"로 남는다 — 아래 히트맵은 "이미지만으로 얼마나
설명되는가"이지 "모델의 전체 판단 근거"가 아니다.
"""
import json
import pickle
from pathlib import Path

from plot_style import plt
import numpy as np
import torch
import torch.nn.functional as F

from data import DEFECT_CLASSES, load_labeled
from derived_features import FEATURE_NAMES, HybridCNN, extract_features
from train_cnn import RANDOM_STATE, build_balanced_subset, resize_wafer

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
MODEL_DIR = ROOT / "results" / "models"

RNG = np.random.default_rng(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def load_final_model():
    """최종 평가 스크립트가 저장한 최종 모델·스케일러·test 데이터를 그대로 불러온다.

    lot_final_evaluation.py(lot 기반, split_type="lot_based")와 구버전
    final_evaluation.py(웨이퍼 단위 무작위 분할) 둘 다 지원한다.
    """
    with open(MODEL_DIR / "final_test_indices.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open(MODEL_DIR / "final_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    if meta.get("split_type") == "lot_based":
        with open(MODEL_DIR / "lot_test_subset.pkl", "rb") as f:
            test_sub = pickle.load(f)
        test_maps = list(test_sub["waferMap"])
        test_labels = list(test_sub["failureType"])
        n_test = len(test_sub)
    else:
        df = load_labeled()
        subset = build_balanced_subset(df)
        te_idx = meta["te_idx"]
        test_maps = [subset["waferMap"].iloc[i] for i in te_idx]
        test_labels = [subset["failureType"].iloc[i] for i in te_idx]
        n_test = len(te_idx)

    label_to_idx = {c: i for i, c in enumerate(DEFECT_CLASSES)}
    X_te = np.stack([resize_wafer(m) for m in test_maps])
    y_te = np.array([label_to_idx[c] for c in test_labels])
    use_feats = meta["use_feats"]
    if use_feats:
        F_te_raw = np.array([extract_features(m) for m in test_maps], dtype=np.float32)
        F_te = scaler.transform(F_te_raw).astype(np.float32)
    else:
        F_te = np.zeros((len(X_te), len(FEATURE_NAMES)), dtype=np.float32)

    model = HybridCNN(len(DEFECT_CLASSES), len(FEATURE_NAMES), use_feats=use_feats)
    model.load_state_dict(torch.load(MODEL_DIR / "final_model.pt", map_location="cpu"))
    model.eval()

    print(f"최종 모델 로드 완료 (config={meta['best_config']}, use_feats={use_feats}, "
          f"split_type={meta.get('split_type', 'random')}, test={n_test})")
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
    model, X_te, F_te, y_te, test_maps, test_labels, label_to_idx = load_final_model()

    # 마지막 conv 블록: features[6]=Conv2d, [7]=ReLU (여기서 hook, AdaptiveAvgPool 이전 -> 16x16 공간정보 보존)
    target_layer = model.features[7]
    cam_engine = GradCAM(model, target_layer)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    report = {}
    for ax, cls in zip(axes.flat, DEFECT_CLASSES):
        cls_idx = label_to_idx[cls]
        candidates = np.where((y_te == cls_idx))[0]
        # 정확히 맞춘 사례 중 무작위로 하나를 선택 ("첫 번째로 맞춘 사례"는 우연히 쉬운
        # 예시일 수 있어, 대표성을 위해 후보 중 무작위 추출로 바꿈)
        correct_candidates = []
        for i in candidates:
            x = torch.from_numpy(X_te[i:i+1]).unsqueeze(1)
            f = torch.from_numpy(F_te[i:i+1])
            with torch.no_grad():
                pred = model(x, f).argmax(1).item()
            if pred == cls_idx:
                correct_candidates.append(i)
        if correct_candidates:
            chosen = int(RNG.choice(correct_candidates))
        elif len(candidates):
            chosen = int(RNG.choice(candidates))
        else:
            chosen = None
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
        report[cls] = {
            "test_index": int(chosen),
            "predicted_correctly": bool(chosen in correct_candidates),
            "probability": round(float(prob), 4),
        }

    fig.suptitle("Grad-CAM: 클래스별 모델이 주목한 영역 (정확히 맞춘 사례)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "21_grad_cam_by_class.png", dpi=130)
    plt.close(fig)

    # Scratch 심층 분석: 가장 어려웠던 클래스가 실제로 선을 보고 있는지 확인
    scratch_idx = label_to_idx["Scratch"]
    scratch_pool = np.where(y_te == scratch_idx)[0]
    scratch_candidates = RNG.choice(scratch_pool, size=min(4, len(scratch_pool)), replace=False) if len(scratch_pool) else scratch_pool
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
