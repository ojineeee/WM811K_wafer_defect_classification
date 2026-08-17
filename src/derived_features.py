"""웨이퍼맵에서 도메인 기반 파생변수를 추출하고, CNN에 추가했을 때
실제로 성능이 개선되는지 검증한다.

CNN은 이미지에서 특징을 스스로 학습하므로 파생변수가 불필요하다고
보기 쉽지만, 64x64로 축소하는 과정에서 손실되는 정보(원본 해상도의
불량 밀도, 결함의 공간적 산포도 등)를 보완할 수 있는지 확인한다.
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)  # 재현성: CPU 멀티스레드 연산의 부동소수점 비결정성 제거
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from data import DEFECT_CLASSES, load_labeled
from train_cnn import (BATCH_SIZE, CAP_PER_CLASS, EPOCHS, IMG_SIZE, RANDOM_STATE,
                       build_balanced_subset, resize_wafer)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

FEATURE_NAMES = [
    "defect_density", "defect_radial_mean", "defect_radial_std",
    "defect_spread", "defect_row_concentration", "defect_col_concentration",
    "wafer_area", "aspect_ratio",
]


def extract_features(m):
    """원본 해상도(리사이즈 전) 웨이퍼맵에서 공간적 특징을 추출."""
    valid = m != 0
    defect = m == 2
    n_valid = valid.sum()
    n_defect = defect.sum()

    if n_valid == 0 or n_defect == 0:
        return [0.0] * len(FEATURE_NAMES)

    h, w = m.shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    ys, xs = np.nonzero(defect)

    # 중심으로부터의 정규화된 거리 -> Center/Edge 계열 구분에 직접적
    norm = np.hypot(cy, cx) or 1.0
    dists = np.hypot(ys - cy, xs - cx) / norm

    # 불량 좌표의 산포 정도 -> Scratch(길쭉함) vs Loc(뭉침) 구분 목적
    spread = float(np.hypot(ys.std(), xs.std()))

    # 특정 행/열에 몰려있는 정도 -> Scratch의 선형성 포착 목적
    row_counts = defect.sum(axis=1)
    col_counts = defect.sum(axis=0)
    row_conc = float(row_counts.max() / n_defect)
    col_conc = float(col_counts.max() / n_defect)

    return [
        float(n_defect / n_valid),
        float(dists.mean()),
        float(dists.std()),
        spread,
        row_conc,
        col_conc,
        float(n_valid),
        float(w / h),
    ]


class HybridDataset(Dataset):
    def __init__(self, images, feats, labels):
        self.images = images
        self.feats = feats
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.images[i]).unsqueeze(0),
            torch.from_numpy(self.feats[i]).float(),
            self.labels[i],
        )


class HybridCNN(nn.Module):
    """train_cnn.SmallCNN과 동일한 CNN 백본 + 파생변수를 분류기 직전에 결합."""

    def __init__(self, n_classes, n_feats, use_feats=True):
        super().__init__()
        self.use_feats = use_feats
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        in_dim = 64 * 4 * 4 + (n_feats if use_feats else 0)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, f):
        z = self.features(x).flatten(1)
        if self.use_feats:
            z = torch.cat([z, f], dim=1)
        return self.classifier(z)


def run(use_feats, data, label):
    X_tr_img, X_te_img, F_tr, F_te, y_tr, y_te = data
    train_loader = DataLoader(HybridDataset(X_tr_img, F_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(HybridDataset(X_te_img, F_te, y_te), batch_size=256, shuffle=False)

    counts = np.bincount(y_tr, minlength=len(DEFECT_CLASSES))
    weights = torch.tensor(1.0 / np.sqrt(counts + 1), dtype=torch.float32)
    weights = weights / weights.sum() * len(DEFECT_CLASSES)

    torch.manual_seed(RANDOM_STATE)
    model = HybridCNN(len(DEFECT_CLASSES), F_tr.shape[1], use_feats=use_feats)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(EPOCHS):
        model.train()
        for xb, fb, yb in train_loader:
            optimizer.zero_grad()
            criterion(model(xb, fb), yb).backward()
            optimizer.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, fb, yb in test_loader:
            preds.append(model(xb, fb).argmax(1).numpy())
            trues.append(yb.numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)

    rep = classification_report(trues, preds, target_names=DEFECT_CLASSES,
                                 output_dict=True, zero_division=0)
    print(f"\n=== {label} ===")
    print(classification_report(trues, preds, target_names=DEFECT_CLASSES, zero_division=0))
    return {
        "macro_f1": round(float(rep["macro avg"]["f1-score"]), 4),
        "accuracy": round(float(rep["accuracy"]), 4),
        "per_class_f1": {c: round(float(rep[c]["f1-score"]), 4) for c in DEFECT_CLASSES},
    }


def main():
    t0 = time.time()
    df = load_labeled()
    subset = build_balanced_subset(df)

    print("파생변수 추출 중...")
    feats = np.array([extract_features(m) for m in subset["waferMap"]], dtype=np.float32)
    images = np.stack([resize_wafer(m) for m in subset["waferMap"]])
    labels = subset["failureType"].map({c: i for i, c in enumerate(DEFECT_CLASSES)}).values

    # 클래스별 파생변수 평균 (해석용)
    feat_df = pd.DataFrame(feats, columns=FEATURE_NAMES)
    feat_df["class"] = subset["failureType"].values
    class_means = feat_df.groupby("class")[FEATURE_NAMES].mean().reindex(DEFECT_CLASSES)
    print("\n클래스별 파생변수 평균:")
    print(class_means.round(3).to_string())
    class_means.round(4).to_csv(ROOT / "results" / "derived_feature_by_class.csv")

    idx = np.arange(len(labels))
    tr, te = train_test_split(idx, test_size=0.2, stratify=labels, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(feats[tr])  # train으로만 fit (누수 방지)
    data = (images[tr], images[te], scaler.transform(feats[tr]).astype(np.float32),
            scaler.transform(feats[te]).astype(np.float32), labels[tr], labels[te])

    baseline = run(False, data, "CNN only (파생변수 없음)")
    hybrid = run(True, data, "CNN + 파생변수")

    comparison = {
        "cnn_only": baseline,
        "cnn_plus_derived_features": hybrid,
        "macro_f1_delta": round(hybrid["macro_f1"] - baseline["macro_f1"], 4),
        "scratch_f1_delta": round(
            hybrid["per_class_f1"]["Scratch"] - baseline["per_class_f1"]["Scratch"], 4
        ),
        "feature_names": FEATURE_NAMES,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(ROOT / "results" / "derived_feature_experiment.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    classes = DEFECT_CLASSES
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, [baseline["per_class_f1"][c] for c in classes], 0.4, label="CNN only")
    ax.bar(x + 0.2, [hybrid["per_class_f1"][c] for c in classes], 0.4, label="CNN + derived features")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1: CNN only vs CNN + derived features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_derived_feature_comparison.png", dpi=120)
    plt.close(fig)

    print("\n=== 비교 결과 ===")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
