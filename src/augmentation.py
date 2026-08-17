"""데이터 증강(회전·반전) 효과 검증.

웨이퍼맵은 회전·반전해도 결함 패턴의 종류가 바뀌지 않는다
(Center를 90도 돌려도 Center, Scratch를 뒤집어도 Scratch).
이 도메인 성질을 이용해 소수 클래스를 늘렸을 때 실제로 성능이
개선되는지, 증강만 넣고 빼서 비교한다.

주의: 증강은 반드시 train split에만 적용한다. test에 증강본이
섞이면 같은 웨이퍼의 변형이 학습·평가 양쪽에 들어가 성능이
부풀려진다(데이터 누수).
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_num_threads(1)  # 재현성: CPU 멀티스레드 연산의 부동소수점 비결정성 제거
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data import DEFECT_CLASSES, load_labeled
from derived_features import (FEATURE_NAMES, HybridCNN, HybridDataset,
                              extract_features)
from train_cnn import (BATCH_SIZE, EPOCHS, RANDOM_STATE, build_balanced_subset,
                       resize_wafer)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# 소수 클래스만 증강한다. 이미 충분한 클래스까지 늘리면
# 학습 시간만 늘고 불균형 해소 효과는 없다.
AUGMENT_TARGETS = ["Scratch", "Random", "Donut", "Near-full", "Loc"]


def augment_variants(m):
    """회전 4방향 x 좌우반전 = 최대 8개 변형 (원본 포함)."""
    out = []
    for k in range(4):
        r = np.rot90(m, k)
        out.append(r)
        out.append(np.fliplr(r))
    return out


def build_augmented_train(subset, tr_idx):
    """train split의 소수 클래스만 증강해 (waferMap, label) 리스트 반환."""
    maps, labels = [], []
    for i in tr_idx:
        m = subset["waferMap"].iloc[i]
        cls = subset["failureType"].iloc[i]
        if cls in AUGMENT_TARGETS:
            variants = augment_variants(m)
        else:
            variants = [m]
        for v in variants:
            maps.append(v)
            labels.append(cls)
    return maps, labels


def train_eval(train_maps, train_labels, test_maps, test_labels, tag):
    label_to_idx = {c: i for i, c in enumerate(DEFECT_CLASSES)}

    X_tr = np.stack([resize_wafer(m) for m in train_maps])
    F_tr = np.array([extract_features(m) for m in train_maps], dtype=np.float32)
    y_tr = np.array([label_to_idx[c] for c in train_labels])

    X_te = np.stack([resize_wafer(m) for m in test_maps])
    F_te = np.array([extract_features(m) for m in test_maps], dtype=np.float32)
    y_te = np.array([label_to_idx[c] for c in test_labels])

    scaler = StandardScaler().fit(F_tr)
    F_tr = scaler.transform(F_tr).astype(np.float32)
    F_te = scaler.transform(F_te).astype(np.float32)

    train_loader = DataLoader(HybridDataset(X_tr, F_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(HybridDataset(X_te, F_te, y_te), batch_size=256, shuffle=False)

    counts = np.bincount(y_tr, minlength=len(DEFECT_CLASSES))
    weights = torch.tensor(1.0 / np.sqrt(counts + 1), dtype=torch.float32)
    weights = weights / weights.sum() * len(DEFECT_CLASSES)

    torch.manual_seed(RANDOM_STATE)
    model = HybridCNN(len(DEFECT_CLASSES), F_tr.shape[1], use_feats=True)
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
    print(f"\n=== {tag} (train={len(y_tr)}) ===")
    print(classification_report(trues, preds, target_names=DEFECT_CLASSES, zero_division=0))
    return {
        "train_size": int(len(y_tr)),
        "macro_f1": round(float(rep["macro avg"]["f1-score"]), 4),
        "accuracy": round(float(rep["accuracy"]), 4),
        "per_class_f1": {c: round(float(rep[c]["f1-score"]), 4) for c in DEFECT_CLASSES},
    }


def main():
    t0 = time.time()
    subset = build_balanced_subset(load_labeled())
    labels_arr = subset["failureType"].values

    # split을 먼저 하고, 증강은 train에만 적용 (누수 방지)
    idx = np.arange(len(subset))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=labels_arr,
                                      random_state=RANDOM_STATE)
    test_maps = [subset["waferMap"].iloc[i] for i in te_idx]
    test_labels = [subset["failureType"].iloc[i] for i in te_idx]

    base_maps = [subset["waferMap"].iloc[i] for i in tr_idx]
    base_labels = [subset["failureType"].iloc[i] for i in tr_idx]
    aug_maps, aug_labels = build_augmented_train(subset, tr_idx)

    from collections import Counter
    print("증강 전 train 분포:", dict(Counter(base_labels)))
    print("증강 후 train 분포:", dict(Counter(aug_labels)))

    baseline = train_eval(base_maps, base_labels, test_maps, test_labels, "증강 없음")
    augmented = train_eval(aug_maps, aug_labels, test_maps, test_labels, "회전·반전 증강")

    comparison = {
        "augment_targets": AUGMENT_TARGETS,
        "no_augmentation": baseline,
        "with_augmentation": augmented,
        "macro_f1_delta": round(augmented["macro_f1"] - baseline["macro_f1"], 4),
        "per_class_f1_delta": {
            c: round(augmented["per_class_f1"][c] - baseline["per_class_f1"][c], 4)
            for c in DEFECT_CLASSES
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(ROOT / "results" / "augmentation_experiment.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    x = np.arange(len(DEFECT_CLASSES))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, [baseline["per_class_f1"][c] for c in DEFECT_CLASSES], 0.4, label="No augmentation")
    ax.bar(x + 0.2, [augmented["per_class_f1"][c] for c in DEFECT_CLASSES], 0.4, label="With rotation/flip augmentation")
    ax.set_xticks(x)
    ax.set_xticklabels(DEFECT_CLASSES, rotation=45, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1: effect of rotation/flip augmentation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_augmentation_comparison.png", dpi=120)
    plt.close(fig)

    print("\n=== 비교 결과 ===")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
