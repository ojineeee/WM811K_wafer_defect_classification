"""lot 순서 기반 분할(chronological-analog) 검증.

lot_drift.py에서 lot 순서 초반(전체의 20%) 구간의 불량 비율이 약 50%로,
후반 구간(약 5~10%)과 크게 다르다는 걸 확인했다. 지금까지의 모든 실험은
무작위 stratified split이라 이 drift를 무시하고 있었다. SECOM에서 랜덤
분할이 실제 성능을 과대평가했던 것과 같은 위험이 있는지, lot 순서 그대로
앞 80% 학습 / 뒤 20% 평가로 다시 검증한다.

train 쪽은 기존과 동일하게 클래스별 최대 2,000장으로 균형을 맞추지만,
test 쪽(미래 구간)은 그 시점에 "실제로 존재하는 만큼"만 쓴다 — 균형을
맞추지 않는 것 자체가 실전을 더 정직하게 반영한다(희귀 클래스가 그
구간에 거의 없다면 그 사실 자체가 결과의 일부).
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data import DEFECT_CLASSES, load_labeled
from derived_features import FEATURE_NAMES, HybridCNN, HybridDataset, extract_features
from train_cnn import BATCH_SIZE, EPOCHS, RANDOM_STATE, resize_wafer

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"

TRAIN_CAP_PER_CLASS = 2000
TEST_CAP_PER_CLASS = 500  # 미래 구간 평가셋 크기 상한 (계산량 제어 목적, 업샘플링 아님)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def cap_per_class(df, cap, seed=RANDOM_STATE):
    parts = []
    for cls in DEFECT_CLASSES:
        sub = df[df["failureType"] == cls]
        if len(sub) > cap:
            sub = sub.sample(cap, random_state=seed)
        parts.append(sub)
    return __import__("pandas").concat(parts).reset_index(drop=True)


def build_xy(sub):
    images = np.stack([resize_wafer(m) for m in sub["waferMap"]])
    feats = np.array([extract_features(m) for m in sub["waferMap"]], dtype=np.float32)
    labels = sub["failureType"].map({c: i for i, c in enumerate(DEFECT_CLASSES)}).values
    return images, feats, labels


def main():
    t0 = time.time()
    df = load_labeled()
    df["lot_num"] = df["lotName"].str.extract(r"lot(\d+)").astype(int)
    df = df.sort_values("lot_num").reset_index(drop=True)

    split_point = int(len(df) * 0.8)
    pool_train, pool_test = df.iloc[:split_point], df.iloc[split_point:]

    print(f"train lot 범위: lot{pool_train['lot_num'].min()} ~ lot{pool_train['lot_num'].max()}")
    print(f"test  lot 범위: lot{pool_test['lot_num'].min()} ~ lot{pool_test['lot_num'].max()}")
    print(f"train 불량률(non-none): {(pool_train['failureType']!='none').mean():.4f}")
    print(f"test  불량률(non-none): {(pool_test['failureType']!='none').mean():.4f}")

    train_sub = cap_per_class(pool_train, TRAIN_CAP_PER_CLASS)
    test_sub = cap_per_class(pool_test, TEST_CAP_PER_CLASS)

    print("\ntrain 클래스 분포:\n", train_sub["failureType"].value_counts())
    print("\ntest 클래스 분포 (미래 구간에 자연적으로 존재하는 만큼):\n", test_sub["failureType"].value_counts())

    X_tr_img, F_tr, y_tr = build_xy(train_sub)
    X_te_img, F_te, y_te = build_xy(test_sub)

    scaler = StandardScaler().fit(F_tr)
    F_tr_s = scaler.transform(F_tr).astype(np.float32)
    F_te_s = scaler.transform(F_te).astype(np.float32)

    train_loader = DataLoader(HybridDataset(X_tr_img, F_tr_s, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(HybridDataset(X_te_img, F_te_s, y_te), batch_size=256, shuffle=False)

    counts = np.bincount(y_tr, minlength=len(DEFECT_CLASSES))
    weights = torch.tensor(1.0 / np.sqrt(counts + 1), dtype=torch.float32)
    weights = weights / weights.sum() * len(DEFECT_CLASSES)

    torch.manual_seed(RANDOM_STATE)
    model = HybridCNN(len(DEFECT_CLASSES), F_tr_s.shape[1], use_feats=True)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
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
    print("\n=== lot 순서 분할 결과 ===")
    print(classification_report(trues, preds, target_names=DEFECT_CLASSES, zero_division=0))

    # 참고: 무작위 분할(derived_feature_experiment.json)의 cnn_plus_derived_features 결과
    random_split_reference = {
        "macro_f1": 0.8436,
        "accuracy": None,
    }

    result = {
        "train_lot_range": [int(pool_train["lot_num"].min()), int(pool_train["lot_num"].max())],
        "test_lot_range": [int(pool_test["lot_num"].min()), int(pool_test["lot_num"].max())],
        "train_defect_rate": round(float((pool_train["failureType"] != "none").mean()), 4),
        "test_defect_rate": round(float((pool_test["failureType"] != "none").mean()), 4),
        "test_class_support": {c: int((test_sub["failureType"] == c).sum()) for c in DEFECT_CLASSES},
        "lot_split_macro_f1": round(float(rep["macro avg"]["f1-score"]), 4),
        "lot_split_accuracy": round(float(rep["accuracy"]), 4),
        "lot_split_per_class_f1": {c: round(float(rep[c]["f1-score"]), 4) for c in DEFECT_CLASSES},
        "random_split_reference_macro_f1": random_split_reference["macro_f1"],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(ROOT / "results" / "lot_split_comparison.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels_plot = ["random_split", "lot_split"]
    values = [random_split_reference["macro_f1"], result["lot_split_macro_f1"]]
    ax.bar(labels_plot, values, color=["#4C72B0", "#C44E52"])
    ax.set_ylabel("Macro F1")
    ax.set_title("Random split vs Lot-order split — WM-811K")
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "17_lot_split_comparison.png", dpi=120)
    plt.close(fig)

    print("\n결과 저장 완료: results/lot_split_comparison.json, figures/17_lot_split_comparison.png")


if __name__ == "__main__":
    main()
