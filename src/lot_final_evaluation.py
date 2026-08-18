"""lot 기준 Train/Validation/Test 3-way 분리 최종 평가.

`final_evaluation.py`는 train/val/test 3-way로 "모델 선택"과 "최종 평가"를
분리했지만, 그 분리 자체는 **웨이퍼 단위 무작위 분할**(클래스별로 상한을 둔
subset을 다시 stratified하게 60/20/20으로 나눈 것)이었다. 같은 lot의 웨이퍼가
train/validation/test 세 곳에 함께 들어갈 수 있다는 뜻이다. `lot_split_validation.py`
에서 이미 확인했듯 lot 순서에 따라 불량 비율이 극단적으로 다르므로(초반 lot 약
50% vs 나머지 5~10%), 같은 lot이 여러 split에 걸치면 "새로운 lot에 대한 일반화"를
과대평가할 위험이 있다.

이 스크립트는 `lot_split_validation.py`(고유 lot 기준 분할)와
`final_evaluation.py`(모델 선택/최종 평가 분리)의 방법론을 합쳐서:

1. 고유 lot을 lot 번호 순으로 정렬 후 앞 60%/다음 20%/마지막 20%를
   train/validation/test lot으로 나눈다 (같은 lot의 웨이퍼는 항상 한 곳에만 속함).
2. train은 클래스당 최대 2,000장으로 제한하고 소수 클래스에만 증강을 적용한다.
   validation/test는 증강하지 않으며, 계산 시간을 제어하기 위해 클래스당 최대
   500장을 고정 시드로 무작위 추출한다(업샘플링은 하지 않는다).
3. 4개 조합(baseline/derived/augment/both)을 train으로 학습해 validation에서
   비교하고 최선의 조합을 고른다 — test는 이 단계에서 전혀 보지 않는다.
4. 최선 조합만 train+validation으로 재학습해 test에서 딱 한 번 평가한다. 이
   test 점수를 클래스 상한이 적용된 신규 lot holdout의 대표 성능으로 보고한다.
5. 신뢰구간은 웨이퍼가 아니라 lot을 재표본추출하는 cluster bootstrap으로 계산해,
   같은 lot 안의 웨이퍼들이 독립 표본이라는 과도한 가정을 피한다.

이 수치는 `final_evaluation.py`가 보고하는 무작위 분할 수치(0.885, 동일 분포
가정)와는 다른 질문에 답한다는 점에 유의해야 한다 — 후자는 "이 클래스 분포에서
패턴을 얼마나 잘 배우는가", 이 스크립트는 "새로운 lot(생산 배치)에서도 그 패턴이
얼마나 유지되는가"를 답한다.
"""
import json
import pickle
import time
from pathlib import Path

from plot_style import plt
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data import DEFECT_CLASSES, load_labeled
from derived_features import FEATURE_NAMES, HybridCNN, HybridDataset, extract_features
from train_cnn import BATCH_SIZE, EPOCHS, RANDOM_STATE, resize_wafer
from augmentation import AUGMENT_TARGETS, augment_variants

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
MODEL_DIR = ROOT / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CAP_PER_CLASS = 2000
EVAL_CAP_PER_CLASS = 500  # validation/test 계산량 제어용 상한 (업샘플링 아님)
N_BOOTSTRAP = 1000

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def cap_per_class(df, cap, seed=RANDOM_STATE):
    parts = []
    for cls in DEFECT_CLASSES:
        sub = df[df["failureType"] == cls]
        if len(sub) > cap:
            sub = sub.sample(cap, random_state=seed)
        parts.append(sub)
    return pd.concat(parts).reset_index(drop=True)


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


def make_xyf(maps, labels_str, zero_feats=False):
    images = np.stack([resize_wafer(m) for m in maps])
    if zero_feats:
        feats = np.zeros((len(maps), len(FEATURE_NAMES)), dtype=np.float32)
    else:
        feats = np.array([extract_features(m) for m in maps], dtype=np.float32)
    labels = np.array([DEFECT_CLASSES.index(c) for c in labels_str])
    return images, feats, labels


def train_model(X_tr, F_tr, y_tr, use_feats, epochs=EPOCHS):
    train_loader = DataLoader(HybridDataset(X_tr, F_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    counts = np.bincount(y_tr, minlength=len(DEFECT_CLASSES))
    weights = torch.tensor(1.0 / np.sqrt(counts + 1), dtype=torch.float32)
    weights = weights / weights.sum() * len(DEFECT_CLASSES)

    torch.manual_seed(RANDOM_STATE)
    model = HybridCNN(len(DEFECT_CLASSES), F_tr.shape[1], use_feats=use_feats)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        model.train()
        for xb, fb, yb in train_loader:
            optimizer.zero_grad()
            criterion(model(xb, fb), yb).backward()
            optimizer.step()
    return model


def predict(model, X, F):
    ds = HybridDataset(X, F, np.zeros(len(X), dtype=int))
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, fb, _ in loader:
            preds.append(model(xb, fb).argmax(1).numpy())
    return np.concatenate(preds)


def lot_cluster_bootstrap_macro_f1(
    y_true, preds_dict, lot_ids, n=N_BOOTSTRAP, seed=RANDOM_STATE
):
    """lot을 복원추출하고 선택된 lot의 모든 웨이퍼로 Macro F1 CI를 계산한다."""
    y_true = np.asarray(y_true)
    lot_ids = np.asarray(lot_ids)
    if len(y_true) != len(lot_ids):
        raise ValueError("y_true와 lot_ids의 길이가 같아야 합니다.")

    unique_lots = np.unique(lot_ids)
    indices_by_lot = {lot: np.flatnonzero(lot_ids == lot) for lot in unique_lots}
    rng = np.random.default_rng(seed)
    labels = np.arange(len(DEFECT_CLASSES))
    scores = {k: [] for k in preds_dict}
    for _ in range(n):
        sampled_lots = rng.choice(unique_lots, size=len(unique_lots), replace=True)
        idx = np.concatenate([indices_by_lot[lot] for lot in sampled_lots])
        yt = y_true[idx]
        for k, preds in preds_dict.items():
            scores[k].append(
                f1_score(
                    yt,
                    np.asarray(preds)[idx],
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            )
    result = {}
    for k, v in scores.items():
        result[k] = {
            "mean": round(float(np.mean(v)), 4),
            "ci_lower": round(float(np.percentile(v, 2.5)), 4),
            "ci_upper": round(float(np.percentile(v, 97.5)), 4),
        }
    return result


def main():
    t0 = time.time()
    df = load_labeled()
    df["lot_num"] = df["lotName"].str.extract(r"lot(\d+)").astype(int)

    unique_lots = np.sort(df["lot_num"].unique())
    n = len(unique_lots)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    train_lots = set(unique_lots[:train_end])
    val_lots = set(unique_lots[train_end:val_end])
    test_lots = set(unique_lots[val_end:])
    assert train_lots.isdisjoint(val_lots) and val_lots.isdisjoint(test_lots) and train_lots.isdisjoint(test_lots)

    pool_train = df[df["lot_num"].isin(train_lots)]
    pool_val = df[df["lot_num"].isin(val_lots)]
    pool_test = df[df["lot_num"].isin(test_lots)]
    print(f"lot 범위 -> train: {min(train_lots)}~{max(train_lots)} ({len(train_lots)} lots, {len(pool_train)}장)")
    print(f"lot 범위 -> val:   {min(val_lots)}~{max(val_lots)} ({len(val_lots)} lots, {len(pool_val)}장)")
    print(f"lot 범위 -> test:  {min(test_lots)}~{max(test_lots)} ({len(test_lots)} lots, {len(pool_test)}장)")

    train_sub = cap_per_class(pool_train, TRAIN_CAP_PER_CLASS).reset_index(drop=True)
    val_sub = cap_per_class(pool_val, EVAL_CAP_PER_CLASS).reset_index(drop=True)
    test_sub = cap_per_class(pool_test, EVAL_CAP_PER_CLASS).reset_index(drop=True)
    print(f"\ntrain subset: {len(train_sub)}장, val subset: {len(val_sub)}장, test subset: {len(test_sub)}장")
    print("\ntrain 클래스 분포:\n", train_sub["failureType"].value_counts())
    print("\nval 클래스 분포 (그 lot 구간에 실제로 존재하는 만큼):\n", val_sub["failureType"].value_counts())
    print("\ntest 클래스 분포 (그 lot 구간에 실제로 존재하는 만큼):\n", test_sub["failureType"].value_counts())

    # ---------- 1단계: val에서 4개 조합 비교 (모델 선택 전용, test는 아직 보지 않음) ----------
    val_maps = list(val_sub["waferMap"])
    val_labels = list(val_sub["failureType"])
    X_val, F_val_raw, y_val = make_xyf(val_maps, val_labels)

    tr_idx = np.arange(len(train_sub))
    base_maps = list(train_sub["waferMap"])
    base_labels = list(train_sub["failureType"])
    aug_maps, aug_labels = build_augmented(train_sub, tr_idx)
    print(f"\n[1단계] base train={len(base_maps)}  augmented train={len(aug_maps)}")

    conditions = {
        "baseline": (base_maps, base_labels, False),
        "derived": (base_maps, base_labels, True),
        "augment": (aug_maps, aug_labels, False),
        "both": (aug_maps, aug_labels, True),
    }

    val_preds = {}
    val_point = {}
    for name, (maps, labels_str, use_feats) in conditions.items():
        X_tr, F_tr_raw, y_tr = make_xyf(maps, labels_str, zero_feats=not use_feats)
        if use_feats:
            scaler = StandardScaler().fit(F_tr_raw)
            F_tr = scaler.transform(F_tr_raw).astype(np.float32)
            F_val = scaler.transform(F_val_raw).astype(np.float32)
        else:
            F_tr, F_val = F_tr_raw, np.zeros((len(X_val), len(FEATURE_NAMES)), dtype=np.float32)

        model = train_model(X_tr, F_tr, y_tr, use_feats)
        preds = predict(model, X_val, F_val)
        val_preds[name] = preds
        mf1 = f1_score(y_val, preds, average="macro")
        val_point[name] = round(float(mf1), 4)
        print(f"[val:{name}] use_feats={use_feats}  macro_f1={mf1:.4f}  elapsed={time.time()-t0:.1f}s")

    val_ci = lot_cluster_bootstrap_macro_f1(
        y_val, val_preds, val_sub["lot_num"].to_numpy(), seed=RANDOM_STATE
    )
    best_name = max(val_point, key=val_point.get)
    print(f"\nval(신규 lot) 기준 최선 조합: {best_name} (macro_f1={val_point[best_name]})")

    # ---------- 2단계: 최선 조합만 train+val로 재학습, test(더 새로운 lot)에서 딱 한 번 평가 ----------
    trainval_sub = pd.concat([train_sub, val_sub]).reset_index(drop=True)
    use_feats_final = best_name in ("derived", "both")
    use_augment_final = best_name in ("augment", "both")

    if use_augment_final:
        trainval_idx = np.arange(len(trainval_sub))
        final_maps, final_labels = build_augmented(trainval_sub, trainval_idx)
    else:
        final_maps = list(trainval_sub["waferMap"])
        final_labels = list(trainval_sub["failureType"])

    test_maps = list(test_sub["waferMap"])
    test_labels = list(test_sub["failureType"])
    X_te, F_te_raw, y_te = make_xyf(test_maps, test_labels)

    X_final, F_final_raw, y_final = make_xyf(final_maps, final_labels, zero_feats=not use_feats_final)
    if use_feats_final:
        final_scaler = StandardScaler().fit(F_final_raw)
        F_final = final_scaler.transform(F_final_raw).astype(np.float32)
        F_te = final_scaler.transform(F_te_raw).astype(np.float32)
    else:
        final_scaler = None
        F_final, F_te = F_final_raw, np.zeros((len(X_te), len(FEATURE_NAMES)), dtype=np.float32)

    print(f"\n[2단계] 최종({best_name}) 재학습: train+val={len(final_maps)}  test={len(test_maps)}")
    final_model = train_model(X_final, F_final, y_final, use_feats_final)
    test_preds = predict(final_model, X_te, F_te)

    rep = classification_report(y_te, test_preds, target_names=DEFECT_CLASSES,
                                 output_dict=True, zero_division=0)
    print("\n=== 최종 lot-test 평가 (단 한 번, 신규 lot에 대한 일반화 성능) ===")
    print(classification_report(y_te, test_preds, target_names=DEFECT_CLASSES, zero_division=0))

    test_ci = lot_cluster_bootstrap_macro_f1(
        y_te,
        {best_name: test_preds},
        test_sub["lot_num"].to_numpy(),
        seed=RANDOM_STATE + 1,
    )[best_name]

    # 최종 모델/스케일러 저장 (grad_cam.py 재사용 목적 — lot 기반 최종 모델로 갱신)
    torch.save(final_model.state_dict(), MODEL_DIR / "final_model.pt")
    with open(MODEL_DIR / "final_scaler.pkl", "wb") as f:
        pickle.dump(final_scaler, f)
    with open(MODEL_DIR / "final_test_indices.json", "w", encoding="utf-8") as f:
        json.dump({
            "split_type": "lot_based",
            "test_lot_range": [int(min(test_lots)), int(max(test_lots))],
            "best_config": best_name,
            "use_feats": use_feats_final,
        }, f)
    # grad_cam.py가 재현 가능하도록 test subset 자체도 저장 (lot 기반은 test 인덱스가
    # 원본 라벨 데이터프레임 기준이 아니라 cap_per_class로 새로 뽑은 subset이라 별도 저장 필요)
    test_sub.drop(columns=["waferMap"]).to_json(
        MODEL_DIR / "lot_test_subset_meta.json", orient="records", force_ascii=False
    )
    with open(MODEL_DIR / "lot_test_subset.pkl", "wb") as f:
        pickle.dump(test_sub, f)

    result = {
        "note": "신뢰구간은 lot 단위 cluster bootstrap으로 test lot 재표본추출 변동을 "
                "반영합니다(재학습 시드 변동과 재수집 변동은 미포함). validation/test는 "
                "클래스당 최대 500장으로 제한한 subset이므로 Accuracy를 생산 분포의 정확도로 "
                "해석하면 안 됩니다. 이 수치는 final_evaluation.py의 무작위 분할 수치(0.885)와 "
                "다른 질문에 답합니다 — 신규 lot에서 패턴 분류 성능이 유지되는지를 평가합니다.",
        "bootstrap_unit": "lot",
        "split": {
            "train_lots": len(train_lots), "val_lots": len(val_lots), "test_lots": len(test_lots),
            "train_lot_range": [int(min(train_lots)), int(max(train_lots))],
            "val_lot_range": [int(min(val_lots)), int(max(val_lots))],
            "test_lot_range": [int(min(test_lots)), int(max(test_lots))],
            "train_subset_n": len(train_sub), "val_subset_n": len(val_sub), "test_subset_n": len(test_sub),
        },
        "stage1_val_selection": {
            "point_macro_f1": val_point,
            "lot_cluster_bootstrap_ci": val_ci,
            "best_config": best_name,
        },
        "stage2_final_test_evaluation": {
            "config": best_name,
            "macro_f1": round(float(rep["macro avg"]["f1-score"]), 4),
            "macro_f1_lot_cluster_bootstrap_ci": {
                "ci_lower": test_ci["ci_lower"], "ci_upper": test_ci["ci_upper"]
            },
            "accuracy": round(float(rep["accuracy"]), 4),
            "per_class_f1": {c: round(float(rep[c]["f1-score"]), 4) for c in DEFECT_CLASSES},
            "per_class_support": {c: int(rep[c]["support"]) for c in DEFECT_CLASSES},
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(ROOT / "results" / "lot_train_val_test_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["baseline", "derived", "augment", "both"]
    means = [val_point[n] for n in names]
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
    ax.bar(names, means, color=colors)
    ax.axhline(result["stage2_final_test_evaluation"]["macro_f1"], color="black", linestyle="--",
                label=f"최종 lot-test macro F1 ({best_name}, 1회 평가)")
    ax.set_ylabel("Macro F1")
    ax.set_title("lot 기준 Val 조합 선택 vs 최종 lot-test 평가 (1회) — WM-811K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "24_lot_train_val_test_evaluation.png", dpi=120)
    plt.close(fig)

    print("\n결과 저장 완료: results/lot_train_val_test_evaluation.json, "
          "figures/24_lot_train_val_test_evaluation.png")
    print("최종 모델 저장 완료: results/models/final_model.pt (lot 기반으로 갱신)")


if __name__ == "__main__":
    main()
