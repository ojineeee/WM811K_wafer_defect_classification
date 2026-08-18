"""Train/Validation/Test 3-way 분리로 "어느 조합이 최선인가"와 "최종 성능이
얼마인가"를 분리해서 답한다.

기존 ablation_with_ci.py는 baseline/derived/augment/both 네 조합을 전부
같은 test set(20%)에서 비교하고, 그 test set의 macro F1로 "어느 조합이
최선인지" 결정과 "최종 성능이 얼마인지" 보고를 동시에 했다. 조합을 4개나
비교하고 그중 가장 높은 걸 골라 그대로 "최종 성능"이라고 보고하면, test set이
사실상 모델 선택에도 쓰인 셈이 되어 보고된 숫자가 낙관적으로 편향될 수 있다
(다중 비교로 인한 선택 편향 — winner's curse).

이 스크립트는 train(60%)/val(20%)/test(20%)로 분리해서:
1. 4개 조합 비교와 부트스트랩 신뢰구간은 val에서만 수행해 "어느 조합이
   최선인지"를 결정한다 (test는 이 단계에서 전혀 보지 않는다).
2. 최선으로 확인된 조합만 train+val로 재학습시켜, test에서 딱 한 번 평가한다.
   이 test 점수만 "최종 성능"으로 보고한다.
3. 최종 모델과 스케일러를 저장해 grad_cam.py가 재학습 없이 재사용하게 한다.

주의(부트스트랩 CI의 한계): 아래 신뢰구간은 "이 test set을 다시 뽑았다면
얼마나 흔들렸을까"만 반영한다. 다른 랜덤 시드로 모델을 다시 학습했을 때의
변동이나, 데이터를 다시 수집했을 때의 변동까지는 포함하지 않는다.
"""
import json
import pickle
import time
from pathlib import Path

from plot_style import plt
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data import DEFECT_CLASSES, load_labeled
from derived_features import FEATURE_NAMES, HybridCNN, HybridDataset, extract_features
from train_cnn import BATCH_SIZE, EPOCHS, RANDOM_STATE, build_balanced_subset, resize_wafer
from augmentation import AUGMENT_TARGETS, augment_variants

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
MODEL_DIR = ROOT / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000
RNG = np.random.default_rng(RANDOM_STATE)

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


def bootstrap_macro_f1(y_true, preds_dict, n=N_BOOTSTRAP):
    n_samples = len(y_true)
    scores = {k: [] for k in preds_dict}
    for _ in range(n):
        idx = RNG.integers(0, n_samples, n_samples)
        yt = y_true[idx]
        for k, preds in preds_dict.items():
            scores[k].append(f1_score(yt, preds[idx], average="macro"))
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
    subset = build_balanced_subset(df)
    labels_arr = subset["failureType"].values

    idx = np.arange(len(subset))
    tr_idx, rest_idx = train_test_split(idx, test_size=0.4, stratify=labels_arr, random_state=RANDOM_STATE)
    val_idx, te_idx = train_test_split(
        rest_idx, test_size=0.5, stratify=labels_arr[rest_idx], random_state=RANDOM_STATE
    )
    print(f"train={len(tr_idx)}  val={len(val_idx)}  test={len(te_idx)}")

    # ---------- 1단계: val에서 4개 조합 비교 (모델 선택 전용, test는 아직 보지 않음) ----------
    val_maps = [subset["waferMap"].iloc[i] for i in val_idx]
    val_labels = [subset["failureType"].iloc[i] for i in val_idx]
    X_val, F_val_raw, y_val = make_xyf(val_maps, val_labels)

    base_maps = [subset["waferMap"].iloc[i] for i in tr_idx]
    base_labels = [subset["failureType"].iloc[i] for i in tr_idx]
    aug_maps, aug_labels = build_augmented(subset, tr_idx)
    print(f"[1단계] base train={len(base_maps)}  augmented train={len(aug_maps)}")

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

    val_ci = bootstrap_macro_f1(y_val, val_preds)
    best_name = max(val_point, key=val_point.get)
    print(f"\nval 기준 최선 조합: {best_name} (macro_f1={val_point[best_name]})")

    # ---------- 2단계: 최선 조합만 train+val로 재학습, test에서 딱 한 번 평가 ----------
    trval_idx = np.concatenate([tr_idx, val_idx])
    use_feats_final = best_name in ("derived", "both")
    use_augment_final = best_name in ("augment", "both")

    if use_augment_final:
        final_maps, final_labels = build_augmented(subset, trval_idx)
    else:
        final_maps = [subset["waferMap"].iloc[i] for i in trval_idx]
        final_labels = [subset["failureType"].iloc[i] for i in trval_idx]

    test_maps = [subset["waferMap"].iloc[i] for i in te_idx]
    test_labels = [subset["failureType"].iloc[i] for i in te_idx]
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
    print("\n=== 최종 test 평가 (단 한 번) ===")
    print(classification_report(y_te, test_preds, target_names=DEFECT_CLASSES, zero_division=0))

    test_ci = bootstrap_macro_f1(y_te, {best_name: test_preds})[best_name]

    # 최종 모델/스케일러 저장 (grad_cam.py 재사용 목적 — 재학습 없이 동일 모델·test set 재현)
    torch.save(final_model.state_dict(), MODEL_DIR / "final_model.pt")
    with open(MODEL_DIR / "final_scaler.pkl", "wb") as f:
        pickle.dump(final_scaler, f)
    with open(MODEL_DIR / "final_test_indices.json", "w", encoding="utf-8") as f:
        json.dump({
            "te_idx": [int(i) for i in te_idx],
            "best_config": best_name,
            "use_feats": use_feats_final,
        }, f)

    result = {
        "note": "부트스트랩 CI는 테스트셋 재표본추출 변동만 반영합니다 (재학습 시드 변동, 재수집 변동은 미포함).",
        "split": {"train": len(tr_idx), "val": len(val_idx), "test": len(te_idx)},
        "stage1_val_selection": {
            "point_macro_f1": val_point,
            "bootstrap_ci": val_ci,
            "best_config": best_name,
        },
        "stage2_final_test_evaluation": {
            "config": best_name,
            "macro_f1": round(float(rep["macro avg"]["f1-score"]), 4),
            "macro_f1_bootstrap_ci": {"ci_lower": test_ci["ci_lower"], "ci_upper": test_ci["ci_upper"]},
            "accuracy": round(float(rep["accuracy"]), 4),
            "per_class_f1": {c: round(float(rep[c]["f1-score"]), 4) for c in DEFECT_CLASSES},
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(ROOT / "results" / "train_val_test_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["baseline", "derived", "augment", "both"]
    means = [val_ci[n]["mean"] for n in names]
    lowers = [val_ci[n]["mean"] - val_ci[n]["ci_lower"] for n in names]
    uppers = [val_ci[n]["ci_upper"] - val_ci[n]["mean"] for n in names]
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
    ax.bar(names, means, yerr=[lowers, uppers], capsize=5, color=colors)
    ax.axhline(result["stage2_final_test_evaluation"]["macro_f1"], color="black", linestyle="--",
                label=f"최종 test macro F1 ({best_name}, 1회 평가)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Val 기준 조합 선택 vs 최종 test 평가 (1회) — WM-811K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "23_train_val_test_evaluation.png", dpi=120)
    plt.close(fig)

    print("\n결과 저장 완료: results/train_val_test_evaluation.json, figures/23_train_val_test_evaluation.png")
    print("최종 모델 저장 완료: results/models/final_model.pt")


if __name__ == "__main__":
    main()
