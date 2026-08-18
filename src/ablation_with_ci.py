"""파생변수 x 증강 2x2 완전 조합 + 부트스트랩 신뢰구간.

지금까지 derived_features.py, augmentation.py를 따로 실행하면서
(1) "증강만 단독" 조합이 빠져 있었고 (2) 서로 다른 프로세스 실행이라
CPU 스레드 비결정성으로 숫자가 미세하게 어긋났었다. 이 스크립트는
네 조합(베이스라인 / +파생변수 / +증강 / +파생변수+증강)을 동일한
프로세스·동일한 train/test 분할에서 전부 다시 학습시켜 완전히
일관된 숫자를 얻고, 부트스트랩으로 개선폭의 신뢰구간까지 계산한다.
"""
import json
import time
from pathlib import Path

from plot_style import plt
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
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


def train_and_predict(X_tr, F_tr, y_tr, X_te, F_te, use_feats):
    train_loader = DataLoader(HybridDataset(X_tr, F_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    te_ds = HybridDataset(X_te, F_te, np.zeros(len(X_te), dtype=int))  # 라벨 미사용, 형식 맞추기용
    test_loader = DataLoader(te_ds, batch_size=256, shuffle=False)

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
    preds = []
    with torch.no_grad():
        for xb, fb, _ in test_loader:
            preds.append(model(xb, fb).argmax(1).numpy())
    return np.concatenate(preds)


def bootstrap_macro_f1(y_true, preds_dict, n=N_BOOTSTRAP):
    """조건별 macro F1 분포 + 두 조건 간 델타의 신뢰구간(paired bootstrap)."""
    n_samples = len(y_true)
    scores = {k: [] for k in preds_dict}
    deltas = {}
    pairs = [("baseline", "derived"), ("baseline", "augment"), ("baseline", "both"),
             ("derived", "both"), ("augment", "both")]
    delta_samples = {p: [] for p in pairs}

    for _ in range(n):
        idx = RNG.integers(0, n_samples, n_samples)
        yt = y_true[idx]
        fold_scores = {}
        for k, preds in preds_dict.items():
            fold_scores[k] = f1_score(yt, preds[idx], average="macro")
            scores[k].append(fold_scores[k])
        for a, b in pairs:
            delta_samples[(a, b)].append(fold_scores[b] - fold_scores[a])

    result = {}
    for k, v in scores.items():
        result[k] = {
            "mean": round(float(np.mean(v)), 4),
            "ci_lower": round(float(np.percentile(v, 2.5)), 4),
            "ci_upper": round(float(np.percentile(v, 97.5)), 4),
        }
    delta_result = {}
    for (a, b), v in delta_samples.items():
        delta_result[f"{b}_minus_{a}"] = {
            "mean_delta": round(float(np.mean(v)), 4),
            "ci_lower": round(float(np.percentile(v, 2.5)), 4),
            "ci_upper": round(float(np.percentile(v, 97.5)), 4),
            "significant": bool(np.percentile(v, 2.5) > 0 or np.percentile(v, 97.5) < 0),
        }
    return result, delta_result


def main():
    t0 = time.time()
    df = load_labeled()
    subset = build_balanced_subset(df)
    labels_arr = subset["failureType"].values

    idx = np.arange(len(subset))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=labels_arr, random_state=RANDOM_STATE)

    test_maps = [subset["waferMap"].iloc[i] for i in te_idx]
    test_labels = [subset["failureType"].iloc[i] for i in te_idx]
    X_te, F_te_raw, y_te = make_xyf(test_maps, test_labels)

    base_maps = [subset["waferMap"].iloc[i] for i in tr_idx]
    base_labels = [subset["failureType"].iloc[i] for i in tr_idx]
    aug_maps, aug_labels = build_augmented(subset, tr_idx)

    print(f"base train={len(base_maps)}  augmented train={len(aug_maps)}  test={len(test_maps)}")

    conditions = {
        "baseline": (base_maps, base_labels, False),
        "derived": (base_maps, base_labels, True),
        "augment": (aug_maps, aug_labels, False),
        "both": (aug_maps, aug_labels, True),
    }

    preds_dict = {}
    point_metrics = {}
    for name, (maps, labels_str, use_feats) in conditions.items():
        X_tr, F_tr_raw, y_tr = make_xyf(maps, labels_str, zero_feats=not use_feats)
        if use_feats:
            scaler = StandardScaler().fit(F_tr_raw)
            F_tr = scaler.transform(F_tr_raw).astype(np.float32)
            F_te = scaler.transform(F_te_raw).astype(np.float32)
        else:
            F_tr, F_te = F_tr_raw, np.zeros((len(X_te), len(FEATURE_NAMES)), dtype=np.float32)

        preds = train_and_predict(X_tr, F_tr, y_tr, X_te, F_te, use_feats)
        preds_dict[name] = preds
        mf1 = f1_score(y_te, preds, average="macro")
        point_metrics[name] = round(float(mf1), 4)
        print(f"[{name}] train={len(maps)}  use_feats={use_feats}  macro_f1={mf1:.4f}  "
              f"elapsed={time.time()-t0:.1f}s")

    ci_scores, ci_deltas = bootstrap_macro_f1(y_te, preds_dict)

    print("\n=== 부트스트랩 결과 (macro F1, 95% CI) ===")
    for k, v in ci_scores.items():
        print(f"  {k}: {v['mean']} ({v['ci_lower']} ~ {v['ci_upper']})")
    print("\n=== 개선폭 델타 (95% CI, significant=CI가 0을 포함하지 않음) ===")
    for k, v in ci_deltas.items():
        print(f"  {k}: {v['mean_delta']:+.4f} ({v['ci_lower']:+.4f} ~ {v['ci_upper']:+.4f})  significant={v['significant']}")

    result = {
        "point_macro_f1": point_metrics,
        "bootstrap_ci": ci_scores,
        "delta_ci": ci_deltas,
        "n_test": int(len(y_te)),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(ROOT / "results" / "ablation_2x2_with_ci.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["baseline", "derived", "augment", "both"]
    means = [ci_scores[n]["mean"] for n in names]
    lowers = [ci_scores[n]["mean"] - ci_scores[n]["ci_lower"] for n in names]
    uppers = [ci_scores[n]["ci_upper"] - ci_scores[n]["mean"] for n in names]
    ax.bar(names, means, yerr=[lowers, uppers], capsize=5, color=["#4C72B0", "#55A868", "#DD8452", "#C44E52"])
    ax.set_ylabel("Macro F1 (95% bootstrap CI)")
    ax.set_title("2x2 Ablation: derived features x augmentation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "19_ablation_2x2_ci.png", dpi=120)
    plt.close(fig)

    print("\n결과 저장 완료: results/ablation_2x2_with_ci.json, figures/19_ablation_2x2_ci.png")


if __name__ == "__main__":
    main()
