"""저장된 lot 최종 모델로 test Macro F1과 lot-cluster CI만 다시 계산한다.

모델을 재학습하지 않는다. ``lot_final_evaluation.py``가 저장한 모델, 스케일러,
test subset을 불러와 추론하고 ``results/lot_train_val_test_evaluation.json``의
최종 test 평가 결과만 갱신한다.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report

from data import DEFECT_CLASSES
from derived_features import FEATURE_NAMES, HybridCNN
from lot_final_evaluation import lot_cluster_bootstrap_macro_f1, make_xyf, predict

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "results" / "models"
RESULT_PATH = ROOT / "results" / "lot_train_val_test_evaluation.json"


def main():
    with open(MODEL_DIR / "final_test_indices.json", encoding="utf-8") as f:
        model_meta = json.load(f)
    if model_meta.get("split_type") != "lot_based":
        raise ValueError("저장된 final_model이 lot 기반 최종 모델이 아닙니다.")

    with open(MODEL_DIR / "lot_test_subset.pkl", "rb") as f:
        test_sub = pickle.load(f)
    with open(MODEL_DIR / "final_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    test_maps = list(test_sub["waferMap"])
    test_labels = list(test_sub["failureType"])
    X_test, F_test_raw, y_test = make_xyf(
        test_maps,
        test_labels,
        zero_feats=not model_meta["use_feats"],
    )
    if model_meta["use_feats"]:
        if scaler is None:
            raise ValueError("파생변수 모델인데 저장된 scaler가 없습니다.")
        F_test = scaler.transform(F_test_raw).astype(np.float32)
    else:
        F_test = np.zeros((len(X_test), len(FEATURE_NAMES)), dtype=np.float32)

    model = HybridCNN(
        len(DEFECT_CLASSES),
        len(FEATURE_NAMES),
        use_feats=model_meta["use_feats"],
    )
    state_dict = torch.load(
        MODEL_DIR / "final_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    test_preds = predict(model, X_test, F_test)

    report = classification_report(
        y_test,
        test_preds,
        labels=np.arange(len(DEFECT_CLASSES)),
        target_names=DEFECT_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    test_ci = lot_cluster_bootstrap_macro_f1(
        y_test,
        {model_meta["best_config"]: test_preds},
        test_sub["lot_num"].to_numpy(),
        seed=43,
    )[model_meta["best_config"]]

    with open(RESULT_PATH, encoding="utf-8") as f:
        result = json.load(f)

    old_point = result["stage2_final_test_evaluation"].get("macro_f1")
    new_point = round(float(report["macro avg"]["f1-score"]), 4)
    if old_point is not None and not np.isclose(old_point, new_point, atol=1e-4):
        raise ValueError(
            f"저장된 점 추정치({old_point})와 재추론 결과({new_point})가 다릅니다. "
            "모델과 test subset 조합을 확인하세요."
        )

    result["bootstrap_unit"] = "lot"
    result["note"] = (
        "최종 test 신뢰구간은 저장된 최종 모델을 재학습하지 않고 test lot을 "
        "재표본추출하는 lot-cluster bootstrap으로 계산했습니다. 재학습 시드 변동, "
        "재수집 변동, 클래스 상한 적용 전 생산 분포는 반영하지 않습니다."
    )
    stage1 = result["stage1_val_selection"]
    stage1.pop("bootstrap_ci", None)
    stage1["ci_note"] = (
        "기존 wafer bootstrap CI는 제거했습니다. validation의 lot-cluster CI는 "
        "전체 모델 선택 단계를 다시 실행할 때 생성됩니다."
    )
    result["stage2_final_test_evaluation"] = {
        "config": model_meta["best_config"],
        "macro_f1": new_point,
        "macro_f1_lot_cluster_bootstrap_ci": {
            "ci_lower": test_ci["ci_lower"],
            "ci_upper": test_ci["ci_upper"],
        },
        "accuracy": round(float(report["accuracy"]), 4),
        "per_class_f1": {
            c: round(float(report[c]["f1-score"]), 4) for c in DEFECT_CLASSES
        },
        "per_class_support": {
            c: int(report[c]["support"]) for c in DEFECT_CLASSES
        },
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"lot-test Macro F1: {new_point:.4f} "
        f"(lot-cluster 95% CI {test_ci['ci_lower']:.4f}~{test_ci['ci_upper']:.4f})"
    )
    print(f"결과 갱신 완료: {RESULT_PATH}")


if __name__ == "__main__":
    main()
