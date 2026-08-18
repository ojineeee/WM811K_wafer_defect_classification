"""lot 순서(생산 순서 proxy)에 따른 불량 패턴 drift 분석.

WM-811K에는 SECOM 같은 타임스탬프가 없지만, lotName이 'lot1', 'lot2', ...
형태의 순번이라 생산 순서의 근사치로 쓸 수 있다. SECOM에서 시간 드리프트가
발견됐던 것과 동일한 위험이 여기에도 있는지 확인한다.
"""
import json
from pathlib import Path

from plot_style import plt
import numpy as np
import pandas as pd

from data import DEFECT_CLASSES, load_labeled

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"


def main():
    df = load_labeled()
    df["lot_num"] = df["lotName"].str.extract(r"lot(\d+)").astype(int)
    df = df.sort_values("lot_num").reset_index(drop=True)

    df["decile"] = pd.qcut(df["lot_num"], 10, labels=False, duplicates="drop")
    defect_rate_by_decile = df.groupby("decile").apply(lambda g: (g["failureType"] != "none").mean())

    summary = {
        "n_unique_lots": int(df["lot_num"].nunique()),
        "lot_num_range": [int(df["lot_num"].min()), int(df["lot_num"].max())],
        "defect_rate_by_decile": {int(k): round(float(v), 4) for k, v in defect_rate_by_decile.items()},
        "defect_rate_decile0_1_avg": round(float(defect_rate_by_decile.iloc[:2].mean()), 4),
        "defect_rate_decile2_9_avg": round(float(defect_rate_by_decile.iloc[2:].mean()), 4),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    defect_rate_by_decile.plot(kind="bar", ax=ax, color="#C44E52")
    ax.set_xlabel("lot order decile (0=earliest, 9=latest)")
    ax.set_ylabel("defect rate (non-'none' ratio)")
    ax.set_title("Defect rate by lot order — WM-811K")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "16_lot_drift.png", dpi=120)
    plt.close(fig)

    with open(ROOT / "results" / "lot_drift_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
