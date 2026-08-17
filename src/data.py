"""WM-811K 원본(.pkl) 로드 + 라벨 정제."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PKL_PATH = ROOT / "data" / "raw" / "extracted" / "MIR-WM811K" / "Python" / "WM811K.pkl"

DEFECT_CLASSES = [
    "none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Random", "Scratch", "Near-full",
]


def load_raw():
    df = pd.read_pickle(PKL_PATH)
    return df


def load_labeled():
    """failureType이 실제로 라벨링된(9개 클래스 중 하나) 행만 반환.

    원본 pkl에서 라벨이 없는 행은 문자열 '0'으로 채워져 있어 정수 0과
    혼동하기 쉬우므로 명시적으로 걸러낸다.
    """
    df = load_raw()
    labeled = df[df["failureType"].isin(DEFECT_CLASSES)].copy()
    labeled["failureType"] = labeled["failureType"].astype(str)
    return labeled.reset_index(drop=True)


if __name__ == "__main__":
    df = load_labeled()
    print("라벨링된 웨이퍼 수:", len(df))
    print(df["failureType"].value_counts())
