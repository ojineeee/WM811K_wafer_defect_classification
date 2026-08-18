"""WM-811K EDA: 클래스 분포, 웨이퍼맵 크기 분포, 클래스별 샘플 시각화."""
import json
from pathlib import Path

from plot_style import plt
import numpy as np

from data import DEFECT_CLASSES, load_labeled, load_raw

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    raw = load_raw()
    df = load_labeled()
    summary = {}

    summary["n_total_wafers"] = int(len(raw))
    summary["n_labeled_wafers"] = int(len(df))
    summary["label_ratio_pct"] = round(100 * len(df) / len(raw), 2)

    counts = df["failureType"].value_counts().reindex(DEFECT_CLASSES)
    summary["class_counts"] = {k: int(v) for k, v in counts.items()}
    summary["defect_only_total"] = int(counts.drop("none").sum())
    summary["most_rare_class"] = counts.drop("none").idxmin()
    summary["most_rare_count"] = int(counts.drop("none").min())
    summary["imbalance_ratio_none_vs_rarest"] = round(
        counts["none"] / counts.drop("none").min(), 1
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_yscale("log")
    ax.set_title("Class distribution (labeled subset, log scale)")
    ax.set_ylabel("count (log)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_class_distribution.png", dpi=120)
    plt.close(fig)

    # 웨이퍼맵 크기 분포
    shapes = df["waferMap"].apply(lambda m: m.shape)
    heights = shapes.apply(lambda s: s[0])
    widths = shapes.apply(lambda s: s[1])
    summary["wafer_size_height_range"] = [int(heights.min()), int(heights.max())]
    summary["wafer_size_width_range"] = [int(widths.min()), int(widths.max())]
    summary["n_unique_shapes"] = int(shapes.nunique())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(widths, heights, s=3, alpha=0.15, color="#55A868")
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    ax.set_title(f"Wafer map size distribution ({shapes.nunique()} unique shapes)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_wafer_size_scatter.png", dpi=120)
    plt.close(fig)

    # 클래스별 샘플 웨이퍼맵 시각화 (3x3)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    cmap = matplotlib.colors.ListedColormap(["white", "#a6c8ff", "#c44e52"])
    for ax, cls in zip(axes.flat, DEFECT_CLASSES):
        sample = df[df["failureType"] == cls]["waferMap"].iloc[0]
        ax.imshow(sample, cmap=cmap, vmin=0, vmax=2)
        ax.set_title(f"{cls} (n={int(counts[cls])})", fontsize=10)
        ax.axis("off")
    fig.suptitle("Sample wafer map per class (blue=normal die, red=defect die)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_class_samples.png", dpi=120)
    plt.close(fig)

    # dieSize (웨이퍼당 다이 개수) 분포 — 클래스별 차이가 있는지
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in ["none", "Center", "Scratch", "Near-full"]:
        vals = df.loc[df["failureType"] == cls, "dieSize"]
        ax.hist(vals, bins=40, alpha=0.5, label=cls, density=True)
    ax.set_xlabel("dieSize")
    ax.set_ylabel("density")
    ax.set_title("dieSize distribution by class (selected)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_diesize_by_class.png", dpi=120)
    plt.close(fig)

    with open(ROOT / "results" / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
