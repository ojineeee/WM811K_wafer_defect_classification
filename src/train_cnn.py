"""WM-811K 웨이퍼 결함 패턴 CNN 분류.

GPU가 없는 환경이므로 811,457장 전체가 아니라 클래스별 균형 잡힌
부분집합(subsample)으로 학습한다. 웨이퍼맵은 크기가 제각각이라
64x64로 최근접 이웃(nearest) 리사이즈해 크기를 통일한다.
"""
import json
import time
from pathlib import Path

from plot_style import plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from data import DEFECT_CLASSES, load_labeled

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
IMG_SIZE = 64
CAP_PER_CLASS = 2000
BATCH_SIZE = 64
EPOCHS = 15

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def resize_wafer(arr, size=IMG_SIZE):
    img = Image.fromarray(arr.astype(np.uint8))
    img = img.resize((size, size), resample=Image.NEAREST)
    return np.array(img, dtype=np.float32) / 2.0  # 0,1,2 -> 0,0.5,1.0


def build_balanced_subset(df, cap=CAP_PER_CLASS):
    parts = []
    for cls in DEFECT_CLASSES:
        sub = df[df["failureType"] == cls]
        if len(sub) > cap:
            sub = sub.sample(cap, random_state=RANDOM_STATE)
        parts.append(sub)
    out = __import__("pandas").concat(parts).reset_index(drop=True)
    return out


class WaferDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, H, W)
        y = self.labels[idx]
        return x, y


class SmallCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    t0 = time.time()
    df = load_labeled()
    subset = build_balanced_subset(df)
    print("서브샘플 클래스 분포:\n", subset["failureType"].value_counts())

    print("웨이퍼맵 리사이즈 중...")
    images = np.stack([resize_wafer(m) for m in subset["waferMap"]])
    label_to_idx = {c: i for i, c in enumerate(DEFECT_CLASSES)}
    labels = subset["failureType"].map(label_to_idx).values

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE
    )
    print(f"train={len(X_train)}, test={len(X_test)}  (리사이즈+분할까지 {time.time()-t0:.1f}s)")

    train_ds = WaferDataset(X_train, y_train)
    test_ds = WaferDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # 클래스 가중치: 서브샘플 후에도 남아있는 불균형(Near-full 149개 등) 보정
    class_counts = np.bincount(y_train, minlength=len(DEFECT_CLASSES))
    class_weights = torch.tensor(1.0 / np.sqrt(class_counts + 1), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(DEFECT_CLASSES)

    device = torch.device("cpu")
    model = SmallCNN(len(DEFECT_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {"epoch": [], "train_loss": [], "val_macro_f1": []}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb)
                preds.append(out.argmax(1).numpy())
                trues.append(yb.numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        val_f1 = f1_score(trues, preds, average="macro")

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_macro_f1"].append(val_f1)
        print(f"epoch {epoch:2d}/{EPOCHS}  loss={epoch_loss:.4f}  val_macro_f1={val_f1:.4f}  "
              f"elapsed={time.time()-t0:.1f}s")

    # 최종 평가
    report = classification_report(trues, preds, target_names=DEFECT_CLASSES, output_dict=True, zero_division=0)
    print("\n", classification_report(trues, preds, target_names=DEFECT_CLASSES, zero_division=0))

    cm = confusion_matrix(trues, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(DEFECT_CLASSES)))
    ax.set_yticks(range(len(DEFECT_CLASSES)))
    ax.set_xticklabels(DEFECT_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(DEFECT_CLASSES)
    for i in range(len(DEFECT_CLASSES)):
        for j in range(len(DEFECT_CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (held-out test)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(history["epoch"], history["train_loss"], color="#C44E52", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="#C44E52")
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["val_macro_f1"], color="#4C72B0", label="val macro F1")
    ax2.set_ylabel("val macro F1", color="#4C72B0")
    ax1.set_title("Training curve")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_training_curve.png", dpi=120)
    plt.close(fig)

    final_report = {
        "img_size": IMG_SIZE,
        "cap_per_class": CAP_PER_CLASS,
        "epochs": EPOCHS,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "accuracy": round(float(report["accuracy"]), 4),
        "per_class_f1": {c: round(float(report[c]["f1-score"]), 4) for c in DEFECT_CLASSES},
        "per_class_support": {c: int(report[c]["support"]) for c in DEFECT_CLASSES},
        "total_elapsed_sec": round(time.time() - t0, 1),
    }
    with open(ROOT / "results" / "cnn_final_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
