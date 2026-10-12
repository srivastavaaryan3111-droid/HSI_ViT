"""Evaluate the validation-selected model on the held-out test set."""
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from pytorch_dataset import test_loader
from vit_model import HSI_ViT


BASE_DIR = Path(__file__).resolve().parent.parent
NUM_CLASSES = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    model_path = BASE_DIR / "models" / "hsi_vit_v2.pth"
    if not model_path.exists():
        raise FileNotFoundError("Model checkpoint not found. Run `python src/train.py` first.")
    model = HSI_ViT().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Unsupported checkpoint. Rebuild splits and train the corrected v2 pipeline.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    actual, predicted = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            predicted.extend(logits.argmax(1).cpu().tolist())
            actual.extend(labels.tolist())

    labels = list(range(NUM_CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0
    )
    print("Test results (model selected by validation loss):")
    print(f"Accuracy: {accuracy_score(actual, predicted):.4f}")
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(actual, predicted, average=average, zero_division=0)
        print(f"{average.title()} precision: {p:.4f} | recall: {r:.4f} | F1: {f:.4f}")

    print("\nPer-class results (class IDs are zero-based in the arrays):")
    print(classification_report(actual, predicted, labels=labels, zero_division=0, digits=4))
    print("Confusion matrix (rows=true, columns=predicted; class IDs 0-15):")
    print(confusion_matrix(actual, predicted, labels=labels))
    print("\nClass imbalance in test split:")
    for class_id, count, p, r, f in zip(labels, support, precision, recall, f1):
        print(f"Class {class_id + 1:2d}: support={count:4d} precision={p:.4f} recall={r:.4f} F1={f:.4f}")


if __name__ == "__main__":
    main()
