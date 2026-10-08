from pathlib import Path
import torch
import torch.nn as nn

from pytorch_dataset import test_loader
from vit_model import HSI_ViT


# -----------------------------
# 1. Device
# -----------------------------

device = torch.device("cpu")

print("Using device:", device)


# -----------------------------
# 2. Create model
# -----------------------------

model = HSI_ViT(
    input_channels=30,
    patch_size=11,
    num_classes=16,
    embed_dim=64,
    num_heads=4,
    num_layers=2
)

model = model.to(device)


# -----------------------------
# 3. Load trained model
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR / "models" / "hsi_vit.pth"

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

print("Model loaded successfully.")


# -----------------------------
# 4. Evaluation
# -----------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


# -----------------------------
# 5. Accuracy
# -----------------------------

accuracy = 100 * correct / total

print("\nTest Results:")
print("Correct predictions:", correct)
print("Total samples:", total)
print(f"Test Accuracy: {accuracy:.2f}%")