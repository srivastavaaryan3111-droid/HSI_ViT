from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_dataset import train_loader
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
# 3. Loss function
# -----------------------------

criterion = nn.CrossEntropyLoss()


# -----------------------------
# 4. Optimizer
# -----------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------
# 5. Training
# -----------------------------

epochs = 1

model.train()

for epoch in range(epochs):

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )


# -----------------------------
# 6. Save model
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR / "models" / "hsi_vit.pth"

torch.save(model.state_dict(), model_path)

print("\nModel saved to:", model_path)