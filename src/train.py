"""Train the HSI ViT and select its checkpoint using validation loss."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import joblib

from pytorch_dataset import train_loader, val_loader
from vit_model import HSI_ViT


BASE_DIR = Path(__file__).resolve().parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
SEED = 42
MODEL_CONFIG = {"input_channels": 30, "patch_size": 11, "num_classes": 16,
                "embed_dim": 64, "num_heads": 4, "num_layers": 2, "dropout": 0.1}


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    from preprocessing import PREPROCESSOR_PATH, load_preprocessor
    preprocessor = load_preprocessor()
    model = HSI_ViT().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model_path = BASE_DIR / "models" / "hsi_vit_v2.pth"
    best_val_loss = float("inf")

    print(f"Using device: {device}")
    print(f"Training for up to {EPOCHS} epochs; checkpoint selected on validation loss.")
    for epoch in range(EPOCHS):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] train loss={train_loss:.4f} accuracy={train_accuracy:.2%} | val loss={val_loss:.4f} accuracy={val_accuracy:.2%}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {"format_version": 2, "model_state_dict": model.state_dict(),
                          "model_config": MODEL_CONFIG, "preprocessor_path": str(PREPROCESSOR_PATH.name),
                          "class_mapping": {str(i + 1): i for i in range(16)},
                          "preprocessing_config": {"pca_components": 30, "patch_size": 11, "padding": "reflect"},
                          "training_config": {"epochs": EPOCHS, "learning_rate": 0.001, "batch_size": 64,
                                              "optimizer": "Adam", "loss": "CrossEntropyLoss"},
                          "random_seed": SEED, "best_validation_loss": best_val_loss}
            temporary = model_path.with_suffix(".pth.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(model_path)
            print(f"  Saved best checkpoint to {model_path}")


if __name__ == "__main__":
    main()
