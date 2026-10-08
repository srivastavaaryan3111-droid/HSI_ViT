from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


# -----------------------------
# 1. Project directory
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


# -----------------------------
# 2. Load train/test data
# -----------------------------

X_train = np.load(BASE_DIR / "data" / "X_train.npy")
X_test = np.load(BASE_DIR / "data" / "X_test.npy")

y_train = np.load(BASE_DIR / "data" / "y_train.npy")
y_test = np.load(BASE_DIR / "data" / "y_test.npy")


print("Loaded data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# -----------------------------
# 3. Convert NumPy → PyTorch
# -----------------------------

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


# -----------------------------
# 4. Create PyTorch datasets
# -----------------------------

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)


# -----------------------------
# 5. Create DataLoaders
# -----------------------------

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# -----------------------------
# 6. Test one batch
# -----------------------------

images, labels = next(iter(train_loader))

print("\nDataLoader test:")
print("Batch images:", images.shape)
print("Batch labels:", labels.shape)
print("First 10 labels:", labels[:10])