"""Load prepared train/validation/test arrays as PyTorch data loaders."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent.parent
BATCH_SIZE = 64


def _loader(split, shuffle=False):
    x_path, y_path = BASE_DIR / "data" / f"X_{split}.npy", BASE_DIR / "data" / f"y_{split}.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Prepared splits are missing. Run `python src/dataset_split.py` first.")
    x, y = np.load(x_path), np.load(y_path)
    if x.ndim != 4 or x.shape[1:] != (11, 11, 30) or y.ndim != 1 or len(x) != len(y) or len(y) == 0:
        raise ValueError(f"Malformed {split} split: X={x.shape}, y={y.shape}.")
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


train_loader = _loader("train", shuffle=True)
val_loader = _loader("val")
test_loader = _loader("test")
