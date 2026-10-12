"""Dataset inspection entry point using shared validation."""
from preprocessing import load_indian_pines
import numpy as np


def main():
    image, labels = load_indian_pines()
    classes, counts = np.unique(labels, return_counts=True)
    print("HSI shape:", image.shape, "dtype:", image.dtype)
    print("Ground truth shape:", labels.shape)
    print("Class counts:", dict(zip(classes.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()
