import scipy.io as sio

# Load .mat files
image_data = sio.loadmat("data/Indian_pines_corrected.mat")
ground_truth = sio.loadmat("data/Indian_pines_gt.mat")

# Extract actual arrays
hsi = image_data["indian_pines_corrected"]
gt = ground_truth["indian_pines_gt"]

# Print information
print("HSI shape:", hsi.shape)
print("Ground truth shape:", gt.shape)

print("HSI data type:", hsi.dtype)
print("Ground truth data type:", gt.dtype)
import numpy as np

# Find all unique class labels
classes, counts = np.unique(gt, return_counts=True)

print("\nClasses and number of pixels:")

for cls, count in zip(classes, counts):
    print(f"Class {cls}: {count} pixels")