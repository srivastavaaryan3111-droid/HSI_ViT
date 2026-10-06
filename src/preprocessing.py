import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
image_data = sio.loadmat("data/Indian_pines_corrected.mat")
ground_truth = sio.loadmat("data/Indian_pines_gt.mat")

hsi = image_data["indian_pines_corrected"]
gt = ground_truth["indian_pines_gt"]

print("Original HSI shape:", hsi.shape)

# Convert to float32
hsi = hsi.astype(np.float32)

# Get dimensions
H, W, B = hsi.shape

# Reshape:
# (145, 145, 200) -> (21025, 200)
hsi_2d = hsi.reshape(-1, B)

# Normalize
scaler = StandardScaler()
hsi_normalized = scaler.fit_transform(hsi_2d)

print("After normalization:", hsi_normalized.shape)

# PCA
n_components = 30

pca = PCA(n_components=n_components)

hsi_pca = pca.fit_transform(hsi_normalized)

# Convert back to image format
hsi_pca = hsi_pca.reshape(H, W, n_components)

print("After PCA:", hsi_pca.shape)

# Percentage of variance retained
variance = np.sum(pca.explained_variance_ratio_) * 100

print("Variance retained:", variance, "%")