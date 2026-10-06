import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# -----------------------------
# 1. Load data
# -----------------------------

image_data = sio.loadmat("data/Indian_pines_corrected.mat")
ground_truth = sio.loadmat("data/Indian_pines_gt.mat")

hsi = image_data["indian_pines_corrected"]
gt = ground_truth["indian_pines_gt"]

hsi = hsi.astype(np.float32)

print("Original HSI:", hsi.shape)


# -----------------------------
# 2. Normalize
# -----------------------------

H, W, B = hsi.shape

hsi_2d = hsi.reshape(-1, B)

scaler = StandardScaler()
hsi_normalized = scaler.fit_transform(hsi_2d)


# -----------------------------
# 3. PCA
# -----------------------------

n_components = 30

pca = PCA(n_components=n_components)

hsi_pca = pca.fit_transform(hsi_normalized)

hsi_pca = hsi_pca.reshape(H, W, n_components)

print("After PCA:", hsi_pca.shape)


# -----------------------------
# 4. Pad the image
# -----------------------------

patch_size = 11
margin = patch_size // 2

padded_hsi = np.pad(
    hsi_pca,
    ((margin, margin), (margin, margin), (0, 0)),
    mode="reflect"
)

print("Padded HSI:", padded_hsi.shape)


# -----------------------------
# 5. Extract patches
# -----------------------------

patches = []
labels = []

for row in range(H):
    for col in range(W):

        # Ignore background
        if gt[row, col] == 0:
            continue

        # Extract 11 × 11 × 30 patch
        patch = padded_hsi[
            row:row + patch_size,
            col:col + patch_size,
            :
        ]

        patches.append(patch)
        labels.append(gt[row, col])


# Convert to NumPy arrays
patches = np.array(patches)
labels = np.array(labels)


# -----------------------------
# 6. Print results
# -----------------------------

print("Patches shape:", patches.shape)
print("Labels shape:", labels.shape)
print("First patch shape:", patches[0].shape)
print("First label:", labels[0])