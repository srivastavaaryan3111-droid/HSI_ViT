from pathlib import Path
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent

image_data = sio.loadmat(
    BASE_DIR / "data" / "Indian_pines_corrected.mat"
)

ground_truth = sio.loadmat(
    BASE_DIR / "data" / "Indian_pines_gt.mat"
)

hsi = image_data["indian_pines_corrected"]
gt = ground_truth["indian_pines_gt"]

hsi = hsi.astype(np.float32)

H, W, B = hsi.shape


# -----------------------------
# 2. Normalize
# -----------------------------

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


# -----------------------------
# 4. Patch extraction
# -----------------------------

patch_size = 11
margin = patch_size // 2

padded_hsi = np.pad(
    hsi_pca,
    ((margin, margin), (margin, margin), (0, 0)),
    mode="reflect"
)

patches = []
labels = []

for row in range(H):
    for col in range(W):

        if gt[row, col] == 0:
            continue

        patch = padded_hsi[
            row:row + patch_size,
            col:col + patch_size,
            :
        ]

        patches.append(patch)
        labels.append(gt[row, col])


patches = np.array(patches)
labels = np.array(labels)


print("Total patches:", len(patches))
print("Patch shape:", patches.shape)
print("Labels shape:", labels.shape)


# -----------------------------
# 5. Convert labels to 0-based
# -----------------------------

labels = labels - 1

X_train, X_test, y_train, y_test = train_test_split(
    patches,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nNumber of classes:", len(np.unique(labels)))
# -----------------------------
# 6. Save split data
# -----------------------------

np.save(BASE_DIR / "data" / "X_train.npy", X_train)
np.save(BASE_DIR / "data" / "X_test.npy", X_test)
np.save(BASE_DIR / "data" / "y_train.npy", y_train)
np.save(BASE_DIR / "data" / "y_test.npy", y_test)

print("\nTrain/test data saved successfully.")