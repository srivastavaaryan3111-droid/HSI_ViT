"""Shared, leakage-safe preprocessing and artifact helpers for HSI_ViT."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "data" / "Indian_pines_corrected.mat"
LABEL_PATH = ROOT / "data" / "Indian_pines_gt.mat"
PREPROCESSOR_PATH = ROOT / "models" / "hsi_preprocessor.joblib"
PATCH_SIZE = 11
N_COMPONENTS = 30
RANDOM_SEED = 42


def load_indian_pines(image_path: Path = IMAGE_PATH, label_path: Path = LABEL_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the expected Indian Pines cube and label map."""
    for path in (image_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset file not found: {path}")
    image_data, label_data = sio.loadmat(image_path), sio.loadmat(label_path)
    if "indian_pines_corrected" not in image_data or "indian_pines_gt" not in label_data:
        raise ValueError("MAT files must contain indian_pines_corrected and indian_pines_gt variables.")
    image = np.asarray(image_data["indian_pines_corrected"], dtype=np.float32)
    labels = np.asarray(label_data["indian_pines_gt"])
    if image.ndim != 3 or labels.ndim != 2 or image.shape[:2] != labels.shape:
        raise ValueError(f"Expected HxWxB image and matching HxW labels; got {image.shape} and {labels.shape}.")
    if image.shape[2] < N_COMPONENTS or not np.isfinite(image).all():
        raise ValueError(f"Image needs at least {N_COMPONENTS} bands and only finite values.")
    if not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() > 16:
        raise ValueError("Ground-truth labels must be integers in the range 0..16.")
    if np.count_nonzero(labels) == 0:
        raise ValueError("Ground truth contains no labeled pixels.")
    return image, labels.astype(np.int64)


def fit_preprocessor(image: np.ndarray, train_coordinates: np.ndarray, n_components: int = N_COMPONENTS) -> dict[str, Any]:
    """Fit scaler and PCA on training-center spectra only."""
    if image.ndim != 3 or n_components <= 0 or n_components > min(len(train_coordinates), image.shape[2]):
        raise ValueError("PCA component count must be positive and no greater than training samples or bands.")
    spectra = image[train_coordinates[:, 0], train_coordinates[:, 1], :]
    scaler = StandardScaler().fit(spectra)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED).fit(scaler.transform(spectra))
    return {"scaler": scaler, "pca": pca, "input_bands": image.shape[2], "n_components": n_components,
            "patch_size": PATCH_SIZE, "padding": "reflect", "random_seed": RANDOM_SEED}


def transform_image(image: np.ndarray, artifact: dict[str, Any]) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != artifact["input_bands"]:
        raise ValueError(f"Expected an HxWx{artifact['input_bands']} image, got {image.shape}.")
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinite values.")
    h, w, bands = image.shape
    scaled = artifact["scaler"].transform(image.reshape(-1, bands))
    return artifact["pca"].transform(scaled).reshape(h, w, artifact["n_components"]).astype(np.float32)


def extract_patches(cube: np.ndarray, coordinates: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    if cube.ndim != 3 or patch_size % 2 != 1:
        raise ValueError("Cube must be HxWxC and patch size must be odd.")
    margin = patch_size // 2
    padded = np.pad(cube, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")
    if not len(coordinates):
        raise ValueError("Cannot extract patches for an empty split.")
    return np.stack([padded[r:r + patch_size, c:c + patch_size] for r, c in coordinates]).astype(np.float32)


def save_preprocessor(artifact: dict[str, Any], path: Path = PREPROCESSOR_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(artifact, temporary)
    temporary.replace(path)


def load_preprocessor(path: Path = PREPROCESSOR_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Preprocessing artifact missing: {path}. Run `python src/dataset_split.py` first.")
    artifact = joblib.load(path)
    if artifact.get("n_components") != N_COMPONENTS or artifact.get("patch_size") != PATCH_SIZE:
        raise ValueError("Preprocessing artifact configuration is incompatible with this model.")
    return artifact
