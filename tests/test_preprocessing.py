import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing import extract_patches, fit_preprocessor, transform_image


def test_fits_only_given_training_spectra_and_transforms_shape():
    rng = np.random.default_rng(7)
    image = rng.normal(size=(9, 10, 6)).astype(np.float32)
    coords = np.array([[r, c] for r in range(3) for c in range(4)])
    artifact = fit_preprocessor(image, coords, n_components=3)
    transformed = transform_image(image, artifact)
    patches = extract_patches(transformed, np.array([[0, 0], [8, 9]]), 3)
    assert transformed.shape == (9, 10, 3)
    assert patches.shape == (2, 3, 3, 3)
    np.testing.assert_allclose(artifact["scaler"].mean_, image[coords[:, 0], coords[:, 1]].mean(axis=0))


def test_rejects_nonfinite_image():
    image = np.ones((3, 3, 4), dtype=np.float32)
    artifact = fit_preprocessor(image, np.array([[0, 0], [0, 1], [1, 0]]), n_components=2)
    image[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        transform_image(image, artifact)
