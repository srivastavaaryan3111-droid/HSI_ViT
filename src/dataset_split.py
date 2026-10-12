"""Create stratified 70/10/20 pixel splits and fit train-only preprocessing."""
import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing import (N_COMPONENTS, PATCH_SIZE, RANDOM_SEED, ROOT,
                           extract_patches, fit_preprocessor, load_indian_pines,
                           save_preprocessor, transform_image)


def build_dataset(spatial: bool = False):
    image, gt = load_indian_pines()
    coordinates = np.argwhere(gt > 0)
    labels = gt[gt > 0] - 1
    counts = np.bincount(labels, minlength=16)
    if np.any(counts < 3):
        rare = [str(i + 1) for i, count in enumerate(counts) if count < 3]
        raise ValueError("Stratified 70/10/20 split requires at least 3 pixels per class; rare classes: " + ", ".join(rare))
    train_coords, held_coords, y_train, y_held = train_test_split(
        coordinates, labels, test_size=0.30, random_state=RANDOM_SEED, stratify=labels)
    val_coords, test_coords, y_val, y_test = train_test_split(
        held_coords, y_held, test_size=2/3, random_state=RANDOM_SEED, stratify=y_held)
    # spatial flag is explicit in the CLI/docs; the baseline remains random pixel stratification.
    if spatial:
        raise NotImplementedError("Spatially separated split is not implemented; use default stratified pixel split.")

    artifact = fit_preprocessor(image, train_coords, N_COMPONENTS)
    cube = transform_image(image, artifact)
    splits = {"train": (train_coords, y_train), "val": (val_coords, y_val), "test": (test_coords, y_test)}
    for name, (coords, split_labels) in splits.items():
        np.save(ROOT / "data" / f"X_{name}.npy", extract_patches(cube, coords, PATCH_SIZE))
        np.save(ROOT / "data" / f"y_{name}.npy", split_labels.astype(np.int64))
    save_preprocessor(artifact)
    print("Saved stratified pixel splits:", ", ".join(f"{k}={len(v[1])}" for k, v in splits.items()))
    print(f"Training-only PCA variance retained: {artifact['pca'].explained_variance_ratio_.sum():.2%}")
    print("Note: adjacent/overlapping spatial patches can occur across random pixel splits.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", action="store_true", help="Reserved; spatial holdout mode is not implemented")
    build_dataset(spatial=parser.parse_args().spatial)
