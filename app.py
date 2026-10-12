"""Streamlit explorer and full-cube classifier for compatible v2 checkpoints."""
from pathlib import Path
import sys

import io
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import streamlit as st
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import extract_patches, load_preprocessor, transform_image  # noqa: E402
from vit_model import HSI_ViT  # noqa: E402

st.set_page_config(page_title="HSI ViT Classification", page_icon="🔬", layout="wide")
st.title("🔬 Hyperspectral Image Classification")
st.caption("Indian Pines ViT classifier with saved training-only preprocessing.")

CLASS_NAMES = [
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
    "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
    "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat",
    "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
]
CLASS_COLORS = ["#377eb8", "#ff7f00", "#4daf4a", "#e41a1c", "#984ea3", "#a65628",
                "#f781bf", "#999999", "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
                "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"]


@st.cache_resource
def load_pipeline():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed. Install project dependencies with `python -m pip install -r requirements.txt`.") from exc
    checkpoint_path = ROOT / "models" / "hsi_vit_v2.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError("Corrected checkpoint unavailable. Prepare data and run `python src/train.py` first.")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format_version") != 2:
        raise ValueError("This is an older checkpoint without compatible saved preprocessing. Train the v2 pipeline.")
    model = HSI_ViT(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, load_preprocessor()


try:
    model, preprocessing = load_pipeline()
except (FileNotFoundError, ValueError) as exc:
    st.warning(str(exc))
    model = preprocessing = None

dataset_files = [ROOT / "data" / "Indian_pines_corrected.mat", ROOT / "data" / "Indian_pines_gt.mat"]
missing_dataset_files = [path.name for path in dataset_files if not path.is_file()]
st.caption("Bundled dataset files: " + (", ".join(missing_dataset_files) + " missing" if missing_dataset_files else "available"))
if missing_dataset_files:
    st.info("For built-in dataset preparation/evaluation, add " + " and ".join(missing_dataset_files) + " to the data folder.")

st.subheader("Indian Pines class legend")
legend_cols = st.columns(4)
for class_id, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS), start=1):
    legend_cols[(class_id - 1) % len(legend_cols)].markdown(
        f'<div style="display:flex;align-items:center;margin:0.25rem 0;">'
        f'<span style="background:{color};width:16px;height:16px;display:inline-block;margin-right:8px;'
        f'border:1px solid #777;"></span><span>{class_id}. {name}</span></div>',
        unsafe_allow_html=True)

uploaded = st.file_uploader("Upload a compatible Indian Pines .mat cube", type=["mat"])
if uploaded:
    try:
        mat = sio.loadmat(uploaded)
    except (ValueError, OSError, NotImplementedError) as exc:
        st.error(f"Could not read the uploaded MAT file: {exc}")
        st.stop()
    if "indian_pines_corrected" not in mat:
        st.error("MAT file must contain 'indian_pines_corrected'.")
        st.stop()
    image = np.asarray(mat["indian_pines_corrected"], dtype=np.float32)
    if image.ndim != 3 or not np.isfinite(image).all():
        st.error("Cube must be a finite HxWxB array.")
        st.stop()
    h, w, bands = image.shape
    c1, c2, c3 = st.columns(3)
    c1.metric("Height", h); c2.metric("Width", w); c3.metric("Bands", bands)
    band = st.slider("Spectral band", 0, bands - 1, min(50, bands - 1))
    fig, ax = plt.subplots(); ax.imshow(image[:, :, band], cmap="gray"); ax.axis("off")
    st.pyplot(fig); plt.close(fig)

    if model is not None and st.button("Classify full image", type="primary"):
        try:
            cube = transform_image(image, preprocessing)
            coords = np.indices((h, w)).reshape(2, -1).T
            patches = extract_patches(cube, coords)
            loader = DataLoader(TensorDataset(torch.from_numpy(patches)), batch_size=128, shuffle=False)
            outputs = []
            with torch.inference_mode():
                for (batch,) in loader:
                    outputs.append(model(batch).argmax(dim=1).numpy())
            class_map = np.concatenate(outputs).reshape(h, w).astype(np.uint8) + 1
            class_ids, class_counts = np.unique(class_map, return_counts=True)
            st.success(f"Inference complete: classified all {h*w:,} pixels.")
            st.subheader("Predicted class distribution")
            distribution = {CLASS_NAMES[int(cls) - 1]: int(count) for cls, count in zip(class_ids, class_counts)}
            st.bar_chart(distribution)
            st.dataframe([{"Class ID": int(cls), "Class": CLASS_NAMES[int(cls) - 1],
                           "Pixels": int(count), "Percent": f"{100 * count / class_map.size:.2f}%"}
                          for cls, count in zip(class_ids, class_counts)], hide_index=True, use_container_width=True)
            from matplotlib.colors import ListedColormap, BoundaryNorm
            cmap = ListedColormap(CLASS_COLORS)
            norm = BoundaryNorm(np.arange(0.5, 17.5, 1), cmap.N)
            fig, ax = plt.subplots(); ax.imshow(class_map, cmap=cmap, norm=norm); ax.axis("off")
            handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=CLASS_COLORS[i],
                                  label=f"{i+1}. {CLASS_NAMES[i]}") for i in range(16)]
            ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
            st.pyplot(fig); plt.close(fig)
            buffer = io.BytesIO(); np.save(buffer, class_map)
            st.download_button("Download classification map (.npy)", buffer.getvalue(), "classification_map.npy")
        except Exception as exc:
            st.error(f"Could not classify this image: {exc}")
