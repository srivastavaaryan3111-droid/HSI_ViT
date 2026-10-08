import streamlit as st
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
from pathlib import Path

# Allow importing from src
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

from vit_model import HSI_ViT


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="HSI Classification",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Hyperspectral Image Classification")
st.subheader("Vision Transformer Based Classification")

st.write(
    "Upload a hyperspectral image and explore its spectral information "
    "and land-cover predictions."
)

st.divider()


# =========================================================
# Load trained model
# =========================================================

@st.cache_resource
def load_model():

    model = HSI_ViT(
        input_channels=30,
        patch_size=11,
        num_classes=16,
        embed_dim=64,
        num_heads=4,
        num_layers=2
    )

    model_path = BASE_DIR / "models" / "hsi_vit.pth"

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu"
        )
    )

    model.eval()

    return model


model = load_model()


# =========================================================
# Upload HSI
# =========================================================

st.header("Upload Hyperspectral Image")

uploaded_file = st.file_uploader(
    "Choose an Indian Pines .mat file",
    type=["mat"]
)


if uploaded_file is not None:

    # =====================================================
    # Load HSI
    # =====================================================

    data = sio.loadmat(uploaded_file)

    if "indian_pines_corrected" not in data:

        st.error(
            "This file does not contain "
            "'indian_pines_corrected'."
        )

        st.stop()


    hsi = data["indian_pines_corrected"].astype(
        np.float32
    )

    st.success(
        "Hyperspectral image loaded successfully!"
    )


    # =====================================================
    # Dataset information
    # =====================================================

    st.write("### Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Height",
            hsi.shape[0]
        )

    with col2:
        st.metric(
            "Width",
            hsi.shape[1]
        )

    with col3:
        st.metric(
            "Spectral Bands",
            hsi.shape[2]
        )

    st.write(
        "HSI Shape:",
        hsi.shape
    )


    # =====================================================
    # Spectral band visualization
    # =====================================================

    st.write("### Spectral Band Visualization")

    band = st.slider(
        "Select Spectral Band",
        min_value=0,
        max_value=hsi.shape[2] - 1,
        value=50
    )

    fig, ax = plt.subplots()

    ax.imshow(
        hsi[:, :, band],
        cmap="gray"
    )

    ax.axis("off")

    st.pyplot(fig)


    # =====================================================
    # Prediction
    # =====================================================

    st.divider()

    st.header("🧠 Vision Transformer Prediction")

    st.write(
        "The uploaded HSI is processed using the same "
        "PCA and patch extraction pipeline used during training."
    )


    if st.button(
        "🚀 Run HSI-ViT Classification"
    ):

        with st.spinner(
            "Processing hyperspectral image..."
        ):

            # ---------------------------------------------
            # 1. Normalize
            # ---------------------------------------------

            H, W, B = hsi.shape

            hsi_2d = hsi.reshape(
                -1,
                B
            )

            scaler = StandardScaler()

            hsi_normalized = scaler.fit_transform(
                hsi_2d
            )


            # ---------------------------------------------
            # 2. PCA
            # ---------------------------------------------

            pca = PCA(
                n_components=30
            )

            hsi_pca = pca.fit_transform(
                hsi_normalized
            )

            hsi_pca = hsi_pca.reshape(
                H,
                W,
                30
            )


            # ---------------------------------------------
            # 3. Padding
            # ---------------------------------------------

            patch_size = 11
            margin = patch_size // 2

            padded_hsi = np.pad(
                hsi_pca,
                (
                    (margin, margin),
                    (margin, margin),
                    (0, 0)
                ),
                mode="reflect"
            )


            # ---------------------------------------------
            # 4. Sample pixels for prediction
            # ---------------------------------------------

            # Predict a central region rather than all
            # 21,025 pixels at once.

            center_row = H // 2
            center_col = W // 2

            patch = padded_hsi[
                center_row:center_row + patch_size,
                center_col:center_col + patch_size,
                :
            ]


            # ---------------------------------------------
            # 5. Convert patch to tensor
            # ---------------------------------------------

            patch_tensor = torch.tensor(
                patch,
                dtype=torch.float32
            ).unsqueeze(0)


            # ---------------------------------------------
            # 6. Model prediction
            # ---------------------------------------------

            with torch.no_grad():

                output = model(
                    patch_tensor
                )

                probabilities = F.softmax(
                    output,
                    dim=1
                )

                predicted_class = torch.argmax(
                    probabilities,
                    dim=1
                ).item()

                confidence = probabilities[
                    0,
                    predicted_class
                ].item()


            # ---------------------------------------------
            # 7. Display result
            # ---------------------------------------------

            st.success(
                "Classification completed!"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Class",
                    predicted_class + 1
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )