import streamlit as st
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="HSI Classification",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Hyperspectral Image Classification")
st.subheader("Vision Transformer Based Classification")

st.write(
    "Upload a hyperspectral image and explore its spectral information."
)

st.divider()

# Upload HSI
st.header("Upload Hyperspectral Image")

uploaded_file = st.file_uploader(
    "Choose an Indian Pines .mat file",
    type=["mat"]
)

if uploaded_file is not None:

    # Read uploaded .mat file
    data = sio.loadmat(uploaded_file)

    # Extract HSI
    if "indian_pines_corrected" in data:
        hsi = data["indian_pines_corrected"]

        st.success("Hyperspectral image loaded successfully!")

        # Display information
        st.write("### Dataset Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Height", hsi.shape[0])

        with col2:
            st.metric("Width", hsi.shape[1])

        with col3:
            st.metric("Spectral Bands", hsi.shape[2])

        st.write("HSI Shape:", hsi.shape)

        # Select spectral band
        band = st.slider(
            "Select Spectral Band",
            min_value=0,
            max_value=hsi.shape[2] - 1,
            value=50
        )

        # Display selected band
        st.write(f"### Spectral Band {band}")

        fig, ax = plt.subplots()
        ax.imshow(hsi[:, :, band], cmap="gray")
        ax.axis("off")

        st.pyplot(fig)

    else:
        st.error(
            "This file does not contain 'indian_pines_corrected'."
        )