# HSI_ViT: Hyperspectral Image Classification Using Vision Transformers

## 📌 Project Overview

HSI_ViT is a hyperspectral image classification project that uses a Vision Transformer (ViT) model to classify land-cover types from hyperspectral images.

Unlike ordinary RGB images, hyperspectral images contain multiple spectral bands that provide detailed information about the materials present in each pixel.

This project uses the **Indian Pines dataset** and PyTorch to classify hyperspectral image patches into 16 land-cover classes.

The project includes data preprocessing, dimensionality reduction, patch extraction, model training, evaluation, and a Streamlit-based user interface.

## 🎯 Objectives

- Load and preprocess hyperspectral image data.
- Reduce spectral dimensionality using Principal Component Analysis (PCA).
- Extract spatial-spectral patches around labeled pixels.
- Implement a Vision Transformer model using PyTorch.
- Train and evaluate the classification model.
- Develop a Streamlit interface for hyperspectral image visualization and model integration.

## 📊 Dataset

### Indian Pines Dataset

The project uses the Indian Pines hyperspectral dataset.

| Property | Description |
|---|---|
| Dataset | Indian Pines |
| Image dimensions | 145 × 145 pixels |
| Original spectral bands | 200 |
| Number of classes | 16 |
| Background label | 0 |
| Patch size | 11 × 11 pixels |
| PCA components | 30 |
| Total labeled pixels | 10,249 |

### Required Dataset Files

Place the following MATLAB files inside the `data/` directory:

- `Indian_pines_corrected.mat` — Hyperspectral image data.
- `Indian_pines_gt.mat` — Ground-truth class labels.

Expected structure:

```text
data/
├── Indian_pines_corrected.mat
└── Indian_pines_gt.mat
```

If the dataset files are not included in the repository, download them from an authorized dataset source before running the project.

## 🏗️ Project Structure

```text
HSI_ViT/
├── app.py
├── data/
│   ├── Indian_pines_corrected.mat
│   ├── Indian_pines_gt.mat
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
├── models/
│   └── hsi_vit.pth
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── patch_extraction.py
│   ├── dataset_split.py
│   ├── pytorch_dataset.py
│   ├── vit_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── test.py
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** The `.npy` files are generated during preprocessing and do not need to be committed to GitHub. They can be regenerated from the original dataset.

## ⚙️ Methodology

The project follows these stages:

### 1. Data Loading

The hyperspectral image and ground-truth labels are loaded from MATLAB `.mat` files.

The corrected Indian Pines image has dimensions:

`145 × 145 × 200`

### 2. Data Preprocessing

StandardScaler is used for feature standardization, and PCA reduces the original 200 spectral bands to 30 components.

This reduces the input dimensionality while retaining useful spectral information.

### 3. Patch Extraction

An `11 × 11` spatial patch is extracted around each labeled pixel.

Each patch contains 30 spectral components per pixel, producing an input shape of:

`11 × 11 × 30`

Pixels labeled `0` are excluded because they represent background.

The remaining class labels are converted from `1–16` to `0–15` for compatibility with PyTorch's classification loss.

### 4. Train-Test Split

The labeled patches are divided into training and testing sets using an 80:20 split.

The split uses a fixed random state and stratification.

| Dataset | Samples |
|---|---:|
| Total labeled patches | 10,249 |
| Training samples | 8,199 |
| Testing samples | 2,050 |

### 5. PyTorch Data Loading

The generated NumPy arrays are converted into PyTorch tensors.

- Input patches use `torch.float32`.
- Class labels use `torch.long`.
- Batch size: 64.

PyTorch DataLoaders provide batches of data for training and evaluation.

### 6. Vision Transformer Architecture

The HSI_ViT model uses a lightweight Transformer architecture to learn spatial-spectral representations.

| Component | Configuration |
|---|---|
| Input patch | 11 × 11 × 30 |
| Patch tokens | 121 |
| Embedding dimension | 64 |
| Attention heads | 4 |
| Transformer encoder layers | 2 |
| Feed-forward dimension | 128 |
| Dropout | 0.1 |
| Output classes | 16 |

Each pixel in a patch is represented by its 30 spectral features.

A linear embedding layer converts these features into 64-dimensional vectors. A learnable classification token and positional embeddings are added before the sequence passes through the Transformer encoder.

The final classification-token representation is passed to a linear layer that produces scores for the 16 classes.

### 7. Model Training

The model is trained using PyTorch.

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Batch size | 64 |
| Current training duration | 1 epoch |
| Device | CPU |

The trained model is saved as:

`models/hsi_vit.pth`

### 8. Model Evaluation

The evaluation script loads the saved model and measures its classification accuracy on the test set.

Additional metrics, including precision, recall, F1-score, and a confusion matrix, are planned improvements.

## 🚀 Installation and Setup

### Prerequisites

- Python
- Conda or Miniconda
- Git
- Visual Studio Code (recommended)

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd HSI_ViT
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your repository's actual URL.

### Step 2: Create a Conda Environment

```bash
conda create -n hsi-vit python=3.11
conda activate hsi-vit
```

Use a Python version compatible with your installed dependencies.

### Step 3: Install Dependencies

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

If `requirements.txt` has not yet been created, the project currently requires packages such as:

```text
numpy
scipy
scikit-learn
matplotlib
torch
streamlit
```

Ensure that the requirements file matches the packages imported by the actual project code.

### Step 4: Add the Dataset

Place the two required `.mat` files inside the `data/` directory.

```text
data/Indian_pines_corrected.mat
data/Indian_pines_gt.mat
```

## ▶️ How to Run the Project

Run these commands from the project's root directory.

### Step 1: Preprocess the Dataset

```bash
python src/dataset_split.py
```

This script performs preprocessing, dimensionality reduction, patch extraction, and train-test splitting.

It generates:

- `data/X_train.npy`
- `data/X_test.npy`
- `data/y_train.npy`
- `data/y_test.npy`

### Step 2: Prepare the PyTorch DataLoaders

```bash
python src/pytorch_dataset.py
```

This script loads the generated arrays and prepares the training and testing DataLoaders.

### Step 3: Train the Model

```bash
python src/train.py
```

The model is trained and the weights are saved to:

`models/hsi_vit.pth`

### Step 4: Evaluate the Model

```bash
python src/evaluate.py
```

This script loads the trained model and calculates test-set classification accuracy.

### Step 5: Launch the Streamlit Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

The current interface supports hyperspectral image upload and spectral-band visualization. Verify model prediction functionality against the current implementation of `app.py`.

## 📈 Preliminary Results

An initial run produced the following results:

| Metric | Result |
|---|---:|
| Training epochs | 1 |
| Training accuracy | 62.63% |
| Correct test predictions | 1,753 out of 2,050 |
| Test accuracy | 85.51% |

These results are preliminary and should not be considered final validated performance.

**Important:** The current preprocessing pipeline fits StandardScaler and PCA before splitting the data into training and testing sets. This introduces data leakage because information from the test set can influence preprocessing.

The pipeline must be corrected and the model re-evaluated before reporting a final accuracy.

## ⚠️ Current Limitations

The following improvements are needed:

1. **Prevent data leakage:** Split the labeled samples before fitting StandardScaler and PCA.
2. **Consistent preprocessing:** Save the fitted preprocessing objects and reuse them during inference.
3. **Improve training:** Train for more epochs and use a validation set to select the best model.
4. **Improve evaluation:** Report precision, recall, macro F1-score, weighted F1-score, and a confusion matrix.
5. **Handle class imbalance:** Examine per-class performance, especially for classes with fewer samples.
6. **Complete the user interface:** Verify that the Streamlit application loads the trained model and produces valid predictions.
7. **Generate a classification map:** Extend inference to classify pixels across the image rather than only a single pixel or patch.
8. **Improve reproducibility:** Document dependency versions, random seeds, and the evaluation procedure.

Random pixel-level splitting can also cause spatial overlap between training and testing patches. Spatially separated evaluation regions should be considered for a more rigorous performance estimate.

## 🖥️ User Interface

The project includes a Streamlit application for interacting with hyperspectral image data.

Planned and ongoing interface features include:

- Uploading a hyperspectral image.
- Displaying image dimensions and spectral bands.
- Integrating the trained Vision Transformer.
- Displaying predicted classes and confidence scores.
- Visualizing a full-image classification map.

Features should be considered complete only after they have been implemented and tested in the current application.

## 📁 GitHub and Generated Files

Generated NumPy arrays can be excluded from version control because they can be recreated from the original dataset.

Recommended `.gitignore` entries:

```gitignore
__pycache__/
*.pyc
.venv/
.env
.vscode/

# Generated dataset arrays
data/*.npy
```

Check the size of the original `.mat` files and trained model before committing them. GitHub rejects individual files larger than 100 MB in ordinary Git repositories.

If the original dataset or model checkpoint is not included in the repository, document how to obtain it or regenerate it.

## 🔬 Future Improvements

- Correct the preprocessing and evaluation pipeline.
- Train and tune the Vision Transformer using a validation set.
- Compare the model against traditional machine-learning baselines.
- Generate full-image classification maps.
- Add class-name labels and a color legend.
- Add detailed classification metrics.
- Improve the Streamlit interface.
- Document reproducible experiments and final results.

## 📝 Conclusion

HSI_ViT explores the use of Vision Transformers for hyperspectral image classification using the Indian Pines dataset.

The current implementation includes preprocessing, dimensionality reduction, patch extraction, PyTorch data loading, model training, evaluation, and an initial Streamlit interface.

The next development priorities are to correct preprocessing leakage, ensure consistent inference preprocessing, improve evaluation, and complete full-image classification-map visualization.

## 🙏 Acknowledgements

This project uses the Indian Pines hyperspectral dataset and the PyTorch deep-learning framework.

Add the appropriate dataset citation and source attribution based on the version of the dataset used.