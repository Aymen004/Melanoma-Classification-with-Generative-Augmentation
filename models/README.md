# Models

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This folder is dedicated to storing trained models.

## Table of Contents

- [🎯 Overview](#-overview)
- [🏗️ Model Types](#️-model-types)
- [📁 Organization](#-organization)
- [💾 Model Formats](#-model-formats)
- [🔄 Loading and Using Models](#-loading-and-using-models)
- [📊 Model Performance](#-model-performance)
- [💡 Best Practices](#-best-practices)
- [📄 License](#-license)

## 🎯 Overview

This directory contains references to 30 trained machine learning models (6 architectures × 5 datasets) used in the melanoma classification project. The models include generators for synthetic data creation and classifiers for medical image analysis. All trained models are hosted on Hugging Face Hub for easy access and deployment.

**Trained Models Summary:**
- **6 Model Architectures**: BioViT, DenseNet121, ResNet50, Swin Transformer, ViT-Base, MedViT
- **5 Datasets**: Original ISIC, DCGAN-augmented, DCGAN-upscaled, DDPM-augmented, DDPM-upscaled
- **Total**: 30 classifier models + generator checkpoints

**Model Hub**: [Mustapha03/melanoma-models](https://huggingface.co/Mustapha03/melanoma-models/tree/main)

**Datasets**: [Google Drive Folder](https://drive.google.com/drive/folders/18xkPSsZbDPsKLzIRJ5TKa3FpEyfRHmqe)

## 🏗️ Model Types

### Generators
- **DCGAN**: Deep Convolutional GAN for generating synthetic skin lesion images
- **DDPM**: Denoising Diffusion Probabilistic Model for high-quality image generation

### Classifiers (30 Models Total)
Trained on 5 datasets with 6 architectures each:

**Architectures:**
- **BioViT**: Vision Transformer for biomedical image classification
- **DenseNet121**: Dense Convolutional Network
- **ResNet50**: Residual Network
- **Swin Transformer**: Hierarchical Vision Transformer
- **ViT-Base**: Vision Transformer (base configuration)
- **MedViT**: Medical Vision Transformer

**Datasets:**
- **Original**: Baseline ISIC dataset
- **DCGAN**: Augmented with DCGAN-generated images
- **DCGAN_Upscaled**: Augmented with upscaled DCGAN images
- **DDPM**: Augmented with DDPM-generated images
- **DDPM_Upscaled**: Augmented with upscaled DDPM images

**Model Naming Convention**: `{Architecture}_{Dataset}` (e.g., `BioViT_DCGAN`, `ResNet50_DDPM_Upscaled`)

## 📁 Organization

Models are organized on Hugging Face Hub for easy access. Local directory structure (when downloaded):

```
models/
├── generators/          # GAN and diffusion models
│   ├── dcgan/
│   │   ├── dcgan_benign/       # DCGAN benign image generators
│   │   └── dcgan_malignant/    # DCGAN malignant image generators
│   └── ddpm/
│   │   ├── ddpm_benign/        # DDPM benign image generators
│   │   └── ddpm_malignant/     # DDPM malignant image generators
├── classifiers/         # Classification models (30 total)
│   ├── biovit/         # BioViT models (5 variants)
│   ├── densenet/       # DenseNet models (5 variants)
│   ├── resnet/         # ResNet models (5 variants)
│   ├── swin/           # Swin Transformer models (5 variants)
│   ├── vit/            # ViT models (5 variants)
│   └── medvit/         # MedViT models (5 variants)
└── checkpoints/         # Temporary checkpoints during training
```

**Download from HF**: All models are available at [Mustapha03/melanoma-models](https://huggingface.co/Mustapha03/melanoma-models/tree/main)

## 💾 Model Formats

Models are saved in various formats depending on the framework:

- **PyTorch (.pth/.pt)**: Native PyTorch state dictionaries
- **Hugging Face**: Model configurations and weights
- **Pickle (.pkl)**: Serialized Python objects
- **ONNX (.onnx)**: Framework-agnostic format for deployment

### File Naming Convention

- `model_final.pth`: Final trained model
- `model_best.pth`: Best performing model (validation)
- `checkpoint_epoch_X.pth`: Training checkpoint at epoch X
- `model_config.json`: Model configuration parameters

## 🔄 Loading and Using Models

### Download from Hugging Face

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download all models
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Mustapha03/melanoma-models", local_dir="./models")
```

### Individual Model Download

```python
from huggingface_hub import hf_hub_download

# Download specific model
model_path = hf_hub_download(
    repo_id="Mustapha03/melanoma-models",
    filename="BioViT_DCGAN/model.pth",
    local_dir="./models"
)
```

### PyTorch Models

```python
import torch
from transformers import AutoModelForImageClassification

# Load from local path
model = torch.load('models/classifiers/biovit/BioViT_DCGAN/model.pth')
model.eval()

# Or using transformers (if applicable)
model = AutoModelForImageClassification.from_pretrained(
    "Mustapha03/melanoma-models",
    subfolder="BioViT_DCGAN"
)

# Inference
with torch.no_grad():
    outputs = model(images)
    predictions = torch.argmax(outputs.logits, dim=1)
```

### Generator Models

```python
from models.generators.dcgan.dcgan_model import Generator

generator = Generator(nz=100, ngf=64, nc=3)
generator.load_state_dict(torch.load('models/generators/dcgan/generator_final.pth'))
generator.eval()

# Generate images
noise = torch.randn(batch_size, 100, 1, 1)
with torch.no_grad():
    fake_images = generator(noise)
```

## 📊 Model Performance

Model performance metrics for all 30 classifiers are tracked in the `results/` directory. The evaluation results show performance across different architectures and data augmentation strategies.

**Key Findings:**
- Synthetic data augmentation improves minority class performance
- DDPM-generated images generally outperform DCGAN
- Transformer-based models (ViT, Swin) show strong performance on augmented datasets

**Available Results:**
- Confusion matrices for all models
- Classification reports with precision/recall/F1 scores
- ROC curves and AUC scores
- GradCAM visualizations
- Complete evaluation metrics in CSV format

**Best Performing Models:** Check `results/models_evaluation/` for detailed rankings.

**Metrics Include:**
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **F1-Score**: Balanced metric for imbalanced datasets
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Sensitivity/Specificity**: Medical classification metrics

## 📂 Datasets

The models were trained on 5 different dataset configurations:

1. **Original**: Baseline ISIC melanoma dataset (374 malignant, 727 benign)
2. **DCGAN**: Original + DCGAN-generated malignant images
3. **DCGAN_Upscaled**: Original + upscaled DCGAN-generated images
4. **DDPM**: Original + DDPM-generated malignant images
5. **DDPM_Upscaled**: Original + upscaled DDPM-generated images

**Dataset Access**: [Google Drive Folder](https://drive.google.com/drive/folders/18xkPSsZbDPsKLzIRJ5TKa3FpEyfRHmqe)

**Data Statistics:**
- Total images per dataset: ~6,000-7,000
- Balanced classes after augmentation
- 64×64 to 128×128 image resolutions

## 💡 Best Practices

1. **Download Models**: Use Hugging Face Hub for easy access to all 30 trained models
2. **Version Control**: Models include training metadata and version information
3. **Evaluation**: Always evaluate models on the same dataset configuration used for training
4. **Reproducibility**: Models include saved hyperparameters and random seeds
5. **Medical Use**: These are research models - validate thoroughly before clinical use
6. **Backup**: Keep local copies of critical models for offline access

### Git Ignore Recommendations

Add large model files to `.gitignore`:
```
*.pth
*.pt
*.h5
*.pkl
checkpoints/
__pycache__/
```

### Model Validation

```python
# Always validate model loading
try:
    model = torch.load(model_path)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
```

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License - see the [LICENSE](../../LICENSE) file for details.

---

**Last Updated**: December 10, 2025
