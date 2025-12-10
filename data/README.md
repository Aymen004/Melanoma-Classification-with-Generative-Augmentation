# Datasets

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This folder contains the datasets used for training and evaluating melanoma classification models.

## Table of Contents

- [🎯 Overview](#-overview)
- [📥 Download](#-download)
- [📂 Dataset Descriptions](#-dataset-descriptions)
- [🏗️ Organization](#️-organization)
- [🔧 Usage](#-usage)
- [📊 Statistics](#-statistics)
- [📄 License](#-license)

## 🎯 Overview

The project uses 5 different dataset configurations for comprehensive evaluation of synthetic data augmentation techniques in medical image classification:

- **Original ISIC Dataset**: Baseline dataset
- **DCGAN-Augmented**: Original + DCGAN-generated malignant images
- **DCGAN-Upscaled**: Original + upscaled DCGAN-generated images
- **DDPM-Augmented**: Original + DDPM-generated malignant images
- **DDPM-Upscaled**: Original + upscaled DDPM-generated images

## 📥 Download

All datasets are available in a shared Google Drive folder:

**Dataset Access**: [Google Drive Folder](https://drive.google.com/drive/folders/18xkPSsZbDPsKLzIRJ5TKa3FpEyfRHmqe)

### Download Instructions

1. Access the Google Drive link above
2. Download the entire folder or individual dataset folders as needed
3. Extract and place in the `data/` directory following the organization below

## 📂 Dataset Descriptions

### 1. Original ISIC Dataset
- **Source**: ISIC Archive (International Skin Imaging Collaboration)
- **Classes**: Benign, Malignant
- **Images**: 1,101 total (727 benign, 374 malignant)
- **Resolution**: 64×64 pixels
- **Format**: JPEG/PNG

### 2. DCGAN-Augmented Dataset
- **Base**: Original ISIC + DCGAN-generated malignant images
- **Generation**: DCGAN trained on malignant lesions
- **Total Images**: ~6,000-7,000
- **Balance**: Approximately 1:1 benign:malignant ratio

### 3. DCGAN-Upscaled Dataset
- **Base**: Original ISIC + upscaled DCGAN-generated images
- **Upscaling**: Bilinear interpolation to 128×128
- **Quality**: Enhanced resolution for better detail preservation

### 4. DDPM-Augmented Dataset
- **Base**: Original ISIC + DDPM-generated malignant images
- **Generation**: Denoising Diffusion Probabilistic Model
- **Quality**: Higher quality synthetic images compared to DCGAN

### 5. DDPM-Upscaled Dataset
- **Base**: Original ISIC + upscaled DDPM-generated images
- **Resolution**: 128×128 pixels
- **Advantage**: Best quality synthetic data augmentation

## 🏗️ Organization

```
data/
├── original/              # Original ISIC dataset
│   ├── benign/           # Benign lesion images
│   └── malignant/        # Malignant lesion images
├── dcgan_augmented/      # DCGAN-augmented dataset
│   ├── benign/           # Original benign + generated
│   └── malignant/        # Original malignant + generated
├── dcgan_upscaled/       # DCGAN-upscaled dataset
├── ddpm_augmented/       # DDPM-augmented dataset
├── ddpm_upscaled/        # DDPM-upscaled dataset
├── processed/            # Preprocessed data (if any)
└── metadata/             # CSV files, labels, annotations
```

## 🔧 Usage

### Loading Datasets

```python
from data.dataset_loader import MelanomaDataset

# Load original dataset
dataset = MelanomaDataset(
    data_dir='data/original',
    transform=get_transforms()
)

# Load augmented dataset
augmented_dataset = MelanomaDataset(
    data_dir='data/ddpm_augmented',
    transform=get_transforms()
)
```

### DataLoader Example

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

## 📊 Statistics

| Dataset | Total Images | Benign | Malignant |
|---------|-------------|--------|------------|
| Original | 1,279 | 727 | 1031 |248          |
| DCGAN-Augmented | ~12,000 | ~6,050 | ~6,000 |
| DCGAN-Upscaled | ~12,000 | ~6,050 | ~6,000  |
| DDPM-Augmented | ~12,000 | ~6,050 | ~6,000  |
| DDPM-Upscaled |~12,000 | ~6,000 | ~6,000    |

### Image Specifications
- **Format**: RGB images
- **Resolutions**: 64×64 (original/augmented), 128×128 (upscaled)
- **Preprocessing**: Normalization, augmentation (rotation, flip, etc.)

## ⚠️ Important Notes

- **Medical Data**: These datasets contain sensitive medical images
- **Research Use**: Intended for research purposes only
- **Privacy**: Ensure compliance with data privacy regulations
- **Citation**: Cite ISIC dataset and this work when using

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License - see the [LICENSE](../LICENSE) file for details.

The original ISIC dataset has its own licensing terms - please refer to the ISIC Archive for details.

---

**Last Updated**: December 10, 2025
