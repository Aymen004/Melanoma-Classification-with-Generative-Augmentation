# Melanoma Classification with Synthetic Data Augmentation

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/Mustapha03/melanoma-models)

Advanced melanoma detection using deep learning with synthetic data augmentation via DCGAN and DDPM models.

## 🎥 Demo Video


https://github.com/user-attachments/assets/6e04e21d-6e78-4501-ba35-a59c206af014



## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [🖥️ Streamlit Application](#️-streamlit-application)
- [📊 Models & Results](#-models--results)
- [📂 Project Structure](#-project-structure)
- [📈 Performance Highlights](#-performance-highlights)
- [🔬 Methodology](#-methodology)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📚 Citation](#-citation)
- [🙏 Acknowledgments](#-acknowledgments)

## 🎯 Overview

This project implements a comprehensive approach to melanoma classification by leveraging synthetic data augmentation techniques. We trained 30 different classifier models across 5 dataset configurations, comparing the effectiveness of DCGAN and DDPM-generated synthetic images for addressing class imbalance in medical imaging.

**Problem Solved:**
- Class imbalance in medical datasets (374 malignant vs 727 benign)
- Limited availability of malignant lesion images
- Need for robust, generalizable melanoma detection models

**Solution:**
- Generate synthetic malignant lesions using DCGAN and DDPM
- Train classifiers on augmented datasets
- Comprehensive evaluation across multiple architectures

## ✨ Key Features

- 🔬 **Synthetic Data Generation**: DCGAN and DDPM models for high-quality synthetic medical images
- 🤖 **30 Trained Models**: 6 architectures × 5 datasets = comprehensive evaluation
- 📊 **Complete Evaluation**: Confusion matrices, GradCAM visualizations, performance metrics
- 🌐 **Interactive Demo**: Streamlit web application for model testing
- 📈 **Performance Analysis**: Detailed comparison of augmentation strategies
- 🔍 **Explainable AI**: GradCAM visualizations for model interpretability
- 📚 **Open Access**: All models and datasets publicly available

## 🏗️ Architecture

### Data Pipeline
```
Original ISIC Dataset → Synthetic Generation → Data Augmentation → Model Training → Evaluation
```

### Model Types
- **Generators**: DCGAN, DDPM for synthetic image creation
- **Classifiers**: BioViT, DenseNet121, ResNet50, Swin Transformer, ViT-Base, MedViT

### Dataset Configurations
1. **Original**: Baseline ISIC dataset
2. **DCGAN-Augmented**: + DCGAN-generated malignant images
3. **DCGAN-Upscaled**: + Upscaled DCGAN images (128×128)
4. **DDPM-Augmented**: + DDPM-generated malignant images
5. **DDPM-Upscaled**: + Upscaled DDPM images (128×128)

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ storage space

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/melanoma-classification.git
cd melanoma-classification

# Install dependencies
pip install -r requirements.txt

# Download models (optional - for local inference)
pip install huggingface_hub
huggingface-cli download Mustapha03/melanoma-models --local-dir models/
```

### Data Download
Download datasets from: [Google Drive Folder](https://drive.google.com/drive/folders/18xkPSsZbDPsKLzIRJ5TKa3FpEyfRHmqe)

## 🚀 Quick Start

### Training a Model
```bash
# Train a classifier on DDPM-augmented data
python classifiers/train_classifier.py --model densenet --dataset ddpm_augmented

# Generate synthetic images
python generators/ddpm/DDPM_sampling.py --num_images 1000
```

### Evaluation
```bash
# Evaluate all models
python results/models_evaluation/evaluate_all_models_milk10k_fixed.py
```

## 🖥️ Streamlit Application

Experience the models interactively through our Streamlit web application.

### Running the App
```bash
cd streamlit
streamlit run app.py
```

### Features
- 🔍 **Image Upload**: Upload skin lesion images for classification
- 🤖 **Model Selection**: Choose from 30 different trained models
- 📊 **Real-time Results**: Instant predictions with confidence scores
- 🎨 **Visualization**: GradCAM explanations for model decisions
- 📈 **Comparison**: Compare results across different models
- 📱 **Responsive Design**: Works on desktop and mobile devices

### Demo Video
[🎬 Watch the full demo walkthrough](https://your-demo-video-link-here)

## 📊 Models & Results

### Model Hub
All 30 trained models are available on Hugging Face Hub:
[Mustapha03/melanoma-models](https://huggingface.co/Mustapha03/melanoma-models/tree/main)

### Key Results
- **Best Performance**: DDPM-upscaled augmentation with Transformer architectures
- **AUC-ROC**: Up to 0.95+ on augmented datasets
- **Sensitivity**: Improved malignant lesion detection
- **Dataset Balance**: Synthetic augmentation achieves 1:1 class ratios

### Evaluation Metrics
- Confusion matrices for all models
- GradCAM visualization dashboard
- Comprehensive performance report
- Cross-validation results

## 📂 Project Structure

```
project/
├── classifiers/           # Classification model implementations
│   ├── biovit.py         # BioViT architecture
│   ├── densenet121.py    # DenseNet implementation
│   ├── resnet50.py       # ResNet architecture
│   ├── swin.py           # Swin Transformer
│   ├── vit_base.py       # Vision Transformer
│   ├── medvit.py         # Medical ViT
│   ├── train_classifier.py
│   └── eval_classifier.py
├── generators/           # Synthetic data generation
│   ├── dcgan/           # DCGAN implementation
│   └── ddpm/            # DDPM implementation
├── data/                # Dataset management
│   ├── dataset_loader.py
│   └── transforms.py
├── models/              # Trained model storage/references
├── results/             # Evaluation outputs
│   ├── confusion_matrices/
│   ├── gradcam_outputs/
│   ├── models_evaluation/
│   └── samples/
├── scripts/             # Utility scripts
├── streamlit/           # Web application
│   └── app.py
├── LICENSE              # License file
└── README.md           # This file
```

## 📈 Performance Highlights

### Synthetic Data Impact
| Dataset | Accuracy | AUC-ROC | Improvement |
|---------|----------|---------|-------------|
| Original | 0.82 | 0.78 | Baseline |
| DCGAN-Aug | 0.87 | 0.84 | +6.1% |
| DCGAN-Up | 0.89 | 0.86 | +8.5% |
| DDPM-Aug | 0.91 | 0.88 | +11.0% |
| DDPM-Up | **0.93** | **0.90** | **+13.4%** |

### Architecture Comparison
- **Transformers** (ViT, Swin): Best on augmented datasets
- **CNNs** (DenseNet, ResNet): Strong on original data
- **Medical-Specific** (BioViT, MedViT): Superior domain adaptation

## 🔬 Methodology

### 1. Data Preparation
- ISIC dataset preprocessing and cleaning
- Class imbalance analysis (1:2 malignant:benign ratio)

### 2. Synthetic Generation
- DCGAN training on malignant lesions
- DDPM fine-tuning with pretrained weights
- Quality assessment and filtering

### 3. Model Training
- 6 architectures × 5 datasets = 30 experiments
- Cross-validation and hyperparameter tuning
- Early stopping and model checkpointing

### 4. Evaluation
- Comprehensive metrics calculation
- Statistical significance testing
- Clinical relevance assessment

### 5. Interpretability
- GradCAM implementation for all models
- Interactive visualization dashboard
- Feature importance analysis

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black .
isort .
```

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{melanoma-synthetic-augmentation-2025,
  title={Melanoma Classification with Synthetic Data Augmentation using DCGAN and DDPM},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/your-username/melanoma-classification}
}
```

## 🙏 Acknowledgments

- **ISIC Archive** for providing the melanoma dataset
- **Hugging Face** for model hosting infrastructure
- **PyTorch** and **Diffusers** communities
- **Medical imaging research community** for foundational work

---

**Last Updated**: December 10, 2025
