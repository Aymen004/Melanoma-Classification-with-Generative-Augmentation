# DCGAN - Deep Convolutional GAN

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Generate synthetic skin lesion images for data augmentation and class balancing.

## Table of Contents

- [🎯 Overview](#-overview)
- [🏗️ Architecture](#️-architecture)
- [🔧 Prerequisites](#-prerequisites)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [🎓 Training](#-training)
- [📊 Training Metrics](#-training-metrics)
- [📈 Expected Timeline](#-expected-timeline)
- [🔧 Advanced Usage](#-advanced-usage)
- [🐛 Troubleshooting](#-troubleshooting)
- [💡 Best Practices](#-best-practices)
- [📚 Reference](#-reference)
- [📄 License](#-license)

## 🎯 Overview

DCGAN generates synthetic medical images to address class imbalance:
- Original: 374 malignant vs 727 benign (1:2 imbalance)
- Solution: Generate synthetic malignant samples for 1:1 balance
- Result: Improved classifier performance on minority class

## ⚠️ Disclaimer

This tool generates synthetic medical images for research purposes only. Generated images should not be used for clinical diagnosis or treatment decisions. Always consult with qualified medical professionals.

## 🏗️ Architecture

**Generator** (3.5M params): `Noise(100) → 512×4×4 → 256×8×8 → 128×16×16 → 64×32×32 → RGB(3×64×64)`

**Discriminator** (2.8M params): `RGB(3×64×64) → 64×32×32 → 128×16×16 → 256×8×8 → 512×4×4 → Real/Fake`

## 🔧 Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- At least 8GB RAM
- Access to skin lesion image dataset (e.g., ISIC dataset)

## 📦 Installation

Ensure prerequisites are met, then install dependencies:

```bash
pip install -r ../../requirements.txt
```

## 🚀 Quick Start

### 1. Prepare Data

Create CSV with target class images:
```csv
filename
ISIC_0000001
ISIC_0000002
```

### 2. Train

```bash
python train_dcgan.py
```

Edit configuration in script:
```python
lr = 0.0002              # Learning rate
num_epochs = 10000       # Training epochs
batch_size = 64          # Batch size
image_dir = "path/to/images"
target_csv = "malignant_images.csv"
```

### 3. Generate Images

```bash
python dcgan_sampling.py
```

Or in Python:
```python
from dcgan_model import Generator
import torch

device = torch.device('cuda')
generator = Generator(nz=100, ngf=64, nc=3).to(device)
generator.load_state_dict(torch.load('generator_final.pth'))
generator.eval()

noise = torch.randn(1000, 100, 1, 1, device=device)
with torch.no_grad():
    synthetic_images = generator(noise)
```

## 🎓 Training

### Key Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `lr` | 0.0002 | 0.0001-0.0005 | Learning rate |
| `batch_size` | 64 | 32-128 | RTX 3070 optimal: 64 |
| `num_epochs` | 10000 | 500-10000 | More = better quality |
| `nz` | 100 | 50-200 | Latent vector size |

### Training Process

1. **Discriminator**: Train on real (label=1) and fake (label=0) images
2. **Generator**: Generate fakes to fool discriminator (label=1)
3. **Monitoring**: Saves checkpoints & samples every 25 epochs

### Output Files

```
generated_images/fake_images_epoch_*.png
checkpoints/dcgan_checkpoint_epoch_*.pth
generator_final.pth
discriminator_final.pth
training_metrics.png
```

## 📊 Training Metrics

Monitor these values:

- **Loss_D**: ~0.5-1.0 (discriminator loss)
- **Loss_G**: ~1.0-2.0 (generator loss)
- **D(x)**: ~0.7-0.9 (accuracy on real images)
- **D(G(z))**: ~0.5 (accuracy on fake images)

### Good Training Signs

✅ Losses oscillate but stay stable
✅ D(x) stays high (~0.7-0.9)
✅ Generated images become sharper over time

### Common Issues

**Mode Collapse** (similar images): Reduce LR, add noise
**Discriminator Wins** (Loss_D→0): Train generator more, reduce D LR
**Generator Wins** (Loss_G→0): Use label smoothing, increase D capacity

## 📈 Expected Timeline (RTX 3070)

| Epochs | Time | Quality |
|--------|------|---------|
| 100 | 20 min | Basic shapes |
| 500 | 1.5 hrs | Recognizable lesions |
| 1000 | 3 hrs | Good quality |
| 5000 | 15 hrs | Publication quality |

## 🔧 Advanced Usage

### Resume Training

```python
checkpoint = torch.load('dcgan_checkpoint_epoch_100.pth')
generator.load_state_dict(checkpoint['generator_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
```

### Batch Generation

```python
# Generate 5000 images
for batch in range(50):
    noise = torch.randn(100, 100, 1, 1, device='cuda')
    with torch.no_grad():
        images = generator(noise)
    # Save images...
```

## 🐛 Troubleshooting

**Out of Memory**: Reduce `batch_size = 32` or `16`

**Poor Quality**: Check dataset diversity, increase epochs, adjust LR

**Training Divergence**: Lower LR to `0.0001`, add gradient clipping

**File Errors**: Verify paths in `image_dir` and `target_csv`

## 💡 Best Practices

1. **Dataset**: Minimum 200-300 images, uniform lighting/background
2. **Start Small**: Test with 100 epochs first
3. **Monitor**: Check samples every 25 epochs
4. **Early Stop**: Stop if no improvement after 500 epochs
5. **Validation**: Test classifier on synthetic + real data

## 📚 Reference

[Radford et al., 2015 - Unsupervised Representation Learning with DCGANs](https://arxiv.org/abs/1511.06434)

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License - see the [LICENSE](../../../LICENSE) file for details.

---

**Last Updated**: December 10, 2025
