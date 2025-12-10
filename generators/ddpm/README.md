git # DDPM - Denoising Diffusion Probabilistic Model

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Generate high-quality synthetic skin lesion images using pretrained diffusion models.

## Table of Contents

- [🎯 Overview](#-overview)
- [⚠️ Disclaimer](#️-disclaimer)
- [🏗️ Architecture](#️-architecture)
- [🔧 Prerequisites](#-prerequisites)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [🎓 Training](#-training)
- [🎨 Image Generation](#-image-generation)
- [📊 Training Metrics](#-training-metrics)
- [💡 Best Practices](#-best-practices)
- [📈 Expected Timeline](#-expected-timeline)
- [🐛 Troubleshooting](#-troubleshooting)
- [🆚 DDPM vs DCGAN](#-ddpm-vs-dcgan)
- [📚 Reference](#-reference)
- [📄 License](#-license)

## 🎯 Overview

DDPM produces superior quality images through iterative denoising, leveraging pretrained weights for medical imaging.

**Key Advantages over DCGAN:**
- Better image quality and diversity
- More stable training (no mode collapse)
- Pretrained weights (`google/ddpm-ema-imagenet-64`)
- Output: 128×128 high-resolution images

**Use Case:** Generate 5,900+ synthetic malignant lesions to balance dataset (374 → 6,274 samples)

## ⚠️ Disclaimer

This tool generates synthetic medical images for research purposes only. Generated images should not be used for clinical diagnosis or treatment decisions. Always consult with qualified medical professionals.

## 🏗️ Architecture

**Denoising Process:**
```
Pure Noise (T=1000) → Iterative Denoising (50-1000 steps) → Clean Image (T=0)
```

**UNet** (100M params): Encoder (downsample) → Bottleneck with attention → Decoder (upsample)

**Upsampling:** 64×64 (training) → 128×128 (output)

## 🔧 Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU with at least 8GB VRAM (16GB recommended)
- Hugging Face account (for accessing pretrained models)
- Access to skin lesion image dataset (e.g., ISIC dataset)

## 📦 Installation

Ensure prerequisites are met, then install dependencies:

```bash
pip install -r ../../requirements.txt
```

Optional for faster attention:
```bash
pip install xformers
```

## 🚀 Quick Start

### 1. Prepare Data

Create CSV with target class images:
```csv
image_name
ISIC_0000001
```

Update paths in `DDPM_model.py`:
```python
self.data_dir = Path("path/to/images")
self.malignant_csv = Path("path/to/labels.csv")
```

### 2. Train (Fine-tune)

```bash
python DDPM_model.py
```

Configuration in `DDPMPretrainedConfig`:
```python
self.image_size = 64
self.batch_size = 16
self.learning_rate = 1e-5      # Low LR for fine-tuning
self.num_epochs = 3000
self.pretrained_model = "google/ddpm-ema-imagenet-64"
```

### 3. Generate Images

```bash
python DDPM_sampling.py
```

Configuration in `main()`:
```python
TOTAL_IMAGES = 5900        # Number to generate
BATCH_SIZE = 4             # Generation batch
INFERENCE_STEPS = 50       # Quality (50-1000)
```

## 🎓 Training

### Training Process

1. **Initialize:** Load pretrained UNet + DDPMScheduler (1000 timesteps)
2. **Training Step:** Sample timesteps → Add noise → Predict noise → MSE loss
3. **Monitoring:** Save checkpoints & samples every 25 epochs

### Key Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `learning_rate` | 1e-5 | 1e-6 to 1e-4 | Low for fine-tuning |
| `batch_size` | 16 | 4-32 | 8GB VRAM: 8-16 |
| `num_epochs` | 3000 | 1000-5000 | More = better quality |

### Output Files

```
DDPM_pretrained/
├── checkpoints/
│   ├── ddpm_pretrained_best.pt
│   ├── ddpm_pretrained_latest.pt
│   └── ddpm_pretrained_epoch_*.pt
└── samples/
    ├── samples_epoch_*.png
    └── individual_epoch_*/
```

### Resume Training

Automatically resumes from latest checkpoint.

## 🎨 Image Generation

### Generation Process

1. **Load checkpoint:** Auto-finds best → latest → highest epoch
2. **Iterative denoising:** Start with noise, denoise for N steps
3. **Upsample:** 64×64 → 128×128 (bilinear)
4. **Save:** Individual images + preview grids

### Inference Steps vs Quality

| Steps | Time/Image | Quality | Recommended |
|-------|-----------|---------|-------------|
| 25 | ~1s | ⭐⭐⭐ Moderate | Fast |
| 50 | ~2s | ⭐⭐⭐⭐ Good | **Default** |
| 100 | ~4s | ⭐⭐⭐⭐⭐ Excellent | High quality |
| 1000 | ~40s | ⭐⭐⭐⭐⭐ Maximum | Research |

### Output Structure

```
generated_malignant_ddpm_128/
├── individual/
│   └── malignant_ddpm_*.png (128×128)
└── grids/
    ├── preview_grid_16.png (4×4)
    ├── preview_grid_36.png (6×6)
    └── preview_grid_64.png (8×8)
```

## 📊 Training Metrics

**Loss:** MSE between predicted/actual noise
- Good: ~0.01-0.03
- Plateau at ~0.02 is normal

**Quality Indicators:**
- ✅ Loss steadily decreases
- ✅ Samples improve visually
- ✅ Diverse generation (no mode collapse)
- ❌ Loss plateaus high (>0.05)
- ❌ Blurry/uniform samples

## 💡 Best Practices

### Training

1. **Use pretrained model:** Much faster convergence
2. **Low learning rate:** 1e-5 for fine-tuning (high LR destroys pretrained features)
3. **Monitor samples:** Check every 25-50 epochs
4. **Batch size:** 8-16 for 8GB VRAM, 16-32 for 16GB
5. **Early stop:** No improvement after 500-1000 epochs

### Generation

1. **Inference steps:** 50 for balance, 100 for quality
2. **Batch size:** 4 for 8GB VRAM, 8 for 16GB
3. **Quality check:** Inspect preview grids

### GPU Optimization

```python
# Already implemented:
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
scaler = GradScaler('cuda')  # Mixed precision
```

## 📈 Expected Timeline (RTX 3070, 8GB)

**Training:**
- 100 epochs: ~2 hrs → Basic structure
- 500 epochs: ~10 hrs → Good quality
- 1000 epochs: ~20 hrs → High quality
- 3000 epochs: ~60 hrs → Maximum quality

**Generation (5,900 images @ 50 steps):**
- Batch 4: ~90 min
- Batch 8: ~50 min (if VRAM allows)

## 🐛 Troubleshooting

**Out of Memory:**
- Training: Reduce `batch_size = 8` or `4`
- Generation: Reduce `BATCH_SIZE = 2`, `INFERENCE_STEPS = 25`

**Poor Quality:**
- Train longer (2000-3000 epochs)
- Increase inference steps (100-200)
- Lower LR (5e-6 or 1e-6)

**NaN Losses:**
- Reduce learning rate
- Check data normalization
- Remove corrupted images

**Blurry Images:**
- Train longer
- Increase inference steps to 100-200
- Check model loading

## 🆚 DDPM vs DCGAN

**Choose DDPM:**
- ⭐ Image quality critical
- ⭐ Need diversity
- ⭐ Have 16GB+ VRAM
- ⭐ Can wait for longer training

**Choose DCGAN:**
- ⚡ Fast training needed
- ⚡ Limited compute (8GB)
- ⚡ Quick prototyping

**Medical imaging:** DDPM preferred (superior quality/diversity)

## 📚 Reference

[Ho et al., 2020 - Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

Pretrained model: [`google/ddpm-ema-imagenet-64`](https://huggingface.co/google/ddpm-ema-imagenet-64)

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License - see the [LICENSE](../../../LICENSE) file for details.

---

**Last Updated**: December 10, 2025
