# Model Checkpoints

This directory contains trained model weights for melanoma detection.

## 🚀 Quick Download

Download pre-trained models:

```bash
# Download hybrid system (recommended)
python ../scripts/download_weights.py --hybrid

# Download specific model
python ../scripts/download_weights.py --model densenet_ddpm

# List all available models
python ../scripts/download_weights.py --list
```

## 📦 Available Models

### Hybrid System (Recommended)

| Model | Size | Description | Use Case |
|-------|------|-------------|----------|
| **DenseNet_DDPM.pth** | ~30MB | DenseNet-121 classifier | Primary classification |
| **VAE_best.pth** | ~99MB | ConvVAE anomaly detector | Rescue false negatives |

**Hybrid Performance**:
- 🎯 Recall: 99.75% (missed only 19/7,514 cancers)
- 🛡️ Saved: 1,535 lives compared to DenseNet alone
- ⚡ F1-Score: 83.94%

### Alternative Classifiers

| Model | Size | Description |
|-------|------|-------------|
| ResNet_DDPM.pth | ~90MB | ResNet-50 baseline |
| ViT_DDPM.pth | ~330MB | Vision Transformer |
| Swin_DDPM.pth | ~330MB | Swin Transformer |

## 📁 Directory Structure

```
checkpoints/
├── README.md              # This file
├── DenseNet_DDPM.pth      # Main classifier (download)
├── VAE_best.pth           # Anomaly detector (download)
└── [other models]         # Optional alternatives
```

## 🔗 Hugging Face Hub

All models are hosted on Hugging Face:
[Mustapha03/melanoma-models](https://huggingface.co/Mustapha03/melanoma-models)

## 💾 Manual Download

If automatic download fails:

1. **Visit Hugging Face**: https://huggingface.co/Mustapha03/melanoma-models
2. **Download files**:
   - `DenseNet_DDPM.pth`
   - `VAE_best.pth`
3. **Place in this directory**: `checkpoints/`

## ⚙️ Usage

### Python API

```python
from src.inference.hybrid_system import load_hybrid_detector

detector = load_hybrid_detector(
    densenet_checkpoint='checkpoints/DenseNet_DDPM.pth',
    vae_checkpoint='checkpoints/VAE_best.pth'
)

results = detector.predict_single('lesion.jpg')
```

### CLI

```bash
python inference.py --image lesion.jpg \
    --densenet_checkpoint checkpoints/DenseNet_DDPM.pth \
    --vae_checkpoint checkpoints/VAE_best.pth
```

## 🔍 Model Details

### DenseNet_DDPM.pth

- **Architecture**: DenseNet-121 with 3-layer classifier head
- **Training Data**: ISIC dataset + DDPM-generated synthetic images
- **Input Size**: 224×224×3
- **Normalization**: ImageNet (μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225])
- **Classes**: 2 (benign, malignant)
- **Optimal Threshold**: 0.3 (for recall priority)

### VAE_best.pth

- **Architecture**: ConvVAE (4 encoder blocks, 4 decoder blocks)
- **Training Data**: Benign images only (normal skin patterns)
- **Input Size**: 128×128×3
- **Normalization**: [0, 1] range (ToTensor only)
- **Latent Dim**: 512
- **Loss**: L1 reconstruction + β-KL divergence (β=0.0001)
- **Anomaly Threshold**: 0.136 (calibrated on validation set)

## 📊 Performance Comparison

| System | Recall | F1-Score | Missed Cancers |
|--------|--------|----------|----------------|
| DenseNet Only | 79.32% | 76.33% | 1,554 |
| **Hybrid** | **99.75%** | **83.94%** | **19** |

**Improvement**: 98.8% reduction in missed cancers! 🎯

## ⚠️ Important Notes

1. **Different Preprocessings**: 
   - DenseNet expects 224×224 with ImageNet normalization
   - VAE expects 128×128 with [0,1] normalization
   - Hybrid system handles both automatically

2. **Thresholds Matter**:
   - DenseNet threshold: 0.3 (recall-optimized)
   - VAE threshold: 0.136 (calibrated for specificity)

3. **Git LFS**: 
   - Large `.pth` files are excluded from git by `.gitignore`
   - Use download script to fetch models

4. **Disk Space**:
   - Hybrid system: ~130MB
   - All models: ~900MB

## 🛠️ Troubleshooting

### "File not found" error

```bash
# Download models first
python ../scripts/download_weights.py --hybrid
```

### CUDA out of memory

```python
# Use CPU inference
detector = load_hybrid_detector(..., device='cpu')
```

### Slow downloads

```bash
# Use HF mirror (China)
export HF_ENDPOINT=https://hf-mirror.com
python ../scripts/download_weights.py --hybrid
```

## 📝 License

Models are released under Creative Commons BY-NC 4.0:
- ✅ Free for research and education
- ❌ Not for commercial use without permission

## 🙏 Acknowledgments

Models trained with:
- PyTorch 2.0+
- NVIDIA RTX 3070 Laptop GPU
- ISIC Melanoma Dataset
- DDPM synthetic augmentation
