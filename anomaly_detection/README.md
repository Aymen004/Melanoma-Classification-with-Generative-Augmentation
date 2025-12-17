# Anomaly Detection Module - VAE Rescue Mode 🛡️

This module implements the **VAE-based safety net** for melanoma detection that reduced false negatives by 98.8%!

## 🎯 Core Concept

Instead of learning "what cancer looks like" (supervised), our VAE learns **"what normal skin looks like"** (unsupervised). Any deviation from normal patterns = potential anomaly.

**Why this works:**
- Trained ONLY on benign (normal) images
- Struggles to reconstruct malignant lesions → High reconstruction error
- Detects out-of-distribution cases never seen during training

---

## 📂 Files Overview

### 🔧 Core Workflow Scripts

| File | Step | Purpose |
|------|------|---------|
| **train_vae_cuda_optimized.py** | STEP 3 | Train VAE on benign images |
| **calibrate_vae_threshold.py** | STEP 4 | Find optimal threshold (0.136) |
| **evaluate_hybrid_rescue_final.py** | STEP 5 | Evaluate hybrid system |
| **evaluate_densenet_baseline.py** | STEP 3b | Baseline DenseNet metrics |

### 🏗️ Architecture & Utils

| File | Purpose |
|------|---------|
| **VAE_model.py** | ConvVAE architecture (512-dim latent) |
| **inference_vae.py** | Standalone VAE inference utils |

---

## 🚀 Complete Workflow

### STEP 3: Train VAE (2 hours on RTX 3070)

```bash
python train_vae_cuda_optimized.py \
    --data_dir ../data/calibrage/calibrage_data/benign \
    --output_dir ../vae_fix_v2_L1 \
    --epochs 200 \
    --batch_size 32 \
    --latent_dim 512 \
    --beta 0.0001 \
    --loss_type l1

# Output: vae_fix_v2_L1/checkpoints/best_model.pth (99MB)
```

### STEP 4: Calibrate Threshold (5 minutes)

```bash
python calibrate_vae_threshold.py \
    --vae_checkpoint ../vae_fix_v2_L1/checkpoints/best_model.pth \
    --data_dir ../data/calibrage/calibrage_data \
    --labels_csv ../data/calibrage/labels.csv

# Output: Optimal threshold = 0.136, AUC-ROC = 0.762
```

### STEP 5: Evaluate Hybrid System (10 minutes)

```bash
python evaluate_hybrid_rescue_final.py \
    --densenet_checkpoint ../classifiers/DenseNet_DDPM.pth \
    --vae_checkpoint ../vae_fix_v2_L1/checkpoints/best_model.pth \
    --data_dir ../data/test_data/dataset_binary \
    --densenet_threshold 0.3 \
    --vae_threshold 0.136

# Results:
# DenseNet: 1,554 missed cancers ❌
# Hybrid: 19 missed cancers ✅ (1,535 saved!)
```

---

## 📊 Results Summary

| Metric | DenseNet Only | Hybrid System | Improvement |
|--------|---------------|---------------|-------------|
| Recall | 79.32% | **99.75%** | **+20.43%** |
| F1-Score | 76.33% | **83.94%** | **+7.62%** |
| Missed Cancers | 1,554 | **19** | **-98.8%** |

**Rescue Mode**: VAE saved 1,535 lives! 🎯

---

## 🧠 Architecture

```
ConvVAE (trained on benign images only)
Input (128×128×3) → Encoder (4 layers) → Latent (512-dim)
                        ↓
                 Reparameterize z = μ + σε
                        ↓
Latent (512-dim) → Decoder (4 layers) → Output (128×128×3)

Loss = L1(x, x̂) + β*KL(q||p)  where β=0.0001
```

**Rescue Logic:**
```python
if densenet_pred == "BENIGN" and vae_error > 0.136:
    final_pred = "MALIGNANT"  # 🛡️ RESCUE!
```

---

## 💡 Key Parameters

- **latent_dim**: 512 (bottleneck size)
- **beta**: 0.0001 (KL divergence weight)
- **loss_type**: L1 (better texture preservation)
- **threshold**: 0.136 (calibrated on validation set)
- **batch_size**: 32 (reduce if OOM)

---

## 🐛 Troubleshooting

**CUDA out of memory?**
```bash
python train_vae_cuda_optimized.py --batch_size 8
```

**VAE not converging?**
- Increase epochs (try 300)
- Reduce learning rate (0.00005)
- Check data is [0,1] normalized

**Too many false alarms?**
- Increase threshold (try 0.20)
- Or retrain with higher β (0.0005)

---

See [main README](../README.md) for full project documentation.
