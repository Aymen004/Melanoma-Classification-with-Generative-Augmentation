# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-12-17

### 🚀 Major Release - Hybrid Architecture

This release introduces the **Hybrid Melanoma Detection System**, a significant evolution from the previous supervised-only approach.

### Added

- **Hybrid Detection Pipeline**: Combined DenseNet + VAE architecture
  - `src/inference/hybrid_system.py`: Main hybrid inference module
  - Rescue Mode logic for zero-miss detection
  
- **ConvVAE Anomaly Detector**: 
  - `src/models/vae.py`: Full ConvVAE implementation
  - L1 reconstruction loss for texture preservation
  - Trained exclusively on benign images
  
- **Improved Code Structure**:
  - Modular `src/` directory with models, inference, utils
  - Professional docstrings and type hints
  - Comprehensive error handling and logging
  
- **Documentation**:
  - Detailed README with architecture diagrams
  - Project structure guide (PROJECT_STRUCTURE.md)
  - This CHANGELOG

- **Inference Tools**:
  - `inference.py`: Simple CLI for single-image prediction
  - JSON output support
  - Batch processing capabilities

### Changed

- **DenseNet Threshold**: Optimized from 0.5 to 0.3 for better recall
- **Preprocessing**: Explicit separation of DenseNet (224×224) vs VAE (128×128)
- **Results Reporting**: Comprehensive metrics with clinical interpretation

### Performance Improvements

| Metric | v1.0 (DenseNet Only) | v2.0 (Hybrid) | Improvement |
|--------|----------------------|---------------|-------------|
| Recall | 79.32% | **99.75%** | **+20.43%** |
| F1-Score | 76.33% | **83.94%** | **+7.62%** |
| Missed Cancers | 1,554 | **19** | **-98.8%** |

### Fixed

- Preprocessing mismatch between DenseNet and VAE
- Checkpoint loading key remapping issues
- Memory leaks in batch inference

---

## [1.0.0] - 2024-XX-XX

### Initial Release

- DenseNet-121 classifier with DDPM augmentation
- Basic inference pipeline
- Training scripts
- Evaluation metrics

### Features

- Supervised classification on melanoma images
- DDPM synthetic data generation
- F1-Score: ~76-83%
- False Negative Rate: ~20% (⚠️ Medical concern)

### Known Issues

- High false negative rate unacceptable for clinical use
- No safety net for classifier mistakes
- Single-model reliance

---

## Future Roadmap

### [2.1.0] - Planned

- [ ] Web interface for easy deployment
- [ ] ONNX export for cross-platform inference
- [ ] Grad-CAM visualization for interpretability
- [ ] Integration with medical imaging systems (DICOM)

### [2.2.0] - Planned

- [ ] Multi-model ensemble beyond DenseNet
- [ ] Attention mechanisms for lesion localization
- [ ] Uncertainty quantification
- [ ] Active learning for continuous improvement

---

## Contributors

- Lead Developer: [Your Name]
- VAE Architecture: [Contributor]
- Evaluation Framework: [Contributor]

## Acknowledgments

Based on research from "Leveraging Deep Generative Models to Boost Tumor Classification Accuracy"
