# PROJECT STRUCTURE - MLOps Best Practices

## Directory Tree

```
melanoma-hybrid-detection/
│
├── src/                                  # Source code (production)
│   ├── models/                          # Model architectures
│   │   ├── __init__.py
│   │   ├── densenet.py                  # DenseNet-121 classifier
│   │   └── vae.py                       # ConvVAE anomaly detector
│   ├── inference/                       # Inference pipeline
│   │   ├── __init__.py
│   │   └── hybrid_system.py             # Hybrid detection logic
│   ├── training/                        # Training modules
│   │   ├── __init__.py
│   │   ├── train_densenet.py
│   │   └── train_vae.py
│   └── utils/                           # Utilities
│       ├── __init__.py
│       ├── data_loader.py
│       ├── metrics.py
│       └── visualization.py
│
├── scripts/                             # Executable scripts
│   ├── download_weights.py              # Download pre-trained models
│   ├── evaluate_system.py               # Full system evaluation
│   └── calibrate_thresholds.py          # Threshold optimization
│
├── notebooks/                           # Jupyter notebooks (exploration)
│   ├── 01_data_exploration.ipynb
│   ├── 02_vae_training.ipynb
│   ├── 03_vae_calibration.ipynb
│   └── 04_hybrid_evaluation.ipynb
│
├── checkpoints/                         # Model weights (gitignored)
│   ├── DenseNet_DDPM.pth                # Trained classifier
│   ├── VAE_best.pth                     # Trained VAE
│   └── README.md                        # Checkpoint info
│
├── data/                                # Datasets (gitignored)
│   ├── train/
│   ├── val/
│   ├── test/
│   └── README.md                        # Dataset instructions
│
├── results/                             # Evaluation results
│   ├── confusion_matrices/
│   ├── roc_curves/
│   ├── hybrid_evaluation.csv
│   └── README.md
│
├── tests/                               # Unit tests
│   ├── test_models.py
│   ├── test_inference.py
│   └── test_utils.py
│
├── configs/                             # Configuration files
│   ├── model_config.yaml
│   └── training_config.yaml
│
├── .github/                             # GitHub workflows
│   └── workflows/
│       └── tests.yml
│
├── inference.py                         # Main CLI inference script
├── requirements.txt                     # Python dependencies
├── environment.yml                      # Conda environment (optional)
├── setup.py                             # Package installation
├── .gitignore                           # Git ignore rules
├── .gitattributes                       # Git LFS for large files
├── LICENSE                              # MIT License
├── README.md                            # Main documentation
├── CONTRIBUTING.md                      # Contribution guidelines
└── CHANGELOG.md                         # Version history
```

## File Descriptions

### Core Modules

- `src/models/densenet.py`: DenseNet-121 with 3-layer classifier head
- `src/models/vae.py`: ConvVAE with L1 loss for anomaly detection
- `src/inference/hybrid_system.py`: Main hybrid prediction pipeline

### Scripts

- `inference.py`: Single-image CLI tool
- `scripts/evaluate_system.py`: Batch evaluation on test set
- `scripts/download_weights.py`: Download pre-trained checkpoints

### Notebooks

- Exploratory analysis and visualization
- Step-by-step training tutorials
- Threshold calibration examples

## Key Principles

1. **Separation of Concerns**: Models, training, inference are separate
2. **Modularity**: Each component can be tested/used independently
3. **Documentation**: Every module has docstrings
4. **Reproducibility**: Config files for all experiments
5. **Production-Ready**: Logging, error handling, type hints

## Git Workflow

```bash
# Feature branches
git checkout -b feature/new-model

# Commit with conventional commits
git commit -m "feat: add ResNet50 baseline"
git commit -m "fix: correct preprocessing mismatch"
git commit -m "docs: update README with new results"

# Push and create PR
git push origin feature/new-model
```

## Data Not in Git

Large files are excluded via `.gitignore`:
- `checkpoints/*.pth` (use Git LFS or cloud storage)
- `data/` (download separately)
- `results/` (generated locally)

Use `scripts/download_weights.py` to fetch pre-trained models.
