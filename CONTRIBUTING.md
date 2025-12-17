# Contributing to Melanoma Classification

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🌟 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- **Clear title and description**
- **Steps to reproduce** the problem
- **Expected behavior** vs **actual behavior**
- **System information** (OS, Python version, GPU)
- **Error messages** or logs

### Suggesting Features

Feature requests are welcome! Please:
- Check if the feature already exists or is planned
- Explain the problem it solves
- Provide examples of how it would be used
- Consider implementation complexity

### Pull Requests

We welcome PRs! Please follow this workflow:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with clear, commented code
4. **Test thoroughly** on your local setup
5. **Commit with conventional commits**: 
   - `feat: add new model architecture`
   - `fix: correct preprocessing bug`
   - `docs: update README with new results`
6. **Push to your fork**: `git push origin feature/amazing-feature`
7. **Open a Pull Request** with detailed description

## 🎯 Development Guidelines

### Code Style

- **Language**: All code, comments, and docstrings in **English**
- **Formatting**: Follow PEP 8 style guide
- **Type hints**: Use type annotations for function signatures
- **Docstrings**: Use Google-style docstrings

Example:
```python
def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    epochs: int = 100
) -> Dict[str, float]:
    """
    Train a PyTorch model.
    
    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        epochs: Number of training epochs (default: 100)
        
    Returns:
        Dictionary with training metrics (loss, accuracy)
        
    Raises:
        ValueError: If epochs < 1
    """
    # Implementation...
```

### Code Structure

- **Modularity**: Keep functions small and focused
- **DRY**: Don't repeat yourself — extract common logic
- **Error handling**: Use try-except with specific exceptions
- **Logging**: Use Python's `logging` module, not `print()`

### Testing

Before submitting:
- [ ] Code runs without errors
- [ ] No degradation in existing model performance
- [ ] New features have example usage
- [ ] Documentation updated (README, docstrings)

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: add ResNet152 architecture`
- `fix: resolve CUDA memory leak in VAE training`
- `docs: add installation guide for Windows`
- `refactor: simplify data loading pipeline`
- `test: add unit tests for hybrid classifier`
- `style: format code with black`

## 🧪 Testing Locally

### Setup Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/melanoma-classification.git
cd melanoma-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# With coverage report
pytest --cov=src tests/
```

### Testing the Hybrid System

```bash
# Download test weights
python scripts/download_weights.py --hybrid

# Run inference on sample image
python inference.py --image tests/fixtures/sample_lesion.jpg

# Expected output: prediction with rescue status
```

## 📝 Documentation

### Adding New Models

When adding a new model architecture:

1. **Create model file**: `src/models/your_model.py`
2. **Add docstring** explaining architecture
3. **Update README.md** with model description
4. **Add usage example** in docstring or notebook
5. **Create training script** in `scripts/`

### Adding Results

When reporting new experimental results:

1. **Reproduce experiments** 3+ times for consistency
2. **Document hyperparameters** fully
3. **Add confusion matrix** to `results/`
4. **Update performance table** in README
5. **Include GradCAM** visualizations if applicable

## 🎓 Research Contributions

If your contribution involves new research:

- **Cite relevant papers** in code comments
- **Explain methodology** clearly in docstrings
- **Provide mathematical notation** for complex algorithms
- **Compare with baselines** quantitatively

## 📊 Performance Benchmarks

When proposing changes that affect performance:

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Inference Time | 50ms | 45ms | -10% ✅ |
| Memory Usage | 1.2GB | 1.5GB | +25% ⚠️ |
| Recall | 79.32% | 99.75% | +20.43% 🚀 |

## 🤝 Code of Conduct

- **Be respectful** and constructive in discussions
- **Welcome newcomers** and help them get started
- **Focus on the code**, not the person
- **Assume good intentions**

## 🔒 Medical Ethics

This is a research project. Remember:

⚠️ **Medical Disclaimer**: This software is for research purposes only. It is NOT approved for clinical use or medical diagnosis. Always consult qualified healthcare professionals for medical advice.

When contributing:
- **Prioritize safety**: False negatives (missed cancers) are more critical than false positives
- **Document limitations** clearly
- **Test thoroughly** before claiming improvements
- **Consider bias**: Ensure diverse training data

## 📬 Contact

Questions? Reach out:
- **GitHub Issues**: [Create an issue](https://github.com/your-username/melanoma-classification/issues)
- **Email**: your.email@example.com
- **Discord**: [Join our community](https://discord.gg/your-invite)

## 🙏 Recognition

Contributors will be:
- Listed in README.md acknowledgments
- Mentioned in CHANGELOG.md
- Credited in academic papers using this code

---

**Thank you for making melanoma detection better! 🎗️**
