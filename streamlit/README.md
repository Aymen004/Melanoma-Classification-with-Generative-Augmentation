# Breast Health Insight - Medical Image Classifier

A Streamlit web application for breast cancer detection using advanced AI models with explainable AI (XAI) capabilities through Grad-CAM visualization.

## 🎯 Features

- **Multi-Model Support**: Choose from 6 state-of-the-art architectures:
  - ResNet18 (CNN)
  - DenseNet121 (CNN)
  - Vision Transformer (ViT)
  - Swin Transformer
  - BioViT (Biomedical Vision Transformer)
  - MedViT (Medical Vision Transformer)

- **Multiple Data Sources**: Models trained on different data types:
  - Original medical images
  - DCGAN-generated synthetic images
  - DCGAN upscaled images
  - DDPM-generated synthetic images
  - DDPM upscaled images

- **Explainable AI**: Integrated Grad-CAM visualization to understand model decisions
- **Real-time Classification**: Upload images and get instant predictions
- **Performance Metrics**: View detailed accuracy, precision, recall, and F1-scores
- **Visual Features**: Extract and display key image characteristics
- **Responsive Design**: Modern, mobile-friendly interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager
- Trained model checkpoints (see Model Paths section)

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd /path/to/your/project/streamlit
   ```

2. **Install dependencies**
   ```bash
   pip install -r ../../requirements.txt
   ```

   Or install manually:
   ```bash
   pip install streamlit torch torchvision pillow pytorch-grad-cam
   ```

3. **Ensure model checkpoints are available**
   
   The app requires trained model files. Make sure the following paths exist (or update `MODEL_PATHS` in `app.py`):
   
   - ResNet18 models
   - DenseNet models
   - ViT models
   - Swin Transformer models
   - BioViT models
   - MedViT models

### Running the Application

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## 📖 Usage

1. **Select Model**: Choose from the available architectures in the sidebar
2. **Choose Data Source**: Select the training data type (Original, DCGAN, etc.)
3. **Upload Image**: Drag and drop or browse for a breast tissue image
4. **Get Prediction**: Click "Analyze Image" to see:
   - Classification result (Benign/Malignant)
   - Confidence scores
   - Grad-CAM heatmap overlay
   - Visual feature analysis

## 🏗️ Architecture

### Model Loading
- Custom wrapper classes for single-class output
- Automatic device selection (CPU/GPU)
- Checkpoint loading with error handling

### Grad-CAM Implementation
- Architecture-specific target layers
- Custom reshape functions for transformer models
- Multiple Grad-CAM variants (GradCAM, GradCAM++)

### Performance Data
- Real evaluation results from MILK10k dataset
- 30 different model configurations tested
- Comprehensive metrics tracking

## 📊 Supported Models

| Model | Architecture | Target Layer | Grad-CAM Type |
|-------|-------------|--------------|---------------|
| ResNet18 | CNN | layer4[-1] | GradCAM++ |
| DenseNet | CNN | denseblock4 | GradCAM++ |
| ViT | Transformer | blocks[-1].norm1 | GradCAM |
| Swin | Transformer | layers[-1].blocks[-1].norm1 | GradCAM++ |
| BioViT | Medical Transformer | blocks[-1].norm1 | GradCAM |
| MedViT | Medical Transformer | blocks[-1].norm1 | GradCAM |

## 🔧 Configuration

### Model Paths
Update the `MODEL_PATHS` dictionary in `app.py` to point to your trained model checkpoints.

### Customization
- Modify `TARGET_LAYERS` for different Grad-CAM target layers
- Adjust `GRADCAM_TYPES` for different visualization methods
- Update color schemes and styling in the CSS section

## 📈 Performance

The models have been evaluated on the MILK10k dataset with the following average performance:

- **Accuracy**: ~85-95% (varies by model and data source)
- **Precision**: ~82-94%
- **Recall**: ~84-96%
- **F1-Score**: ~83-95%

Detailed results available in `results/models_evaluation/milk10k_complete_evaluation_30_models_FINAL.csv`

## 🛠️ Development

### Project Structure
```
streamlit/
├── app.py              # Main Streamlit application
├── README.md           # This file
└── ...                 # Additional assets
```

### Adding New Models
1. Create a model loading function
2. Add entry to `MODEL_LOADERS`
3. Define target layer in `TARGET_LAYERS`
4. Specify Grad-CAM type in `GRADCAM_TYPES`
5. Update `MODEL_PATHS` with checkpoint locations

## ⚠️ Important Notes

- **Medical Disclaimer**: This tool is for research and educational purposes only. Not intended for clinical diagnosis.
- **Model Requirements**: Ensure all model checkpoints are accessible at the specified paths.
- **Image Format**: Supports common image formats (JPEG, PNG, etc.)
- **Performance**: GPU recommended for faster inference, especially with transformer models.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

See the main project LICENSE file for details.

## 🙏 Acknowledgments

- Built with Streamlit
- PyTorch for deep learning
- PyTorch Grad-CAM for explainability
- MILK10k dataset for evaluation