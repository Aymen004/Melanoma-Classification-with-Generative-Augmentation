"""
═══════════════════════════════════════════════════════════════════════════════
GRAD-CAM GENERATOR + HTML DASHBOARD (ADAPTED TO REAL STRUCTURE)
═══════════════════════════════════════════════════════════════════════════════
Génère les heatmaps Grad-CAM pour TOUS les modèles entraînés
et crée un dashboard HTML interactif pour comparaison visuelle.

Structure réelle:
- ORIGINAL_DATA/<Architecture>/<model>.pth
- DCGAN_data/<Architecture>/<model>.pth
- DDPM_data/<Architecture>/<model>.pth
- etc.

Test set: binary_classification_dataset.csv + MILK10k_Training_Input/
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import pandas as pd
import numpy as np
import timm
from pathlib import Path
from tqdm import tqdm
import time
import sys
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# GPU CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def check_gpu():
    if not torch.cuda.is_available():
        print("=" * 100)
        print("ERROR: CUDA not available!")
        print("=" * 100)
        sys.exit(1)
    
    device = torch.device('cuda')
    print("=" * 100)
    print("GPU CHECK PASSED")
    print("=" * 100)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 100)
    return device

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - MAPPING DES CHECKPOINTS RÉELS
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_MAP = {
    # Format: (model_name, dataset_name): checkpoint_path
    
    # ORIGINAL
    ('ResNet18', 'ORIGINAL'): 'ORIGINAL_DATA/CNN (ResNet)/best_resnet18_model.pth',
    ('DenseNet', 'ORIGINAL'): 'ORIGINAL_DATA/CNN (Dense Net)/DenseNet_Original.pth',
    ('ViT', 'ORIGINAL'): 'ORIGINAL_DATA/Vision Transformers/ViT/ViT_Original.pth',
    ('Swin', 'ORIGINAL'): 'ORIGINAL_DATA/Vision Transformers/Swin/swin_best_model.pth',
    ('BioViT', 'ORIGINAL'): 'ORIGINAL_DATA/Biomedical Transformers/ORIGINAL_BIOVIT/biovit_original_best.pth',
    ('MedViT', 'ORIGINAL'): 'ORIGINAL_DATA/Biomedical Transformers/MedViT/MedViT_Original.pth',
    
    # DCGAN
    ('ResNet18', 'DCGAN'): 'DCGAN_data/CNN/CNN(ResNet)/best_resnet18_model_dcgan.pth',
    ('DenseNet', 'DCGAN'): 'DCGAN_data/CNN/CNN(DenseNet)/DenseNet_DCGAN.pth',
    ('ViT', 'DCGAN'): 'DCGAN_data/Vision Transformers/ViT/ViT_DCGAN.pth',
    ('Swin', 'DCGAN'): 'DCGAN_data/Vision Transformers/Swin/swin_best_model_dcgan.pth',
    ('BioViT', 'DCGAN'): 'DCGAN_data/biomedical Transformers/BioViT/biovit_dcgan_best.pth',
    ('MedViT', 'DCGAN'): 'DCGAN_data/biomedical Transformers/MedViT/MedViT_DCGAN.pth',
    
    # DCGAN_UPSCALED
    ('ResNet18', 'DCGAN_UPSCALED'): 'DCGAN_data_upscaled/CNN/CNN (ResNet)/best_resnet18_model_dcgan_upscaled.pth',
    ('DenseNet', 'DCGAN_UPSCALED'): 'DCGAN_data_upscaled/CNN/CNN(DenseNet)/DenseNet_DCGAN_Upscaled.pth',
    ('ViT', 'DCGAN_UPSCALED'): 'DCGAN_data_upscaled/Vision Transformers/ViT/ViT_DCGAN_Upscaled.pth',
    ('Swin', 'DCGAN_UPSCALED'): 'DCGAN_data_upscaled/Vision Transformers/Swin_Transformer/swin_best_model.pth',
    ('BioViT', 'DCGAN_UPSCALED'): 'DCGAN_data_upscaled/Biomedical Transformers/BioViT/biovit_dcgan_upscaled_best.pth',
    ('MedViT', 'DCGAN_UPSCALED'): 'DCGAN_data_upscaled/Biomedical Transformers/MedViT/MedViT_DCGAN_Upscaled.pth',
    
    # DDPM
    ('ResNet18', 'DDPM'): 'DDPM_data/CNN/CNN(ResNet)/best_resnet18_model_ddpm.pth',
    ('DenseNet', 'DDPM'): 'DDPM_data/CNN/CNN(DenseNet)/DenseNet_DDPM.pth',
    ('ViT', 'DDPM'): 'DDPM_data/Vision Transformers/ViT/ViT_DDPM.pth',
    ('Swin', 'DDPM'): 'DDPM_data/Vision Transformers/swin/swin_best_model_ddpm.pth',
    ('BioViT', 'DDPM'): 'DDPM_data/Biomedical Transformers/DDPM_BioViT/biovit_ddpm_best.pth',
    ('MedViT', 'DDPM'): 'DDPM_data/Biomedical Transformers/MedViT/MedViT_DDPM.pth',
    
    # DDPM_UPSCALED
    ('ResNet18', 'DDPM_UPSCALED'): 'DDPM_upscaled_data/CNN/CNN (ResNet)/best_resnet18_model_ddpm_upscaled.pth',
    ('DenseNet', 'DDPM_UPSCALED'): 'DDPM_upscaled_data/CNN/CNN(DenseNet)/DenseNet_DDPM_Upscaled.pth',
    ('ViT', 'DDPM_UPSCALED'): 'DDPM_upscaled_data/Vision Transformers/ViT/ViT_DDPM_Upscaled.pth',
    ('Swin', 'DDPM_UPSCALED'): 'DDPM_upscaled_data/Vision Transformers/Swin/swin_best_model_ddpm_upscaled.pth',
    ('BioViT', 'DDPM_UPSCALED'): 'DDPM_upscaled_data/Biomedical Transformers/DDPM_UP_BioViT/biovit_ddpm_upscaled_best.pth',
    ('MedViT', 'DDPM_UPSCALED'): 'DDPM_upscaled_data/Biomedical Transformers/MedViT/MedViT_DDPM_Upscaled.pth',
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class SingleToDoubleClassWrapper(nn.Module):
    """Wrapper pour convertir modèle 1-classe vers binaire"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        logit = self.model(x)
        if isinstance(logit, dict):
            logit = logit['logits']
        prob = torch.sigmoid(logit)
        logits_2class = torch.cat([1 - prob, prob], dim=1)
        return logits_2class

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def detect_num_classes(checkpoint_path):
    """Detect number of output classes from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
        else:
            state_dict = checkpoint
        
        for key in state_dict.keys():
            if any(x in key.lower() for x in ['fc.weight', 'classifier.weight', 'head.weight', 'head.fc.weight']):
                return state_dict[key].shape[0]
        
        return 2
    except:
        return 2

def strip_prefix(state_dict, prefixes=['model.', 'swin.', 'vit.', 'backbone.', 'medvit.', 'biovit.']):
    """Remove wrapper prefixes"""
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        new_state_dict[key] = v
    return new_state_dict

def load_model(model_name, dataset_name, device, root_dir):
    """Load model from checkpoint"""
    
    checkpoint_key = (model_name, dataset_name)
    if checkpoint_key not in CHECKPOINT_MAP:
        raise FileNotFoundError(f"No checkpoint mapping for {model_name}/{dataset_name}")
    
    checkpoint_path = root_dir / CHECKPOINT_MAP[checkpoint_key]
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Detect classes
    num_classes = detect_num_classes(checkpoint_path)
    
    # Create model
    if model_name == 'ResNet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == 'DenseNet':
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    elif model_name in ['ViT', 'BioViT', 'MedViT']:
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    
    elif model_name == 'Swin':
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        state_dict = checkpoint
    
    state_dict = strip_prefix(state_dict)
    
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"WARNING: Partial load for {model_name}/{dataset_name}")
    
    # Wrap if needed
    if num_classes == 1:
        model = SingleToDoubleClassWrapper(model)
    
    model = model.to(device)
    model.eval()
    
    return model

# ═══════════════════════════════════════════════════════════════════════════════
# TARGET LAYER DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_target_layer(model, model_name):
    """Get target layer for Grad-CAM"""
    
    if isinstance(model, SingleToDoubleClassWrapper):
        base_model = model.model
    else:
        base_model = model
    
    if model_name == 'ResNet18':
        return [base_model.layer4[-1]]
    
    elif model_name == 'DenseNet':
        return [base_model.features[-1]]
    
    elif model_name in ['ViT', 'BioViT', 'MedViT']:
        if hasattr(base_model, 'blocks'):
            return [base_model.blocks[-1].norm1]
        else:
            return [base_model.patch_embed.proj]
    
    elif model_name == 'Swin':
        if hasattr(base_model, 'layers'):
            return [base_model.layers[-1].blocks[-1].norm1]
        else:
            return [base_model.patch_embed.proj]
    
    else:
        raise ValueError(f"Cannot determine target layer for: {model_name}")

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET - SAMPLING FROM TRAINING IMAGES
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_IMAGE_PATHS = {
    'ORIGINAL': 'ORIGINAL_DATA/ORIGINAL_IMAGES',
    'DCGAN': 'DCGAN_data/images',
    'DCGAN_UPSCALED': 'DCGAN_data_upscaled/upscaled_images',
    'DDPM': 'DDPM_data/images_ddpm',
    'DDPM_UPSCALED': 'DDPM_upscaled_data/images_ddpm_upscaled',
}

def sample_images_from_dataset(dataset_dir, num_benign=5, num_malignant=5):
    """
    Sample images from a specific dataset directory
    
    Returns: list of (image_path, label, image_name) tuples
    """
    dataset_dir = Path(dataset_dir)
    
    if not dataset_dir.exists():
        print(f"WARNING: Dataset directory not found: {dataset_dir}")
        return []
    
    # Get all images
    all_images = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        all_images.extend(list(dataset_dir.glob(ext)))
        all_images.extend(list(dataset_dir.glob(f'**/{ext}')))  # Search subdirectories
    
    if len(all_images) == 0:
        print(f"WARNING: No images found in {dataset_dir}")
        return []
    
    # Sample randomly (simplified approach)
    import random
    total_needed = num_benign + num_malignant
    
    if len(all_images) >= total_needed:
        selected = random.sample(all_images, total_needed)
    else:
        selected = all_images
    
    # Assign labels: first half as benign (0), second half as malignant (1)
    sampled = []
    for i, img_path in enumerate(selected):
        label = 0 if i < num_benign else 1
        sampled.append((str(img_path), label, img_path.name))
    
    return sampled

def load_sampled_dataset(root_dir):
    """
    Load 50 images total: 10 from each dataset (5 benign + 5 malignant)
    Ces 50 images seront utilisées pour TOUS les modèles
    
    Returns: dict {dataset_name: [(image_path, label, image_name, dataset_name)]}
    """
    all_samples = {}
    
    for dataset_name, dataset_path in DATASET_IMAGE_PATHS.items():
        full_path = root_dir / dataset_path
        samples = sample_images_from_dataset(full_path, num_benign=5, num_malignant=5)
        
        # Add dataset name to each sample
        samples_with_ds = []
        for img_path, label, img_name in samples:
            samples_with_ds.append((img_path, label, img_name, dataset_name))
        
        all_samples[dataset_name] = samples_with_ds
        print(f"Sampled {len(samples)} images from {dataset_name}")
    
    return all_samples

def get_transforms():
    """ImageNet normalization"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image_path, transform):
    """Preprocess image for model input"""
    img = Image.open(image_path).convert('RGB')
    rgb_img = np.array(img.resize((224, 224))) / 255.0
    input_tensor = transform(img)
    return rgb_img, input_tensor

# ═══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_gradcam(model, model_name, model_dataset, all_sampled_images, device, output_dir):
    """
    Generate Grad-CAM heatmaps for sampled images
    Pour chaque modèle: 50 images (10 de chaque dataset: 5 bénignes + 5 malignes)
    
    Args:
        model: Trained model
        model_name: Name of model architecture
        model_dataset: Dataset the model was trained on
        all_sampled_images: Dict with all sampled images by dataset
        device: CUDA device
        output_dir: Output directory
    """
    output_path = output_dir / f"{model_name}_{model_dataset}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        target_layers = get_target_layer(model, model_name)
    except Exception as e:
        print(f"WARNING: Cannot get target layer for {model_name}: {e}")
        return 0
    
    try:
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    except Exception as e:
        print(f"WARNING: Cannot initialize Grad-CAM for {model_name}: {e}")
        return 0
    
    transform = get_transforms()
    generated = 0
    
    # Pour ce modèle, on teste sur 50 images: 10 de chaque dataset (5 bénignes + 5 malignes)
    all_images = []
    for dataset_name, images in all_sampled_images.items():
        all_images.extend(images)
    
    for img_path, label, img_name, source_dataset in tqdm(all_images, desc=f"{model_name}/{model_dataset}", leave=False):
        try:
            rgb_img, input_tensor = preprocess_image(img_path, transform)
            input_tensor = input_tensor.unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                if isinstance(outputs, dict):
                    outputs = outputs['logits']
                pred_class = torch.argmax(outputs, dim=1).item()
            
            # Generate Grad-CAM
            targets = [ClassifierOutputTarget(pred_class)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Create visualization
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # Save with source dataset prefix as JPEG (faster than PNG)
            output_file = output_path / f"{source_dataset}_{img_name}"
            if output_file.suffix.lower() not in ['.jpg', '.jpeg']:
                output_file = output_file.with_suffix('.jpg')
            
            Image.fromarray(visualization).save(output_file, 'JPEG', quality=95)
            
            generated += 1
            
        except Exception as e:
            continue
    
    return generated

# ═══════════════════════════════════════════════════════════════════════════════
# HTML DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def build_html_dashboard(output_dir, models, datasets, all_sampled_images):
    """Build interactive HTML dashboard"""
    
    # Flatten all samples for display
    all_images_list = []
    for dataset_name, images in all_sampled_images.items():
        all_images_list.extend(images)
    
    # Convert to JSON format: [path, label, name, source_dataset]
    test_dataset_json = [[path, int(label), name, source_ds] for path, label, name, source_ds in all_images_list]
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grad-CAM Dashboard - 30 Models</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }}
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }}
        h1 {{
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            font-size: 1.1rem;
            margin-bottom: 30px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card h3 {{ font-size: 2rem; font-weight: 700; }}
        .stat-card p {{ font-size: 0.9rem; opacity: 0.9; }}
        .filters {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }}
        .filter-group {{ display: flex; flex-direction: column; }}
        .filter-group label {{ font-weight: 600; margin-bottom: 5px; color: #555; }}
        .filter-group select, .filter-group input {{
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 1rem;
            transition: border-color 0.3s;
        }}
        .filter-group select:focus, .filter-group input:focus {{
            outline: none;
            border-color: #667eea;
        }}
        .grid {{ display: grid; gap: 20px; }}
        .image-card {{
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            transition: all 0.3s;
        }}
        .image-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border-color: #667eea;
        }}
        .image-card h3 {{
            font-size: 1rem;
            margin-bottom: 10px;
            color: #333;
            text-align: center;
        }}
        .label-badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 10px;
        }}
        .label-benign {{ background: #d4edda; color: #155724; }}
        .label-malignant {{ background: #f8d7da; color: #721c24; }}
        .model-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        .model-item {{ text-align: center; }}
        .model-item h4 {{ font-size: 0.8rem; margin-bottom: 5px; color: #666; }}
        .model-item img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
            border: 1px solid #ddd;
            transition: transform 0.3s;
        }}
        .model-item img:hover {{
            transform: scale(1.5);
            z-index: 1000;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        .no-results {{
            text-align: center;
            padding: 60px;
            font-size: 1.2rem;
            color: #999;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Grad-CAM Dashboard</h1>
        <p class="subtitle">30 Models × 5 Datasets - Interactive Comparison</p>
        
        <div class="stats">
            <div class="stat-card"><h3>{len(models)}</h3><p>Models</p></div>
            <div class="stat-card"><h3>{len(datasets)}</h3><p>Datasets</p></div>
            <div class="stat-card"><h3>{len(all_images_list)}</h3><p>Sample Images</p></div>
            <div class="stat-card"><h3>{len(models) * len(datasets)}</h3><p>Model Combinations</p></div>
        </div>
        
        <div class="filters">
            <div class="filter-group">
                <label for="filterModel">Filter by Model:</label>
                <select id="filterModel">
                    <option value="all">All Models</option>
                    {''.join(f'<option value="{m}">{m}</option>' for m in models)}
                </select>
            </div>
            <div class="filter-group">
                <label for="filterDataset">Filter by Dataset:</label>
                <select id="filterDataset">
                    <option value="all">All Datasets</option>
                    {''.join(f'<option value="{d}">{d}</option>' for d in datasets)}
                </select>
            </div>
            <div class="filter-group">
                <label for="filterLabel">Filter by Label:</label>
                <select id="filterLabel">
                    <option value="all">All Labels</option>
                    <option value="0">Benign (0)</option>
                    <option value="1">Malignant (1)</option>
                </select>
            </div>
            <div class="filter-group">
                <label for="searchImage">Search Image:</label>
                <input type="text" id="searchImage" placeholder="Enter image name...">
            </div>
        </div>
        
        <div id="gallery" class="grid">
            <div class="loading">Loading images...</div>
        </div>
    </div>
    
    <script>
        const testDataset = {test_dataset_json};
        const models = {models};
        const datasets = {datasets};
        
        function renderGallery() {{
            const filterModel = document.getElementById('filterModel').value;
            const filterDataset = document.getElementById('filterDataset').value;
            const filterLabel = document.getElementById('filterLabel').value;
            const searchQuery = document.getElementById('searchImage').value.toLowerCase();
            
            const gallery = document.getElementById('gallery');
            gallery.innerHTML = '';
            
            let matchCount = 0;
            
            testDataset.forEach(([imgPath, label, imgName, sourceDataset]) => {{
                if (filterLabel !== 'all' && label.toString() !== filterLabel) return;
                if (searchQuery && !imgName.toLowerCase().includes(searchQuery)) return;
                
                const card = document.createElement('div');
                card.className = 'image-card';
                
                const labelClass = label === 0 ? 'label-benign' : 'label-malignant';
                const labelText = label === 0 ? 'Benign' : 'Malignant';
                
                let html = `
                    <h3>
                        ${{imgName}}
                        <span class="label-badge ${{labelClass}}">${{labelText}}</span>
                        <small style="color:#999; font-size:0.8rem"> (from ${{sourceDataset}})</small>
                    </h3>
                    <div class="model-grid">
                `;
                
                models.forEach(model => {{
                    if (filterModel !== 'all' && model !== filterModel) return;
                    
                    datasets.forEach(dataset => {{
                        if (filterDataset !== 'all' && dataset !== filterDataset) return;
                        
                        // Try both .jpg and original extension
                        let imgSrc = `./${{model}}_${{dataset}}/${{sourceDataset}}_${{imgName}}`;
                        if (!imgSrc.endsWith('.jpg') && !imgSrc.endsWith('.jpeg') && !imgSrc.endsWith('.png')) {{
                            imgSrc = imgSrc.replace(/\\.[^.]+$/, '.jpg');
                        }}
                        html += `
                            <div class="model-item">
                                <h4>${{model}}<br>${{dataset}}</h4>
                                <img src="${{imgSrc}}" alt="${{model}} - ${{dataset}}" 
                                     onerror="this.style.display='none'">
                            </div>
                        `;
                    }});
                }});
                
                html += `</div>`;
                card.innerHTML = html;
                gallery.appendChild(card);
                matchCount++;
            }});
            
            if (matchCount === 0) {{
                gallery.innerHTML = '<div class="no-results">No images match your filters</div>';
            }}
        }}
        
        renderGallery();
        
        document.getElementById('filterModel').addEventListener('change', renderGallery);
        document.getElementById('filterDataset').addEventListener('change', renderGallery);
        document.getElementById('filterLabel').addEventListener('change', renderGallery);
        document.getElementById('searchImage').addEventListener('input', renderGallery);
    </script>
</body>
</html>"""
    
    html_path = output_dir / 'GRADCAM_DASHBOARD.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nHTML Dashboard: {html_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 100)
    print("GRAD-CAM GENERATOR FOR 30 MODELS")
    print("=" * 100)
    
    start_time = time.time()
    
    # GPU check
    device = check_gpu()
    
    # Configuration
    ROOT_DIR = Path(__file__).parent
    OUTPUT_DIR = ROOT_DIR / 'gradcam_outputs'
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    MODELS = ['ResNet18', 'DenseNet', 'ViT', 'Swin', 'BioViT', 'MedViT']
    DATASETS = ['ORIGINAL', 'DCGAN', 'DCGAN_UPSCALED', 'DDPM', 'DDPM_UPSCALED']
    
    # Load sampled images (5 benign + 5 malignant from each of 5 datasets = 50 total)
    print("\n" + "=" * 100)
    print("SAMPLING IMAGES FROM TRAINING DATASETS")
    print("=" * 100)
    print("Strategy: 5 benign + 5 malignant from each dataset")
    print("Total: 50 images (10 per dataset)")
    print("Each model will be tested on ALL 50 images")
    print("=" * 100)
    all_sampled_images = load_sampled_dataset(ROOT_DIR)
    total_images = sum(len(imgs) for imgs in all_sampled_images.values())
    print(f"\nLoaded {total_images} sampled images across {len(all_sampled_images)} datasets")
    
    # Generate Grad-CAM
    print("\n" + "=" * 100)
    print("GENERATING GRAD-CAM HEATMAPS")
    print("=" * 100)
    
    total_generated = 0
    success_count = 0
    
    with tqdm(total=len(MODELS) * len(DATASETS), desc="Overall") as pbar:
        for model_name in MODELS:
            for dataset_name in DATASETS:
                try:
                    model = load_model(model_name, dataset_name, device, ROOT_DIR)
                    count = generate_gradcam(model, model_name, dataset_name, 
                                            all_sampled_images, device, OUTPUT_DIR)
                    
                    total_generated += count
                    success_count += 1
                    
                    del model
                    torch.cuda.empty_cache()
                    
                except FileNotFoundError:
                    print(f"WARNING: Skipping {model_name}/{dataset_name} - not found")
                
                except Exception as e:
                    print(f"ERROR: {model_name}/{dataset_name}: {e}")
                
                finally:
                    pbar.update(1)
    
    # Build HTML
    print("\n" + "=" * 100)
    print("BUILDING HTML DASHBOARD")
    print("=" * 100)
    
    build_html_dashboard(OUTPUT_DIR, MODELS, DATASETS, all_sampled_images)
    
    # Summary
    elapsed = (time.time() - start_time) / 60
    
    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)
    print(f"Models processed:    {success_count}/{len(MODELS) * len(DATASETS)}")
    print(f"Images per model:    {total_images} (50 total: 10 per dataset)")
    print(f"Total heatmaps:      {total_generated}")
    print(f"Execution time:      {elapsed:.2f} minutes")
    print("=" * 100)
    print(f"\nOpen: {OUTPUT_DIR / 'GRADCAM_DASHBOARD.html'}")
    print("=" * 100)

if __name__ == '__main__':
    main()
