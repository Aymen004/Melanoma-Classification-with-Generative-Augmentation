"""
Script d'évaluation unifié pour tous les 30 modèles sur MILK10k
Version corrigée avec gestion automatique des wrappers de state_dict
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import timm
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Chemins
ROOT_DIR = Path(r'C:\Users\AzComputer\Documents\projects\audio-classifcation\audio classification\Classification')
MILK10K_CSV = ROOT_DIR / 'binary_classification_dataset.csv'
MILK10K_IMAGES = ROOT_DIR / 'MILK10k_Training_Input' / 'MILK10k_Training_Input'

# ============================================================================
# DATASET
# ============================================================================
class MILK10kTestDataset(Dataset):
    """Dataset pour MILK10k test set"""
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['id']
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ============================================================================
# TRANSFORMATIONS
# ============================================================================
def get_test_transforms():
    """Transformations identiques pour tous les modèles"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def strip_prefix(state_dict, prefixes):
    """Remove specific prefixes from state_dict keys"""
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        new_state_dict[key] = v
    return new_state_dict

def load_checkpoint_smart(checkpoint_path):
    """Load checkpoint and extract state_dict intelligently"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        # Try common keys
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in checkpoint:
                return checkpoint[key]
        # If no special key, assume entire dict is state_dict
        return checkpoint
    else:
        return checkpoint

# ============================================================================
# MODEL LOADERS WITH AUTOMATIC WRAPPER HANDLING
# ============================================================================
def load_resnet18(checkpoint_path):
    """Load ResNet18 model"""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    state_dict = load_checkpoint_smart(checkpoint_path)
    
    # Remove common prefixes
    state_dict = strip_prefix(state_dict, ['model.', 'resnet.', 'resnet18.'])
    
    model.load_state_dict(state_dict, strict=False)
    return model

def load_densenet(checkpoint_path):
    """Load DenseNet model"""
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    
    state_dict = load_checkpoint_smart(checkpoint_path)
    
    # Remove wrapper prefixes
    state_dict = strip_prefix(state_dict, ['densenet.', 'model.'])
    
    # Handle multi-layer classifier (drop intermediate layers)
    if any('classifier.0.' in k for k in state_dict.keys()):
        # Has multi-layer classifier - use only final layer
        new_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('classifier.'):
                new_dict[k] = v
            # Skip intermediate classifier layers
        state_dict = new_dict
    
    model.load_state_dict(state_dict, strict=False)
    return model

def load_vit(checkpoint_path):
    """Load ViT model"""
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
    
    state_dict = load_checkpoint_smart(checkpoint_path)
    
    # Remove wrapper prefixes
    state_dict = strip_prefix(state_dict, ['vit.', 'model.', 'vision_transformer.'])
    
    model.load_state_dict(state_dict, strict=False)
    return model

def load_swin(checkpoint_path):
    """Load Swin Transformer model"""
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=2)
    
    state_dict = load_checkpoint_smart(checkpoint_path)
    
    # Remove wrapper prefixes
    state_dict = strip_prefix(state_dict, ['swin.', 'model.', 'swin_transformer.'])
    
    model.load_state_dict(state_dict, strict=False)
    return model

def load_biovit(checkpoint_path):
    """Load BioViT model (Hugging Face architecture)"""
    from transformers import ViTForImageClassification
    
    # BioViT uses Hugging Face ViT architecture
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    state_dict = load_checkpoint_smart(checkpoint_path)
    
    # BioViT uses 'vit.' prefix with HF structure
    # Try to load with different prefix patterns
    prefixes_to_try = ['', 'vit.', 'model.']
    
    for prefix in prefixes_to_try:
        try:
            test_dict = strip_prefix(state_dict, [prefix]) if prefix else state_dict
            model.load_state_dict(test_dict, strict=False)
            return model
        except:
            continue
    
    # If all fail, try direct load
    model.load_state_dict(state_dict, strict=False)
    return model

def load_medvit(checkpoint_path):
    """Load MedViT model (backbone + classifier)"""
    # MedViT uses ViT backbone with custom classifier
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    
    # Modify classifier
    model.head = nn.Sequential(
        nn.Linear(model.head.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    
    state_dict = load_checkpoint_smart(checkpoint_path)
    
    # Remove backbone prefix
    if any(k.startswith('backbone.') for k in state_dict.keys()):
        new_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                new_dict[k[9:]] = v  # Remove 'backbone.'
            elif k.startswith('classifier.'):
                # Map classifier layers to head
                layer_num = k.split('.')[1]
                if layer_num in ['0', '2', '5']:  # Linear layers
                    new_key = k.replace('classifier.', 'head.')
                    new_dict[new_key] = v
        state_dict = new_dict
    
    model.load_state_dict(state_dict, strict=False)
    return model

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    # Handle zero division
    precision = precision_score(all_labels, all_preds, zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
    
    return {
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
        'Accuracy': round(accuracy, 2),
        'Precision': round(precision, 2),
        'Recall': round(recall, 2),
        'F1-Score': round(f1, 2)
    }

# ============================================================================
# MODELS CONFIGURATION
# ============================================================================
MODELS_CONFIG = {
    # ORIGINAL DATASET
    'Original - ResNet18': {
        'loader': load_resnet18,
        'checkpoint': ROOT_DIR / r'ORIGINAL_DATA\CNN\CNN (ResNet)\best_resnet18_model.pth'
    },
    'Original - DenseNet': {
        'loader': load_densenet,
        'checkpoint': ROOT_DIR / r'ORIGINAL_DATA\CNN\CNN (Dense Net)\densenet_classifier_original.pth'
    },
    'Original - ViT': {
        'loader': load_vit,
        'checkpoint': ROOT_DIR / r'ORIGINAL_DATA\Vision Transformers\ViT\ViT_ORIGINAL.pth'
    },
    'Original - Swin Transformer': {
        'loader': load_swin,
        'checkpoint': ROOT_DIR / r'ORIGINAL_DATA\Vision Transformers\Swin\swin_best_model_original.pth'
    },
    'Original - BioViT': {
        'loader': load_biovit,
        'checkpoint': ROOT_DIR / r'ORIGINAL_DATA\Biomedical Transformers\ORIGINAL_BIOVIT\biovit_original_best.pth'
    },
    'Original - MedViT': {
        'loader': load_medvit,
        'checkpoint': ROOT_DIR / r'ORIGINAL_DATA\Biomedical Transformers\MedViT\medvit_classifier_original.pth'
    },
    
    # DCGAN DATASET
    'DCGAN - ResNet18': {
        'loader': load_resnet18,
        'checkpoint': ROOT_DIR / r'DCGAN_data\CNN\CNN(ResNet)\best_resnet18_model_dcgan.pth'
    },
    'DCGAN - DenseNet': {
        'loader': load_densenet,
        'checkpoint': ROOT_DIR / r'DCGAN_data\CNN\CNN(DenseNet)\DenseNet_DCGAN.pth'
    },
    'DCGAN - ViT': {
        'loader': load_vit,
        'checkpoint': ROOT_DIR / r'DCGAN_data\Vision Transformers\ViT\ViT_DCGAN.pth'
    },
    'DCGAN - Swin Transformer': {
        'loader': load_swin,
        'checkpoint': ROOT_DIR / r'DCGAN_data\Vision Transformers\Swin\swin_best_model_dcgan.pth'
    },
    'DCGAN - BioViT': {
        'loader': load_biovit,
        'checkpoint': ROOT_DIR / r'DCGAN_data\biomedical Transformers\BioViT\biovit_dcgan_best_continued.pth'
    },
    'DCGAN - MedViT': {
        'loader': load_medvit,
        'checkpoint': ROOT_DIR / r'DCGAN_data\biomedical Transformers\MedViT\MedViT_DCGAN.pth'
    },
    
    # DCGAN_UPSCALED DATASET
    'DCGAN_UPSCALED - ResNet18': {
        'loader': load_resnet18,
        'checkpoint': ROOT_DIR / r'DCGAN_data_upscaled\CNN\CNN (ResNet)\best_resnet18_model_dcgan_upscaled.pth'
    },
    'DCGAN_UPSCALED - DenseNet': {
        'loader': load_densenet,
        'checkpoint': ROOT_DIR / r'DCGAN_data_upscaled\CNN\CNN(DenseNet)\densenet_classifier_dcgan_upscaled.pth'
    },
    'DCGAN_UPSCALED - ViT': {
        'loader': load_vit,
        'checkpoint': ROOT_DIR / r'DCGAN_data_upscaled\Vision Transformers\ViT\vit_dcgan_upscaled.pth'
    },
    'DCGAN_UPSCALED - Swin Transformer': {
        'loader': load_swin,
        'checkpoint': ROOT_DIR / r'DCGAN_data_upscaled\Vision Transformers\Swin_Transformer\swin_dcgan_upscaled.pth'
    },
    'DCGAN_UPSCALED - BioViT': {
        'loader': load_biovit,
        'checkpoint': ROOT_DIR / r'DCGAN_data_upscaled\Biomedical Transformers\BioViT\biovit_dcgan_upscaled_best.pth'
    },
    'DCGAN_UPSCALED - MedViT': {
        'loader': load_medvit,
        'checkpoint': ROOT_DIR / r'DCGAN_data_upscaled\Biomedical Transformers\MedViT\medvit_classifier_dcgan_upscaled.pth'
    },
    
    # DDPM DATASET
    'DDPM - ResNet18': {
        'loader': load_resnet18,
        'checkpoint': ROOT_DIR / r'DDPM_data\CNN\CNN(ResNet)\best_resnet18_model_ddpm.pth'
    },
    'DDPM - DenseNet': {
        'loader': load_densenet,
        'checkpoint': ROOT_DIR / r'DDPM_data\CNN\CNN(DenseNet)\densenet_classifier_ddpm.pth'
    },
    'DDPM - ViT': {
        'loader': load_vit,
        'checkpoint': ROOT_DIR / r'DDPM_data\Vision Transformers\ViT\vit_ddpm.pth'
    },
    'DDPM - Swin Transformer': {
        'loader': load_swin,
        'checkpoint': ROOT_DIR / r'DDPM_data\Vision Transformers\swin\swin_best_model_ddpm.pth'
    },
    'DDPM - BioViT': {
        'loader': load_biovit,
        'checkpoint': ROOT_DIR / r'DDPM_data\Biomedical Transformers\DDPM_BioViT\biovit_ddpm_best.pth'
    },
    'DDPM - MedViT': {
        'loader': load_medvit,
        'checkpoint': ROOT_DIR / r'DDPM_data\Biomedical Transformers\MedViT\medvit_ddpm.pth'
    },
    
    # DDPM_UPSCALED DATASET
    'DDPM_UPSCALED - ResNet18': {
        'loader': load_resnet18,
        'checkpoint': ROOT_DIR / r'DDPM_upscaled_data\CNN\CNN (ResNet)\best_resnet18_model_ddpm_upscaled.pth'
    },
    'DDPM_UPSCALED - DenseNet': {
        'loader': load_densenet,
        'checkpoint': ROOT_DIR / r'DDPM_upscaled_data\CNN\CNN(DenseNet)\densenet_classifier_ddpm_upscaled.pth'
    },
    'DDPM_UPSCALED - ViT': {
        'loader': load_vit,
        'checkpoint': ROOT_DIR / r'DDPM_upscaled_data\Vision Transformers\ViT\vit_ddpm_upscaled.pth'
    },
    'DDPM_UPSCALED - Swin Transformer': {
        'loader': load_swin,
        'checkpoint': ROOT_DIR / r'DDPM_upscaled_data\Vision Transformers\Swin\swin_ddpm_upscaled.pth'
    },
    'DDPM_UPSCALED - BioViT': {
        'loader': load_biovit,
        'checkpoint': ROOT_DIR / r'DDPM_upscaled_data\Biomedical Transformers\DDPM_UP_BioViT\biovit_ddpm_upscaled_best.pth'
    },
    'DDPM_UPSCALED - MedViT': {
        'loader': load_medvit,
        'checkpoint': ROOT_DIR / r'DDPM_upscaled_data\Biomedical Transformers\MedViT\medvit_ddpm_upscaled.pth'
    },
}

# ============================================================================
# MAIN EVALUATION
# ============================================================================
def main():
    print("=" * 100)
    print("ÉVALUATION DE TOUS LES MODÈLES SUR MILK10K TEST SET")
    print("=" * 100)
    
    # Load test dataset (same for all models)
    print("\nChargement du dataset MILK10k...")
    test_dataset = MILK10kTestDataset(
        csv_path=MILK10K_CSV,
        img_dir=MILK10K_IMAGES,
        transform=get_test_transforms()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,  # Désactivé pour éviter les problèmes Windows multiprocessing
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Dataset charge: {len(test_dataset)} images\n")
    
    # Evaluate all models
    results = []
    
    for i, (model_name, config) in enumerate(MODELS_CONFIG.items(), 1):
        print("=" * 100)
        print(f"[{i}/30] {model_name}")
        print("=" * 100)
        
        checkpoint_path = config['checkpoint']
        
        # Check if checkpoint exists
        if not checkpoint_path.exists():
            print(f"WARNING: Checkpoint non trouve: {checkpoint_path}")
            results.append({
                'Model': model_name,
                'TN': 'N/A',
                'FP': 'N/A',
                'FN': 'N/A',
                'TP': 'N/A',
                'Accuracy': 'N/A',
                'Precision': 'N/A',
                'Recall': 'N/A',
                'F1-Score': 'N/A',
                'Status': 'Checkpoint manquant'
            })
            continue
        
        try:
            print("Chargement du modele...")
            model = config['loader'](checkpoint_path)
            model = model.to(device)
            
            print("Evaluation en cours...")
            metrics = evaluate_model(model, test_loader)
            
            result = {
                'Model': model_name,
                **metrics,
                'Status': '✓ Évalué'
            }
            
            results.append(result)
            print(f"Accuracy: {metrics['Accuracy']}% | F1-Score: {metrics['F1-Score']}%")
            
            # Free memory
            del model
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"ERROR lors de l'evaluation: {error_msg}")
            results.append({
                'Model': model_name,
                'TN': 'N/A',
                'FP': 'N/A',
                'FN': 'N/A',
                'TP': 'N/A',
                'Accuracy': 'N/A',
                'Precision': 'N/A',
                'Recall': 'N/A',
                'F1-Score': 'N/A',
                'Status': f'Erreur: {error_msg}'
            })
    
    # Save results
    print("\n" + "=" * 100)
    print("SAUVEGARDE DES RÉSULTATS")
    print("=" * 100)
    
    results_df = pd.DataFrame(results)
    output_path = ROOT_DIR / 'milk10k_evaluation_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Resultats sauvegardes: {output_path}")
    
    # Display summary
    print("\n" + "=" * 100)
    print("RÉSUMÉ DES ÉVALUATIONS")
    print("=" * 100)
    print(results_df.to_string(index=False))
    
    success_count = len([r for r in results if r['Status'] == '✓ Évalué'])
    print(f"\nStatistiques:")
    print(f"   Total de modèles: {len(results)}")
    print(f"   Évalués avec succès: {success_count}")
    print(f"   Échoués: {len(results) - success_count}")

if __name__ == '__main__':
    main()
