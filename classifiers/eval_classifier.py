"""
Unified Evaluation Pipeline for All Classifiers
================================================

This script provides comprehensive evaluation for all classification models:
- Performance metrics (accuracy, precision, recall, F1, AUC)
- Confusion matrix visualization
- ROC curve analysis
- Per-class metrics
- Prediction export

Usage:
    python eval_classifier.py --model densenet121 --checkpoint best_model.pth --data_csv test.csv --img_dir ./images
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, classification_report,
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Import model creation functions
from biovit import create_biovit_model
from densenet121 import create_densenet121_model
from medvit import create_medvit_model
from resnet50 import create_resnet50_model, create_resnet18_model
from swin import create_swin_model, create_swin_base
from vit_base import create_vit_model, create_vit_base


# =====================================================================
# DATASET CLASS
# =====================================================================

class SkinLesionDataset(Dataset):
    """Universal dataset class for skin lesion classification"""
    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
        # Auto-detect image column name
        self.img_col = None
        for col in ['new_name', 'filename', 'image_name', 'image_id']:
            if col in self.data.columns:
                self.img_col = col
                break
        
        if self.img_col is None:
            raise ValueError("No valid image column found in CSV")
        
        # Auto-detect label column
        self.label_col = 'label' if 'label' in self.data.columns else 'target'
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.loc[idx, self.img_col]
        label = int(self.data.loc[idx, self.label_col])
        
        # Try to find image with various extensions
        img_path = None
        extensions = ['', '.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
        
        for ext in extensions:
            test_path = os.path.join(self.img_dir, f"{img_name}{ext}")
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_name}")
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_name


# =====================================================================
# MODEL LOADING
# =====================================================================

def load_model(model_name, checkpoint_path, num_classes=2, device='cuda'):
    """
    Load trained model from checkpoint
    
    Args:
        model_name: Model architecture name
        checkpoint_path: Path to checkpoint file
        num_classes: Number of classes
        device: Device to load model on
    
    Returns:
        model: Loaded model in eval mode
    """
    # Create model
    model_name_lower = model_name.lower()
    
    if model_name_lower == 'biovit':
        model = create_biovit_model(num_classes=num_classes, pretrained=False, device=device)
    elif model_name_lower == 'densenet121':
        model = create_densenet121_model(num_classes=num_classes, pretrained=False, device=device)
    elif model_name_lower == 'medvit':
        model = create_medvit_model(num_classes=num_classes, pretrained=False, device=device)
    elif model_name_lower == 'resnet50':
        model = create_resnet50_model(num_classes=num_classes, pretrained=False, device=device)
    elif model_name_lower == 'resnet18':
        model = create_resnet18_model(num_classes=num_classes, pretrained=False, device=device)
    elif model_name_lower == 'swin':
        model = create_swin_base(num_classes=num_classes, pretrained=False, device=device)
    elif model_name_lower == 'vit':
        model = create_vit_base(num_classes=num_classes, pretrained=False, device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


# =====================================================================
# EVALUATION FUNCTIONS
# =====================================================================

def evaluate_model(model, dataloader, device, num_classes=2):
    """
    Evaluate model and return predictions and metrics
    
    Returns:
        all_labels: True labels
        all_preds: Predicted labels
        all_probs: Prediction probabilities
        img_names: Image filenames
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    img_names = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for images, labels, names in pbar:
            images = images.to(device)
            
            outputs = model(images)
            
            # Handle different output formats
            if outputs.shape[1] == 1:  # Binary single output
                probs = torch.sigmoid(outputs.squeeze())
                preds = (probs > 0.5).long()
            else:  # Multi-class or binary with 2 outputs
                probs = torch.softmax(outputs, dim=1)
                if num_classes == 2:
                    probs = probs[:, 1]  # Probability of positive class
                preds = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy() if probs.dim() == 1 else probs.cpu().numpy())
            img_names.extend(names)
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), img_names


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")


def plot_roc_curve(y_true, y_probs, save_path):
    """Plot and save ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to: {save_path}")
    
    return roc_auc


def print_metrics(y_true, y_pred, y_probs=None, class_names=None):
    """Print comprehensive evaluation metrics"""
    print(f"\n{'='*70}")
    print("EVALUATION METRICS")
    print(f"{'='*70}\n")
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(precision))]
    
    print(f"\nPer-Class Metrics:")
    print(f"{'-'*70}")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print(f"{'-'*70}")
    
    for i, name in enumerate(class_names):
        print(f"{name:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    # Macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro')
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted')
    
    print(f"{'-'*70}")
    print(f"{'Macro Average':<20} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f}")
    print(f"{'Weighted Average':<20} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f}")
    
    # AUC for binary classification
    if y_probs is not None and len(class_names) == 2:
        try:
            roc_auc = roc_auc_score(y_true, y_probs)
            print(f"\nAUC-ROC Score: {roc_auc:.4f}")
        except:
            print("\nAUC-ROC Score: Could not compute")
    
    print(f"\n{'-'*70}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print(f"{'='*70}\n")


def save_predictions(img_names, y_true, y_pred, y_probs, save_path):
    """Save predictions to CSV"""
    results_df = pd.DataFrame({
        'image_name': img_names,
        'true_label': y_true,
        'predicted_label': y_pred,
        'probability': y_probs,
        'correct': y_true == y_pred
    })
    
    results_df.to_csv(save_path, index=False)
    print(f"Predictions saved to: {save_path}")
    
    return results_df


# =====================================================================
# MAIN EVALUATION PIPELINE
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained classifier')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                       choices=['biovit', 'densenet121', 'medvit', 'resnet50', 'resnet18', 'swin', 'vit'],
                       help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Data arguments
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Path to CSV file with image names and labels')
    parser.add_argument('--img_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size for resizing')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Benign', 'Malignant'],
                       help='Class names for visualization')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Evaluation Configuration:")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Model: {args.model}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(args.data_csv)
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Create transform (no augmentation for evaluation)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = SkinLesionDataset(df, args.img_dir, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers,
                           pin_memory=True)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.model, args.checkpoint, args.num_classes, device)
    
    # Evaluate
    print("\nEvaluating model...")
    y_true, y_pred, y_probs, img_names = evaluate_model(model, dataloader, device, args.num_classes)
    
    # Print metrics
    print_metrics(y_true, y_pred, y_probs, args.class_names)
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, f'{args.model}_predictions.csv')
    save_predictions(img_names, y_true, y_pred, y_probs, predictions_path)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, f'{args.model}_confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, args.class_names, cm_path)
    
    # Plot ROC curve (binary classification only)
    if args.num_classes == 2:
        roc_path = os.path.join(args.output_dir, f'{args.model}_roc_curve.png')
        roc_auc = plot_roc_curve(y_true, y_probs, roc_path)
    
    # Save metrics summary
    metrics_summary = {
        'model': args.model,
        'checkpoint': args.checkpoint,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'num_samples': len(y_true),
        'class_distribution': df['label'].value_counts().to_dict()
    }
    
    if args.num_classes == 2:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        metrics_summary.update({
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(roc_auc) if args.num_classes == 2 else None
        })
    
    metrics_path = os.path.join(args.output_dir, f'{args.model}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    
    print(f"\nMetrics summary saved to: {metrics_path}")
    print(f"\n{'='*70}")
    print("Evaluation completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
