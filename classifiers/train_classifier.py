"""
Unified Training Pipeline for All Classifiers
==============================================

This script provides a unified training pipeline for all classification models:
- BioViT, DenseNet121, MedViT, ResNet50, Swin Transformer, ViT

Features:
- Flexible model selection
- Configurable hyperparameters
- Progressive unfreezing strategy
- Early stopping and checkpointing
- Comprehensive metrics and visualization
- Support for various dataset formats

Usage:
    python train_classifier.py --model densenet121 --data_csv train.csv --img_dir ./images
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Import model creation functions
from biovit import create_biovit_model, get_model_info as get_biovit_info
from densenet121 import create_densenet121_model, get_model_info as get_densenet_info
from medvit import create_medvit_model, get_model_info as get_medvit_info
from resnet50 import create_resnet50_model, create_resnet18_model, get_model_info as get_resnet_info
from swin import create_swin_model, create_swin_base, get_model_info as get_swin_info
from vit_base import create_vit_model, create_vit_base, get_model_info as get_vit_info


# =====================================================================
# DATASET CLASS
# =====================================================================

class SkinLesionDataset(Dataset):
    """
    Universal dataset class for skin lesion classification
    Handles various CSV column naming conventions
    """
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
        
        return image, label


# =====================================================================
# DATA TRANSFORMS
# =====================================================================

def get_transforms(img_size=224, augmentation='standard'):
    """
    Get train and validation transforms
    
    Args:
        img_size: Image size for resizing
        augmentation: 'none', 'light', 'standard', 'heavy'
    """
    # Base transforms (always applied)
    base_transform = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ]
    
    # Augmentation transforms
    if augmentation == 'none':
        aug_transforms = []
    elif augmentation == 'light':
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
        ]
    elif augmentation == 'standard':
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    elif augmentation == 'heavy':
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    else:
        aug_transforms = []
    
    # Combine transforms
    train_transform = transforms.Compose(aug_transforms + base_transform)
    val_transform = transforms.Compose(base_transform)
    
    return train_transform, val_transform


# =====================================================================
# MODEL FACTORY
# =====================================================================

def create_model(model_name, num_classes=2, pretrained=True, device='cuda'):
    """
    Factory function to create any model
    
    Args:
        model_name: 'biovit', 'densenet121', 'medvit', 'resnet50', 'resnet18', 'swin', 'vit'
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        device: Device to load model on
    
    Returns:
        model: Initialized model
    """
    model_name = model_name.lower()
    
    if model_name == 'biovit':
        model = create_biovit_model(num_classes=num_classes, pretrained=pretrained, device=device)
    elif model_name == 'densenet121':
        model = create_densenet121_model(num_classes=num_classes, pretrained=pretrained, device=device)
    elif model_name == 'medvit':
        model = create_medvit_model(num_classes=num_classes, pretrained=pretrained, device=device)
    elif model_name == 'resnet50':
        model = create_resnet50_model(num_classes=num_classes, pretrained=pretrained, device=device)
    elif model_name == 'resnet18':
        model = create_resnet18_model(num_classes=num_classes, pretrained=pretrained, device=device)
    elif model_name == 'swin':
        model = create_swin_base(num_classes=num_classes, pretrained=pretrained, device=device)
    elif model_name == 'vit':
        model = create_vit_base(num_classes=num_classes, pretrained=pretrained, device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


# =====================================================================
# TRAINING FUNCTIONS
# =====================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                if outputs.shape[1] == 1:  # Binary with single output
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, labels.float())
                else:  # Multi-class or binary with 2 outputs
                    loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels.float())
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        # Get predictions
        if outputs.dim() == 1:  # Binary single output
            preds = (torch.sigmoid(outputs) > 0.5).long()
        else:  # Multi-class or binary with 2 outputs
            preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            if outputs.shape[1] == 1:  # Binary single output
                outputs_squeezed = outputs.squeeze()
                loss = criterion(outputs_squeezed, labels.float())
                probs = torch.sigmoid(outputs_squeezed)
                preds = (probs > 0.5).long()
            else:  # Multi-class or binary with 2 outputs
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
                preds = torch.argmax(outputs, dim=1)
            
            running_loss += loss.item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs


# =====================================================================
# TRAINING PIPELINE
# =====================================================================

def train_model(config):
    """
    Main training pipeline
    
    Args:
        config: Dictionary with training configuration
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Model: {config['model_name']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['lr']}")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(config['data_csv'])
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=config['val_split'] + config['test_split'], 
                                          stratify=df['label'], random_state=42)
    val_size = config['val_split'] / (config['val_split'] + config['test_split'])
    val_df, test_df = train_test_split(temp_df, test_size=1-val_size, 
                                        stratify=temp_df['label'], random_state=42)
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)}")
    print(f"  Validation: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    
    # Create transforms and datasets
    train_transform, val_transform = get_transforms(
        img_size=config['img_size'],
        augmentation=config['augmentation']
    )
    
    train_dataset = SkinLesionDataset(train_df, config['img_dir'], train_transform)
    val_dataset = SkinLesionDataset(val_df, config['img_dir'], val_transform)
    test_dataset = SkinLesionDataset(test_df, config['img_dir'], val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=config['num_workers'], 
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=config['num_workers'], 
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                             shuffle=False, num_workers=config['num_workers'], 
                             pin_memory=True)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config['model_name'], num_classes=config['num_classes'], 
                        pretrained=config['pretrained'], device=device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    
    # Loss function
    if config['num_classes'] == 2 and config['loss'] == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                              weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], 
                               weight_decay=config['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], 
                             momentum=0.9, weight_decay=config['weight_decay'])
    
    # Scheduler
    if config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    elif config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] and torch.cuda.is_available() else None
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _, _ = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }
            torch.save(checkpoint, os.path.join(config['output_dir'], f"{config['model_name']}_best.pth"))
            print(f"✓ Saved best model (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], f"{config['model_name']}_training_curves.png"), dpi=300)
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"{'='*70}\n")
    
    return model, train_losses, val_losses, train_accs, val_accs


# =====================================================================
# MAIN FUNCTION
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='Train classifier for skin lesion detection')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='densenet121',
                       choices=['biovit', 'densenet121', 'medvit', 'resnet50', 'resnet18', 'swin', 'vit'],
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    # Data arguments
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Path to CSV file with image names and labels')
    parser.add_argument('--img_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size for resizing')
    parser.add_argument('--augmentation', type=str, default='standard',
                       choices=['none', 'light', 'standard', 'heavy'],
                       help='Data augmentation level')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--loss', type=str, default='ce',
                       choices=['ce', 'bce'],
                       help='Loss function (ce=CrossEntropy, bce=Binary CrossEntropy)')
    
    # Other arguments
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                       help='Test split ratio')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU if available')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create config dictionary
    config = vars(args)
    config['model_name'] = config.pop('model')
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Train model
    train_model(config)


if __name__ == "__main__":
    main()
