"""
DenseNet for Dermatoscopic Lesion Classification - DCGAN Data
==============================================================

DenseNet (Densely Connected Convolutional Networks) offers several advantages for medical imaging:

Why DenseNet for Medical Images:
---------------------------------
1. Dense Connectivity: Each layer receives feature maps from all preceding layers, 
   enabling better feature reuse and gradient flow.
2. Parameter Efficiency: Fewer parameters than ResNet while maintaining performance.
3. Feature Propagation: Direct connections from early to late layers help preserve 
   low-level features (textures, edges) crucial for lesion analysis.
4. Alleviation of Vanishing Gradient: Dense connections facilitate gradient flow during training.
5. Regularization Effect: Implicit deep supervision reduces overfitting on small medical datasets.

Architecture Comparison:
-----------------------
- DenseNet121: 8M parameters (best for limited GPU memory)
- DenseNet169: 14M parameters (balanced performance/memory)
- DenseNet201: 20M parameters (highest capacity)

This implementation uses DenseNet121 pretrained on ImageNet with progressive unfreezing
and fine-tuning strategy optimized for medical image classification.

DCGAN Data Configuration:
-------------------------
- Dataset: 12,179 synthetic images (balanced)
- Split: 70/20/10 (train/val/test)
- Source: DCGAN synthetic data
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
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


# =====================================================================
# 1. DATASET CLASS
# =====================================================================

class DermatoscopicDataset(Dataset):
    """Dataset for dermatoscopic lesion images"""
    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
        # Determine which column contains image filenames
        self.img_col = None
        for col in ['new_name', 'filename', 'image_name']:
            if col in self.data.columns:
                self.img_col = col
                break
        
        if self.img_col is None:
            raise ValueError("No valid image column found in dataframe")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.loc[idx, self.img_col]
        
        # Check if extension already exists
        if any(img_name.lower().endswith(ext) for ext in ['.jpg', '.png', '.jpeg']):
            img_path = os.path.join(self.img_dir, img_name)
        else:
            # Try different extensions
            img_path = None
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                potential_path = os.path.join(self.img_dir, f"{img_name}{ext}")
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
        
        if img_path is None or not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_name}")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.loc[idx, 'label'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# =====================================================================
# 2. DATA TRANSFORMS
# =====================================================================

def get_transforms(img_size=224, augmentation=True):
    """
    Get data transforms for training and validation.
    
    Args:
        img_size: target image size
        augmentation: whether to apply data augmentation
    
    Returns:
        train_transform, val_test_transform
    """
    # ImageNet normalization (standard for pretrained models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if augmentation:
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # No augmentation
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    # Validation/test transforms (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_test_transform


# =====================================================================
# 3. DENSENET MODEL
# =====================================================================

class DenseNetClassifier(nn.Module):
    """
    DenseNet-based classifier for binary lesion classification.
    
    Features:
    - Pretrained DenseNet backbone (121, 169, or 201)
    - Custom classification head with dropout
    - Progressive unfreezing capability for fine-tuning
    """
    def __init__(self, model_name='densenet121', num_classes=2, dropout=0.5, pretrained=True):
        """
        Args:
            model_name: 'densenet121', 'densenet169', or 'densenet201'
            num_classes: number of output classes
            dropout: dropout rate in classification head
            pretrained: whether to use ImageNet pretrained weights
        """
        super(DenseNetClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained DenseNet
        if model_name == 'densenet121':
            self.densenet = models.densenet121(pretrained=pretrained)
            self.num_features = 1024
        elif model_name == 'densenet169':
            self.densenet = models.densenet169(pretrained=pretrained)
            self.num_features = 1664
        elif model_name == 'densenet201':
            self.densenet = models.densenet201(pretrained=pretrained)
            self.num_features = 1920
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Remove original classifier
        self.densenet.classifier = nn.Identity()
        
        # Custom classification head optimized for medical imaging
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
        
        print(f"✅ Loaded {model_name} with {self.num_features} features")
    
    def _init_classifier(self):
        """Initialize classification head weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass"""
        features = self.densenet(x)
        logits = self.classifier(features)
        return logits
    
    def freeze_backbone(self):
        """Freeze all DenseNet backbone layers"""
        for param in self.densenet.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze all DenseNet backbone layers"""
        for param in self.densenet.parameters():
            param.requires_grad = True
        print("🔓 Backbone unfrozen")
    
    def unfreeze_last_blocks(self, num_blocks=2):
        """
        Unfreeze last N dense blocks for fine-tuning.
        
        Args:
            num_blocks: number of dense blocks to unfreeze (1-4)
        """
        # First freeze all
        self.freeze_backbone()
        
        # Then unfreeze specific blocks
        blocks_to_unfreeze = []
        if hasattr(self.densenet.features, 'denseblock4'):
            blocks_to_unfreeze.append('denseblock4')
        if num_blocks >= 2 and hasattr(self.densenet.features, 'denseblock3'):
            blocks_to_unfreeze.append('denseblock3')
        if num_blocks >= 3 and hasattr(self.densenet.features, 'denseblock2'):
            blocks_to_unfreeze.append('denseblock2')
        if num_blocks >= 4 and hasattr(self.densenet.features, 'denseblock1'):
            blocks_to_unfreeze.append('denseblock1')
        
        for block_name in blocks_to_unfreeze:
            block = getattr(self.densenet.features, block_name)
            for param in block.parameters():
                param.requires_grad = True
        
        # Also unfreeze batch norm and transition layers after unfrozen blocks
        unfreeze_next = False
        for name, module in self.densenet.features.named_children():
            if name in blocks_to_unfreeze:
                unfreeze_next = True
            if unfreeze_next:
                for param in module.parameters():
                    param.requires_grad = True
        
        print(f"🔓 Unfroze last {num_blocks} dense block(s): {blocks_to_unfreeze}")


# =====================================================================
# 4. TRAINING FUNCTIONS
# =====================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with mixed precision support"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="🏋️  Training", leave=False)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="🔍 Validating", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(all_labels)
    epoch_acc = 100 * accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs


# =====================================================================
# 5. VISUALIZATION FUNCTIONS
# =====================================================================

def plot_training_history(history, save_path='training_history_densenet_dcgan.png'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss - DCGAN')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy - DCGAN')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 0].plot(history['lr'], marker='o', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision & Recall
    axes[1, 1].plot(history['precision'], label='Precision', marker='o')
    axes[1, 1].plot(history['recall'], label='Recall', marker='s')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision and Recall - DCGAN')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 Training history saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix_densenet_dcgan.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized, 
        annot=np.array([[f'{cm[i,j]}\n({cm_normalized[i,j]:.2%})' 
                        for j in range(len(class_names))] 
                       for i in range(len(class_names))]),
        fmt='',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Frequency'},
        square=True,
        linewidths=1,
        linecolor='gray'
    )
    
    plt.title('Confusion Matrix - DenseNet DCGAN\n(Counts and Percentages)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix saved to: {save_path}")
    plt.close()


def plot_roc_curve(labels, probs, save_path='roc_curve_densenet_dcgan.png'):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve - DenseNet DCGAN', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📉 ROC curve saved to: {save_path}")
    plt.close()
    
    return roc_auc


# =====================================================================
# 6. MAIN TRAINING PIPELINE
# =====================================================================

def load_and_split_data(csv_path, test_size=0.10, val_size=0.20, random_state=42):
    """Load and split data"""
    print(f"\n{'='*80}")
    print("📊 LOADING AND SPLITTING DATA")
    print(f"{'='*80}\n")
    
    df = pd.read_csv(csv_path)
    print(f"✅ Total samples: {len(df)}")
    
    print(f"\n📈 Class distribution:")
    for label in sorted(df['label'].unique()):
        count = len(df[df['label'] == label])
        percentage = 100 * count / len(df)
        print(f"   Class {label}: {count} samples ({percentage:.1f}%)")
    
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=random_state
    )
    
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio, stratify=train_val_df['label'], random_state=random_state
    )
    
    print(f"\n✂️  Dataset splits:")
    print(f"   Training:   {len(train_df):,} samples ({100*len(train_df)/len(df):.1f}%)")
    print(f"   Validation: {len(val_df):,} samples ({100*len(val_df)/len(df):.1f}%)")
    print(f"   Test:       {len(test_df):,} samples ({100*len(test_df)/len(df):.1f}%)")
    print(f"{'='*80}\n")
    
    return train_df, val_df, test_df


def train_densenet(config):
    """Main training function"""
    print("\n" + "="*80)
    print("🌳 DENSENET DCGAN LESION CLASSIFIER")
    print("="*80)
    
    print(f"\n⚙️  Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load data
    train_df, val_df, test_df = load_and_split_data(
        config['csv_path'], 
        test_size=config.get('test_size', 0.10),
        val_size=config.get('val_size', 0.20)
    )
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(
        img_size=config.get('img_size', 224),
        augmentation=config.get('augmentation', True)
    )
    
    # Create datasets
    print("🔄 Creating datasets...")
    train_dataset = DermatoscopicDataset(train_df, config['img_dir'], transform=train_transform)
    val_dataset = DermatoscopicDataset(val_df, config['img_dir'], transform=val_test_transform)
    test_dataset = DermatoscopicDataset(test_df, config['img_dir'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.get('batch_size', 32), 
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.get('batch_size', 32), 
        shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.get('batch_size', 32), 
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    print(f"✅ Datasets created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\n{'='*80}")
    print("🤖 INITIALIZING DENSENET MODEL")
    print(f"{'='*80}\n")
    
    model = DenseNetClassifier(
        model_name=config.get('model_name', 'densenet121'),
        num_classes=config.get('num_classes', 2),
        dropout=config.get('dropout', 0.5),
        pretrained=True
    )
    model = model.to(device)
    
    # Freeze backbone initially
    if config.get('freeze_backbone', True):
        model.freeze_backbone()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    early_stopping = EarlyStopping(patience=config.get('patience', 10), mode='max')
    scaler = torch.cuda.amp.GradScaler() if config.get('mixed_precision', True) and torch.cuda.is_available() else None
    
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'lr': [], 'precision': [], 'recall': []
    }
    
    best_val_acc = 0.0
    unfreeze_epoch = config.get('unfreeze_epoch', 5)
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"🚀 STARTING TRAINING - {config.get('epochs', 30)} EPOCHS")
    print(f"{'='*80}\n")
    
    for epoch in range(config.get('epochs', 30)):
        print(f"\n{'─'*80}")
        print(f"📅 Epoch [{epoch + 1}/{config.get('epochs', 30)}]")
        print(f"{'─'*80}")
        
        # Unfreeze backbone after certain epochs
        if epoch == unfreeze_epoch and config.get('freeze_backbone', True):
            print(f"\n🔓 Unfreezing last {config.get('unfreeze_blocks', 2)} dense block(s)...")
            model.unfreeze_last_blocks(config.get('unfreeze_blocks', 2))
            
            # Update optimizer with new trainable parameters
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.get('lr', 1e-4) * 0.1,  # Lower LR for fine-tuning
                weight_decay=config.get('weight_decay', 1e-4)
            )
            print(f"   Reduced learning rate to: {config.get('lr', 1e-4) * 0.1:.2e}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(model, val_loader, criterion, device)
        
        precision, recall, _, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='weighted', zero_division=0
        )
        
        # Update scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        history['precision'].append(precision)
        history['recall'].append(recall)
        
        print(f"\n📊 Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   Precision:  {precision:.4f} | Recall: {recall:.4f}")
        print(f"   Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, config.get('model_save_path'))
            print(f"   💾 New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(f"\n⚠️  Early stopping triggered after epoch {epoch + 1}!")
            break
    
    # Final evaluation
    print(f"\n{'='*80}")
    print("🎯 FINAL EVALUATION ON TEST SET")
    print(f"{'='*80}\n")
    
    checkpoint = torch.load(config.get('model_save_path'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels, test_probs = validate(model, test_loader, criterion, device)
    
    print("📈 Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    
    print(f"\n📊 Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Benign', 'Malignant'], digits=4))
    
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\n🔢 Confusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"\n🎯 Additional Metrics:")
    print(f"   Sensitivity (Recall for Malignant): {sensitivity:.4f}")
    print(f"   Specificity (Recall for Benign):    {specificity:.4f}")
    
    test_probs_malignant = [p[1] for p in test_probs]
    roc_auc = plot_roc_curve(test_labels, test_probs_malignant, config.get('roc_plot_path'))
    print(f"   ROC-AUC Score: {roc_auc:.4f}")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Sensitivity', 'Specificity', 'ROC-AUC'],
        'Value': [
            test_acc / 100,
            precision_recall_fscore_support(test_labels, test_preds, average='weighted')[0],
            precision_recall_fscore_support(test_labels, test_preds, average='weighted')[1],
            precision_recall_fscore_support(test_labels, test_preds, average='weighted')[2],
            sensitivity,
            specificity,
            roc_auc
        ]
    })
    metrics_df.to_csv(config.get('metrics_csv_path'), index=False)
    print(f"\n💾 Metrics saved to: {config.get('metrics_csv_path')}")
    
    plot_training_history(history, config.get('history_plot_path'))
    plot_confusion_matrix(cm, ['Benign', 'Malignant'], config.get('cm_plot_path'))
    
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETED!")
    print(f"{'='*80}\n")
    print(f"📁 Best model saved to: {config.get('model_save_path')}")
    print(f"📊 Final test accuracy: {test_acc:.2f}%")
    print(f"🔥 Best validation accuracy: {best_val_acc:.2f}%")


# =====================================================================
# 7. MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    config = {
        # Data paths (DCGAN Configuration)
        'csv_path': 'dataset_labels_fixed.csv',
        'img_dir': 'images',
        
        # Model configuration
        'model_name': 'densenet121',  # densenet121, densenet169, densenet201
        'num_classes': 2,
        'dropout': 0.5,
        
        # Training hyperparameters
        'batch_size': 32,
        'epochs': 20,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        
        # Data split (70/20/10 for larger DCGAN dataset)
        'test_size': 0.10,
        'val_size': 0.20,
        
        # Image settings
        'img_size': 224,
        'augmentation': True,
        
        # Training strategy
        'freeze_backbone': True,
        'unfreeze_epoch': 5,
        'unfreeze_blocks': 2,
        'mixed_precision': True,
        'patience': 10,
        
        # Paths for saving outputs
        'model_save_path': 'DenseNet_DCGAN.pth',
        'history_plot_path': 'training_history_densenet_dcgan.png',
        'cm_plot_path': 'confusion_matrix_densenet_dcgan.png',
        'roc_plot_path': 'roc_curve_densenet_dcgan.png',
        'metrics_csv_path': 'metrics_DenseNet_DCGAN.csv',
    }
    
    train_densenet(config)
