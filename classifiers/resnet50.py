"""
ResNet50 Model Definition for Dermatoscopic Lesion Classification
==================================================================

ResNet50 is a robust CNN architecture with residual connections.
Excellent baseline for medical image classification.

Key Features:
- Residual connections prevent vanishing gradients
- 50 layers deep
- Pretrained on ImageNet
- Proven performance on medical imaging

Usage:
    from resnet50 import create_resnet50_model
    model = create_resnet50_model(num_classes=2, pretrained=True)
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet50Classifier(nn.Module):
    """
    ResNet50 model for binary/multi-class classification
    Pretrained on ImageNet with custom classification head
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super(ResNet50Classifier, self).__init__()
        self.num_classes = num_classes
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get number of features from last layer
        num_features = self.backbone.fc.in_features
        
        # Replace final fully connected layer with custom classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class ResNet18Classifier(nn.Module):
    """
    ResNet18 (lighter version) for faster training
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super(ResNet18Classifier, self).__init__()
        self.num_classes = num_classes
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Get number of features from last layer
        num_features = self.backbone.fc.in_features
        
        # Replace final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def create_resnet50_model(num_classes=2, pretrained=True, dropout=0.3, device='cuda'):
    """
    Create and initialize ResNet50 model
    
    Args:
        num_classes: Number of output classes (2 for binary)
        pretrained: Load ImageNet pretrained weights
        dropout: Dropout rate
        device: Device to load model on
    
    Returns:
        model: ResNet50 model ready for training
    """
    model = ResNet50Classifier(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    model = model.to(device)
    return model


def create_resnet18_model(num_classes=2, pretrained=True, dropout=0.3, device='cuda'):
    """
    Create and initialize ResNet18 model (lighter version)
    
    Args:
        num_classes: Number of output classes
        pretrained: Load ImageNet pretrained weights
        dropout: Dropout rate
        device: Device to load model on
    
    Returns:
        model: ResNet18 model ready for training
    """
    model = ResNet18Classifier(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    model = model.to(device)
    return model


def freeze_backbone(model, freeze=True):
    """Freeze/unfreeze backbone for progressive training"""
    # Freeze all layers except fc
    for name, param in model.backbone.named_parameters():
        if 'fc' not in name:
            param.requires_grad = not freeze


def unfreeze_last_n_blocks(model, n=2):
    """Unfreeze last n residual blocks for fine-tuning"""
    # ResNet50 has layer1, layer2, layer3, layer4
    layers_to_unfreeze = [f'layer{5-i}' for i in range(n)]
    
    for name, param in model.backbone.named_parameters():
        if any(layer in name for layer in layers_to_unfreeze):
            param.requires_grad = True


def get_model_info(model):
    """Print model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_name = model.__class__.__name__
    
    print(f"{'='*60}")
    print(f"{model_name} Model Information:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Frozen Parameters: {total_params - trainable_params:,}")
    print(f"  Model Size: {total_params * 4 / 1024**2:.2f} MB (FP32)")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing ResNet models on device: {device}\n")
    
    # Test ResNet50
    print("Testing ResNet50:")
    model50 = create_resnet50_model(num_classes=2, pretrained=True, device=device)
    get_model_info(model50)
    
    # Test ResNet18
    print("\nTesting ResNet18:")
    model18 = create_resnet18_model(num_classes=2, pretrained=True, device=device)
    get_model_info(model18)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    output50 = model50(dummy_input)
    output18 = model18(dummy_input)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  ResNet50 output shape: {output50.shape}")
    print(f"  ResNet18 output shape: {output18.shape}")
    print("✅ ResNet models created successfully!")
