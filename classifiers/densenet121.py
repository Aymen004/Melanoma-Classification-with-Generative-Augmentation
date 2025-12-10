"""
DenseNet121 Model Definition for Dermatoscopic Lesion Classification
=====================================================================

DenseNet (Densely Connected Convolutional Networks) advantages:
- Dense connectivity for better gradient flow
- Parameter efficiency
- Feature reuse across layers
- Excellent for medical imaging

Usage:
    from densenet121 import create_densenet121_model
    model = create_densenet121_model(num_classes=2, pretrained=True)
"""

import torch
import torch.nn as nn
from torchvision import models


class DenseNet121Classifier(nn.Module):
    """
    DenseNet121 model for binary/multi-class classification
    Pretrained on ImageNet with custom classification head
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super(DenseNet121Classifier, self).__init__()
        self.num_classes = num_classes
        
        # Load pretrained DenseNet121
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Get number of features from last layer
        num_features = self.backbone.classifier.in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def create_densenet121_model(num_classes=2, pretrained=True, dropout=0.3, device='cuda'):
    """
    Create and initialize DenseNet121 model
    
    Args:
        num_classes: Number of output classes (2 for binary)
        pretrained: Load ImageNet pretrained weights
        dropout: Dropout rate
        device: Device to load model on
    
    Returns:
        model: DenseNet121 model ready for training
    """
    model = DenseNet121Classifier(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    model = model.to(device)
    return model


def freeze_backbone(model, freeze=True):
    """Freeze/unfreeze backbone for progressive training"""
    for param in model.backbone.features.parameters():
        param.requires_grad = not freeze


def unfreeze_last_n_blocks(model, n=2):
    """Unfreeze last n dense blocks for fine-tuning"""
    # DenseNet121 has 4 dense blocks
    blocks_to_unfreeze = [f'denseblock{5-i}' for i in range(n)]
    
    for name, param in model.backbone.features.named_parameters():
        if any(block in name for block in blocks_to_unfreeze):
            param.requires_grad = True


def get_model_info(model):
    """Print model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{'='*60}")
    print(f"DenseNet121 Model Information:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Frozen Parameters: {total_params - trainable_params:,}")
    print(f"  Model Size: {total_params * 4 / 1024**2:.2f} MB (FP32)")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing DenseNet121 on device: {device}")
    
    model = create_densenet121_model(num_classes=2, pretrained=True, device=device)
    get_model_info(model)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print("✅ DenseNet121 model created successfully!")
