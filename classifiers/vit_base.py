"""
Vision Transformer (ViT) Model Definition for Dermatoscopic Lesion Classification
==================================================================================

Standard Vision Transformer (ViT) with self-attention mechanism.
Excellent for capturing global context in medical images.

Key Features:
- Pure attention-based architecture
- Global receptive field from first layer
- Pretrained on ImageNet
- Strong performance on various tasks

Usage:
    from vit_base import create_vit_model
    model = create_vit_model(num_classes=2, pretrained=True)
"""

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("⚠️  timm not available. Install: pip install timm")

try:
    from transformers import ViTModel, ViTForImageClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not available. Install: pip install transformers")


class ViTClassifier(nn.Module):
    """
    Vision Transformer model for binary/multi-class classification
    Uses pure attention mechanism without convolutions
    """
    def __init__(self, model_name='vit_base_patch16_224', num_classes=2, pretrained=True, dropout=0.1):
        super(ViTClassifier, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Try loading from timm first (more variants available)
        if TIMM_AVAILABLE:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features
        elif TRANSFORMERS_AVAILABLE:
            # Fallback to Hugging Face
            self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            num_features = 768
        else:
            raise ImportError("Either timm or transformers must be installed")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        if TRANSFORMERS_AVAILABLE and hasattr(self.backbone, 'config'):
            outputs = self.backbone(x)
            features = outputs.last_hidden_state[:, 0]  # CLS token
        else:
            features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        return logits


def create_vit_model(model_name='vit_base_patch16_224', num_classes=2, 
                     pretrained=True, dropout=0.1, device='cuda'):
    """
    Create and initialize Vision Transformer model
    
    Args:
        model_name: ViT variant ('vit_tiny', 'vit_small', 'vit_base', 'vit_large')
        num_classes: Number of output classes (2 for binary)
        pretrained: Load ImageNet pretrained weights
        dropout: Dropout rate
        device: Device to load model on
    
    Returns:
        model: ViT model ready for training
    """
    model = ViTClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    model = model.to(device)
    return model


def create_vit_tiny(num_classes=2, pretrained=True, dropout=0.1, device='cuda'):
    """Create ViT-Tiny (smaller, faster)"""
    return create_vit_model('vit_tiny_patch16_224', num_classes, pretrained, dropout, device)


def create_vit_small(num_classes=2, pretrained=True, dropout=0.1, device='cuda'):
    """Create ViT-Small (balanced)"""
    return create_vit_model('vit_small_patch16_224', num_classes, pretrained, dropout, device)


def create_vit_base(num_classes=2, pretrained=True, dropout=0.1, device='cuda'):
    """Create ViT-Base (recommended)"""
    return create_vit_model('vit_base_patch16_224', num_classes, pretrained, dropout, device)


def create_vit_large(num_classes=2, pretrained=True, dropout=0.1, device='cuda'):
    """Create ViT-Large (highest capacity)"""
    return create_vit_model('vit_large_patch16_224', num_classes, pretrained, dropout, device)


def freeze_backbone(model, freeze=True):
    """Freeze/unfreeze backbone for progressive training"""
    for param in model.backbone.parameters():
        param.requires_grad = not freeze


def unfreeze_last_n_blocks(model, n=4):
    """Unfreeze last n transformer blocks for fine-tuning"""
    if hasattr(model.backbone, 'blocks'):
        # timm models
        total_blocks = len(model.backbone.blocks)
        for i in range(total_blocks - n, total_blocks):
            for param in model.backbone.blocks[i].parameters():
                param.requires_grad = True
    elif hasattr(model.backbone, 'encoder'):
        # transformers models
        total_layers = len(model.backbone.encoder.layer)
        for i in range(total_layers - n, total_layers):
            for param in model.backbone.encoder.layer[i].parameters():
                param.requires_grad = True


def get_model_info(model):
    """Print model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{'='*60}")
    print(f"Vision Transformer Model Information:")
    print(f"  Variant: {model.model_name}")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Frozen Parameters: {total_params - trainable_params:,}")
    print(f"  Model Size: {total_params * 4 / 1024**2:.2f} MB (FP32)")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Vision Transformer on device: {device}\n")
    
    # Test ViT-Base
    print("Testing ViT-Base:")
    model = create_vit_base(num_classes=2, pretrained=True, device=device)
    get_model_info(model)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print("✅ Vision Transformer model created successfully!")
