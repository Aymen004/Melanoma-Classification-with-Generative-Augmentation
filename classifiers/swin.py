"""
Swin Transformer Model Definition for Dermatoscopic Lesion Classification
==========================================================================

Swin Transformer uses shifted windows for efficient hierarchical feature learning.
Excellent for medical imaging with multi-scale features.

Key Features:
- Hierarchical architecture like CNNs
- Shifted window attention for efficiency
- Multi-scale feature extraction
- State-of-the-art performance

Usage:
    from swin import create_swin_model
    model = create_swin_model(num_classes=2, pretrained=True)
"""

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("⚠️  timm not available. Install: pip install timm")


class SwinTransformerClassifier(nn.Module):
    """
    Swin Transformer model for binary/multi-class classification
    Uses hierarchical shifted window attention
    """
    def __init__(self, model_name='swin_base_patch4_window7_224', num_classes=2, pretrained=True, dropout=0.2):
        super(SwinTransformerClassifier, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for Swin Transformer. Install: pip install timm")
        
        # Load pretrained Swin Transformer
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get number of features
        num_features = self.backbone.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def create_swin_model(model_name='swin_base_patch4_window7_224', num_classes=2, 
                      pretrained=True, dropout=0.2, device='cuda'):
    """
    Create and initialize Swin Transformer model
    
    Args:
        model_name: Swin variant ('swin_tiny', 'swin_small', 'swin_base', 'swin_large')
        num_classes: Number of output classes (2 for binary)
        pretrained: Load ImageNet pretrained weights
        dropout: Dropout rate
        device: Device to load model on
    
    Returns:
        model: Swin Transformer model ready for training
    """
    model = SwinTransformerClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    model = model.to(device)
    return model


def create_swin_tiny(num_classes=2, pretrained=True, dropout=0.2, device='cuda'):
    """Create Swin-Tiny (smaller, faster)"""
    return create_swin_model('swin_tiny_patch4_window7_224', num_classes, pretrained, dropout, device)


def create_swin_small(num_classes=2, pretrained=True, dropout=0.2, device='cuda'):
    """Create Swin-Small (balanced)"""
    return create_swin_model('swin_small_patch4_window7_224', num_classes, pretrained, dropout, device)


def create_swin_base(num_classes=2, pretrained=True, dropout=0.2, device='cuda'):
    """Create Swin-Base (recommended)"""
    return create_swin_model('swin_base_patch4_window7_224', num_classes, pretrained, dropout, device)


def freeze_backbone(model, freeze=True):
    """Freeze/unfreeze backbone for progressive training"""
    for param in model.backbone.parameters():
        param.requires_grad = not freeze


def unfreeze_last_n_stages(model, n=2):
    """Unfreeze last n stages for fine-tuning"""
    # Swin has 4 stages
    if hasattr(model.backbone, 'layers'):
        total_stages = len(model.backbone.layers)
        for i in range(total_stages - n, total_stages):
            for param in model.backbone.layers[i].parameters():
                param.requires_grad = True


def get_model_info(model):
    """Print model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{'='*60}")
    print(f"Swin Transformer Model Information:")
    print(f"  Variant: {model.model_name}")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Frozen Parameters: {total_params - trainable_params:,}")
    print(f"  Model Size: {total_params * 4 / 1024**2:.2f} MB (FP32)")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Swin Transformer on device: {device}\n")
    
    # Test different variants
    print("Testing Swin-Base:")
    model = create_swin_base(num_classes=2, pretrained=True, device=device)
    get_model_info(model)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print("✅ Swin Transformer model created successfully!")
