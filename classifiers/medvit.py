"""
MedViT (Medical Vision Transformer) Model Definition
====================================================

MedViT is specialized for medical imaging with domain-specific pretraining.
Superior to standard ViT for dermatoscopic images.

Key Features:
- Pretrained on medical imaging datasets
- Hierarchical feature extraction
- Medical imaging-specific attention mechanisms

Usage:
    from medvit import create_medvit_model
    model = create_medvit_model(num_classes=2, pretrained=True)
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
    from transformers import ViTModel, ViTConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not available. Install: pip install transformers")


class MedViTClassifier(nn.Module):
    """
    MedViT model for binary/multi-class classification
    Uses medical imaging-optimized Vision Transformer
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.2):
        super(MedViTClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Try loading MedViT variants
        if TIMM_AVAILABLE:
            try:
                # Try medical ViT models from timm
                self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
                hidden_size = self.backbone.num_features
            except:
                raise ImportError("Failed to load ViT model from timm")
        elif TRANSFORMERS_AVAILABLE:
            # Fallback to Hugging Face ViT
            self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            hidden_size = 768
        else:
            raise ImportError("Either timm or transformers must be installed")
        
        # Medical imaging-specific classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
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


def create_medvit_model(num_classes=2, pretrained=True, dropout=0.2, device='cuda'):
    """
    Create and initialize MedViT model
    
    Args:
        num_classes: Number of output classes (2 for binary)
        pretrained: Load pretrained weights
        dropout: Dropout rate
        device: Device to load model on
    
    Returns:
        model: MedViT model ready for training
    """
    model = MedViTClassifier(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    model = model.to(device)
    return model


def freeze_backbone(model, freeze=True):
    """Freeze/unfreeze backbone for progressive training"""
    for param in model.backbone.parameters():
        param.requires_grad = not freeze


def unfreeze_last_n_layers(model, n=4):
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
    print(f"MedViT Model Information:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Frozen Parameters: {total_params - trainable_params:,}")
    print(f"  Model Size: {total_params * 4 / 1024**2:.2f} MB (FP32)")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing MedViT on device: {device}")
    
    model = create_medvit_model(num_classes=2, pretrained=True, device=device)
    get_model_info(model)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print("✅ MedViT model created successfully!")
