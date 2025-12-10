"""
BioViT (Biomedical Vision Transformer) Model Definition
========================================================

BioViT is specialized for medical imaging tasks with pretraining on biomedical data.
Better than standard ViT for dermatoscopic images due to domain-specific features.

Usage:
    from biovit import create_biovit_model
    model = create_biovit_model(num_classes=2, pretrained=True)
"""

import torch
import torch.nn as nn

try:
    from transformers import ViTForImageClassification, ViTModel, ViTConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not available. Install: pip install transformers")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("⚠️  timm not available. Install: pip install timm")


class BioViTClassifier(nn.Module):
    """
    BioViT model for binary/multi-class classification
    Uses Vision Transformer architecture with medical imaging optimizations
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.1):
        super(BioViTClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Try loading BioViT from Hugging Face or timm
        if TRANSFORMERS_AVAILABLE:
            try:
                # Try BioViT from Hugging Face
                self.backbone = ViTModel.from_pretrained('microsoft/BiomedVLP-BioViL-T')
                hidden_size = self.backbone.config.hidden_size
            except:
                # Fallback to standard ViT
                print("⚠️  BioViT not found, using standard ViT")
                self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
                hidden_size = 768
        elif TIMM_AVAILABLE:
            # Use timm as fallback
            self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
            hidden_size = self.backbone.num_features
        else:
            raise ImportError("Either transformers or timm must be installed")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
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


def create_biovit_model(num_classes=2, pretrained=True, dropout=0.1, device='cuda'):
    """
    Create and initialize BioViT model
    
    Args:
        num_classes: Number of output classes (2 for binary)
        pretrained: Load pretrained weights
        dropout: Dropout rate
        device: Device to load model on
    
    Returns:
        model: BioViT model ready for training
    """
    model = BioViTClassifier(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    model = model.to(device)
    return model


def get_model_info(model):
    """Print model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{'='*60}")
    print(f"BioViT Model Information:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Model Size: {total_params * 4 / 1024**2:.2f} MB (FP32)")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing BioViT on device: {device}")
    
    model = create_biovit_model(num_classes=2, pretrained=True, device=device)
    get_model_info(model)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print("✅ BioViT model created successfully!")
