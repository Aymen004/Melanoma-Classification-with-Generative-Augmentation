"""
DenseNet-121 Classifier for Melanoma Detection
Pre-trained on ImageNet, fine-tuned on skin lesion data.
"""

import torch
import torch.nn as nn
from torchvision import models


class DenseNetClassifier(nn.Module):
    """
    DenseNet-121 binary classifier for benign vs malignant classification.
    
    Args:
        pretrained (bool): Use ImageNet pre-trained weights. Default: True
        num_classes (int): Number of output classes. Default: 2 (benign/malignant)
    """
    
    def __init__(self, pretrained=True, num_classes=2):
        super(DenseNetClassifier, self).__init__()
        
        # Load pre-trained DenseNet-121
        self.model = models.densenet121(pretrained=pretrained)
        
        # Replace classifier head
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input images [B, 3, 224, 224] with ImageNet normalization
        
        Returns:
            torch.Tensor: Logits [B, num_classes]
        """
        return self.model(x)
    
    def predict_proba(self, x):
        """
        Get class probabilities.
        
        Args:
            x (torch.Tensor): Input images [B, 3, 224, 224]
        
        Returns:
            torch.Tensor: Probabilities [B, num_classes]
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def load_densenet_checkpoint(checkpoint_path, device='cuda', num_classes=2):
    """
    Load pre-trained DenseNet classifier from checkpoint.
    
    Args:
        checkpoint_path (str): Path to .pth checkpoint file
        device (str): Device to load model on ('cuda' or 'cpu')
        num_classes (int): Number of output classes
    
    Returns:
        DenseNetClassifier: Loaded model in eval mode
    """
    model = DenseNetClassifier(pretrained=False, num_classes=num_classes)
    
    # Load checkpoint with device mapping
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model
