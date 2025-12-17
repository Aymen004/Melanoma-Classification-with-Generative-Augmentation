"""
Hybrid Zero-Miss Melanoma Detection System
==========================================

This module implements the hybrid architecture combining:
1. Supervised classification (DenseNet-121)
2. Unsupervised anomaly detection (ConvVAE)

The system reduces false negatives by using the VAE as a "safety net":
- If DenseNet predicts BENIGN but VAE reconstruction error > threshold
  -> Override to MALIGNANT (suspicious)
  
This dramatically reduces missed cancers while minimizing false alarms.

Architecture Flow:
    Input Image (224x224 for DenseNet, 128x128 for VAE)
        ↓
    ┌─────────────────────┬─────────────────────┐
    │   DenseNet-121      │      ConvVAE        │
    │  (ImageNet norm)    │   ([0,1] norm)      │
    │  Classification     │  Anomaly Score      │
    └──────────┬──────────┴──────────┬──────────┘
               │                     │
               ↓                     ↓
         Probability          Reconstruction
          [Benign,            Error (L1)
           Malignant]              ↓
               │              > threshold?
               └──────────┬──────────┘
                          │
                    Hybrid Decision
                          ↓
                   Final Prediction

Usage:
    from src.inference.hybrid_system import HybridSystem
    
    system = HybridSystem(
        densenet_path='checkpoints/densenet_ddpm.pth',
        vae_path='vae_fix_v2_L1/checkpoints/best_model.pth',
        vae_threshold=0.11
    )
    
    # Single image
    prediction, confidence, details = system.predict_image('path/to/image.jpg')
    
    # Batch prediction
    results = system.predict_batch(image_paths)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Union, Optional
import logging
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vae import ConvVAE
from torchvision.models import densenet121

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenseNetClassifier(nn.Module):
    """DenseNet-121 wrapper for binary classification"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(DenseNetClassifier, self).__init__()
        self.backbone = densenet121(pretrained=pretrained)
        num_features = self.backbone.classifier.in_features
        
        # Custom classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class HybridSystem:
    """
    Hybrid Melanoma Detection System
    
    Combines supervised DenseNet with unsupervised VAE for zero-miss detection.
    The VAE acts as a safety net to catch malignant cases that DenseNet might miss.
    """
    
    def __init__(
        self,
        densenet_path: str,
        vae_path: str,
        vae_threshold: float = 0.11,
        device: Optional[str] = None
    ):
        """
        Initialize hybrid system
        
        Args:
            densenet_path: Path to DenseNet checkpoint (.pth)
            vae_path: Path to VAE checkpoint (.pth)
            vae_threshold: Reconstruction error threshold (calibrated)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.device = self._setup_device(device)
        self.vae_threshold = vae_threshold
        
        # Load models
        logger.info("Loading models...")
        self.densenet = self._load_densenet(densenet_path)
        self.vae = self._load_vae(vae_path)
        
        # Setup preprocessing transforms
        self._setup_transforms()
        
        # Set to evaluation mode
        self.densenet.eval()
        self.vae.eval()
        
        logger.info(f"✅ Hybrid System initialized on {self.device}")
        logger.info(f"   VAE threshold: {vae_threshold:.4f}")
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup computing device"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device_obj = torch.device(device)
        logger.info(f"Using device: {device_obj}")
        return device_obj
    
    def _load_densenet(self, checkpoint_path: str) -> nn.Module:
        """Load DenseNet-121 classifier"""
        try:
            # Create model
            model = DenseNetClassifier(num_classes=2, pretrained=False)
            
            # Load weights
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            logger.info(f"✅ DenseNet loaded from {checkpoint_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load DenseNet: {e}")
            raise
    
    def _load_vae(self, checkpoint_path: str) -> nn.Module:
        """Load ConvVAE model"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract configuration
            if 'config' in checkpoint:
                config_dict = checkpoint['config']
                latent_dim = config_dict.get('latent_dim', 512)
                beta = config_dict.get('beta', 0.0001)
            else:
                # Default values
                latent_dim = 512
                beta = 0.0001
            
            # Create model
            model = ConvVAE(latent_dim=latent_dim, beta=beta)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            
            logger.info(f"✅ VAE loaded from {checkpoint_path}")
            logger.info(f"   Latent dim: {latent_dim}, Beta: {beta}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load VAE: {e}")
            raise
    
    def _setup_transforms(self):
        """Setup preprocessing transforms for both models"""
        
        # DenseNet preprocessing: 224x224, ImageNet normalization
        self.densenet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # VAE preprocessing: 128x128, [0, 1] normalization
        self.vae_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()  # Converts to [0, 1]
        ])
        
        logger.info("Preprocessing transforms configured")
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and validate image"""
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def _densenet_predict(self, image: Image.Image) -> Tuple[int, float, np.ndarray]:
        """
        Get DenseNet prediction
        
        Returns:
            predicted_class: 0 (benign) or 1 (malignant)
            confidence: Probability of predicted class
            probabilities: [P(benign), P(malignant)]
        """
        # Preprocess
        img_tensor = self.densenet_transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.densenet(img_tensor)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
        
        return predicted_class, confidence, probabilities
    
    def _vae_anomaly_score(self, image: Image.Image) -> float:
        """
        Compute VAE reconstruction error (anomaly score)
        
        Returns:
            error: L1 reconstruction error (higher = more anomalous)
        """
        # Preprocess
        img_tensor = self.vae_transform(image).unsqueeze(0).to(self.device)
        
        # Compute reconstruction error
        with torch.no_grad():
            recon, mu, logvar = self.vae(img_tensor)
            error = F.l1_loss(recon, img_tensor, reduction='mean').item()
        
        return error
    
    def predict_image(
        self,
        image_path: str
    ) -> Tuple[str, float, Dict]:
        """
        Predict single image with hybrid system
        
        Args:
            image_path: Path to image file
            
        Returns:
            prediction: 'BENIGN' or 'MALIGNANT'
            confidence: Confidence score [0, 1]
            details: Dictionary with detailed information
        """
        # Load image
        image = self._load_image(image_path)
        
        # Get DenseNet prediction
        dn_class, dn_conf, dn_probs = self._densenet_predict(image)
        dn_label = 'BENIGN' if dn_class == 0 else 'MALIGNANT'
        
        # Get VAE anomaly score
        vae_score = self._vae_anomaly_score(image)
        vae_anomaly = vae_score > self.vae_threshold
        
        # Hybrid decision logic
        if dn_label == 'MALIGNANT':
            # DenseNet already detected malignancy
            final_prediction = 'MALIGNANT'
            final_confidence = dn_conf
            rescue_triggered = False
        elif vae_anomaly:
            # Safety net triggered: VAE detected anomaly
            final_prediction = 'MALIGNANT'
            final_confidence = 0.5 + (vae_score - self.vae_threshold) * 2  # Scaled confidence
            final_confidence = min(final_confidence, 0.95)  # Cap at 0.95
            rescue_triggered = True
        else:
            # Both agree it's benign
            final_prediction = 'BENIGN'
            final_confidence = dn_conf
            rescue_triggered = False
        
        # Detailed information
        details = {
            'densenet_prediction': dn_label,
            'densenet_confidence': float(dn_conf),
            'densenet_probs': {
                'benign': float(dn_probs[0]),
                'malignant': float(dn_probs[1])
            },
            'vae_score': float(vae_score),
            'vae_threshold': float(self.vae_threshold),
            'vae_anomaly_detected': bool(vae_anomaly),
            'rescue_triggered': bool(rescue_triggered),
            'final_prediction': final_prediction,
            'final_confidence': float(final_confidence)
        }
        
        return final_prediction, final_confidence, details
    
    def predict_batch(
        self,
        image_paths: List[str],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Predict multiple images
        
        Args:
            image_paths: List of image file paths
            show_progress: Show progress bar
            
        Returns:
            results: List of prediction dictionaries
        """
        results = []
        
        iterator = image_paths
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(image_paths, desc="Processing images")
        
        for img_path in iterator:
            try:
                prediction, confidence, details = self.predict_image(img_path)
                details['image_path'] = img_path
                results.append(details)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })
        
        return results
    
    def evaluate(
        self,
        image_paths: List[str],
        true_labels: List[int],
        return_details: bool = False
    ) -> Dict:
        """
        Evaluate hybrid system on test set
        
        Args:
            image_paths: List of image paths
            true_labels: List of ground truth labels (0=benign, 1=malignant)
            return_details: Return detailed predictions
            
        Returns:
            metrics: Dictionary with performance metrics
        """
        from sklearn.metrics import classification_report, confusion_matrix
        
        logger.info(f"Evaluating on {len(image_paths)} images...")
        
        predictions = []
        confidences = []
        all_details = []
        
        for img_path, true_label in zip(image_paths, true_labels):
            pred, conf, details = self.predict_image(img_path)
            pred_label = 1 if pred == 'MALIGNANT' else 0
            
            predictions.append(pred_label)
            confidences.append(conf)
            
            if return_details:
                details['true_label'] = true_label
                all_details.append(details)
        
        # Calculate metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'f1_score': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
        }
        
        logger.info(f"✅ Evaluation complete")
        logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   Recall: {metrics['recall']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   F1-Score: {metrics['f1_score']:.4f}")
        
        if return_details:
            return metrics, all_details
        else:
            return metrics


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Hybrid Zero-Miss Melanoma Detection System")
    print("="*60)
    
    # Initialize system (adjust paths as needed)
    system = HybridSystem(
        densenet_path='checkpoints/densenet_best.pth',
        vae_path='vae_fix_v2_L1/checkpoints/best_model.pth',
        vae_threshold=0.11
    )
    
    # Test on single image
    test_image = 'data/test_data/dataset_binary/benign/test_image.jpg'
    
    if Path(test_image).exists():
        prediction, confidence, details = system.predict_image(test_image)
        
        print(f"\nPrediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        print(f"\nDetailed breakdown:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    else:
        print(f"\nTest image not found: {test_image}")
        print("Please provide a valid image path for testing.")
