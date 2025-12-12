"""
Inférence et Détection d'Anomalies avec le VAE
==============================================

Ce module fournit les outils pour:
1. Calculer l'erreur de reconstruction des images
2. Calibrer le seuil de détection d'anomalies
3. Classer les images comme normales ou anormales

Processus de Calibrage:
    1. Passer les images de validation (mix bénin/malin) dans le VAE
    2. Calculer l'erreur de reconstruction (MSE) pour chaque image
    3. Tracer la distribution des erreurs
    4. Fixer le seuil optimal (ex: 95ème percentile des bénins)

Usage:
    from inference_vae import VAEAnomalyDetector
    
    detector = VAEAnomalyDetector(model_path='vae_output/checkpoints/best_model.pth')
    detector.calibrate(val_loader, labels)
    predictions = detector.predict(test_loader)
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    confusion_matrix, classification_report,
    f1_score, accuracy_score
)
from typing import Tuple, List, Optional, Dict
import logging
import json

from VAE_model import VAE, VAEConfig, ConvVAE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_reconstruction_error(
    model: torch.nn.Module,
    images: torch.Tensor,
    method: str = 'mse'
) -> torch.Tensor:
    """
    Calcule l'erreur de reconstruction pour un batch d'images
    
    Args:
        model: VAE model
        images: Batch d'images [B, C, H, W]
        method: 'mse', 'mae', ou 'ssim'
        
    Returns:
        errors: Erreur par image [B]
    """
    model.eval()
    with torch.no_grad():
        reconstructions = model.reconstruct(images)
        
        if method == 'mse':
            # Mean Squared Error par image
            errors = F.mse_loss(reconstructions, images, reduction='none')
            errors = errors.mean(dim=[1, 2, 3])  # Moyenne sur C, H, W
        elif method == 'mae':
            # Mean Absolute Error
            errors = F.l1_loss(reconstructions, images, reduction='none')
            errors = errors.mean(dim=[1, 2, 3])
        elif method == 'combined':
            # Combinaison MSE + gradient-based
            mse = F.mse_loss(reconstructions, images, reduction='none').mean(dim=[1, 2, 3])
            
            # Sobel-like gradient difference
            def sobel(img):
                gx = img[:, :, :, 1:] - img[:, :, :, :-1]
                gy = img[:, :, 1:, :] - img[:, :, :-1, :]
                return gx, gy
            
            gx_orig, gy_orig = sobel(images)
            gx_recon, gy_recon = sobel(reconstructions)
            
            grad_error = (F.mse_loss(gx_orig, gx_recon, reduction='none').mean(dim=[1, 2, 3]) +
                         F.mse_loss(gy_orig, gy_recon, reduction='none').mean(dim=[1, 2, 3])) / 2
            
            errors = mse + 0.5 * grad_error
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return errors


def calibrate_threshold(
    errors_benign: np.ndarray,
    errors_malignant: np.ndarray = None,
    percentile: float = 95.0,
    method: str = 'percentile'
) -> Tuple[float, Dict]:
    """
    Calibre le seuil de détection d'anomalies
    
    Args:
        errors_benign: Erreurs de reconstruction des images bénignes
        errors_malignant: Erreurs des images malignes (optionnel, pour l'évaluation)
        percentile: Percentile des erreurs bénignes pour le seuil
        method: 'percentile', 'optimal_f1', ou 'youden'
        
    Returns:
        threshold: Seuil optimal
        stats: Statistiques de calibrage
    """
    stats = {
        'benign_mean': float(np.mean(errors_benign)),
        'benign_std': float(np.std(errors_benign)),
        'benign_median': float(np.median(errors_benign)),
        'benign_min': float(np.min(errors_benign)),
        'benign_max': float(np.max(errors_benign)),
    }
    
    if method == 'percentile':
        # Seuil basé sur le percentile des bénins
        threshold = float(np.percentile(errors_benign, percentile))
        stats['method'] = f'percentile_{percentile}'
    
    elif method == 'optimal_f1' and errors_malignant is not None:
        # Trouver le seuil qui maximise le F1-score
        all_errors = np.concatenate([errors_benign, errors_malignant])
        labels = np.concatenate([np.zeros(len(errors_benign)), np.ones(len(errors_malignant))])
        
        # Tester différents seuils
        thresholds = np.linspace(all_errors.min(), all_errors.max(), 100)
        best_f1 = 0
        best_threshold = thresholds[0]
        
        for t in thresholds:
            preds = (all_errors > t).astype(int)
            f1 = f1_score(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        threshold = float(best_threshold)
        stats['method'] = 'optimal_f1'
        stats['best_f1'] = float(best_f1)
    
    elif method == 'youden' and errors_malignant is not None:
        # Critère de Youden (maximise Sensibilité + Spécificité - 1)
        all_errors = np.concatenate([errors_benign, errors_malignant])
        labels = np.concatenate([np.zeros(len(errors_benign)), np.ones(len(errors_malignant))])
        
        fpr, tpr, thresholds = roc_curve(labels, all_errors)
        youden_idx = np.argmax(tpr - fpr)
        threshold = float(thresholds[youden_idx])
        
        stats['method'] = 'youden'
        stats['youden_index'] = float(tpr[youden_idx] - fpr[youden_idx])
    
    else:
        # Fallback: percentile
        threshold = float(np.percentile(errors_benign, percentile))
        stats['method'] = f'percentile_{percentile}'
    
    if errors_malignant is not None:
        stats['malignant_mean'] = float(np.mean(errors_malignant))
        stats['malignant_std'] = float(np.std(errors_malignant))
        stats['malignant_median'] = float(np.median(errors_malignant))
    
    stats['threshold'] = threshold
    
    return threshold, stats


class VAEAnomalyDetector:
    """
    Détecteur d'anomalies basé sur VAE
    
    Utilise l'erreur de reconstruction pour détecter les lésions anormales.
    Les images avec une erreur supérieure au seuil sont classées comme anomalies.
    """
    
    def __init__(
        self,
        model_path: str = None,
        model: torch.nn.Module = None,
        config: VAEConfig = None,
        device: torch.device = None
    ):
        """
        Initialize the detector
        
        Args:
            model_path: Path to the saved model checkpoint
            model: Pre-loaded model (alternative to model_path)
            config: VAE configuration
            device: Device to run on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model is not None:
            self.model = model.to(self.device)
            self.config = config or VAEConfig()
        elif model_path is not None:
            self.model, self.config = self._load_model(model_path)
        else:
            raise ValueError("Either model_path or model must be provided")
        
        self.model.eval()
        
        # Calibration
        self.threshold = None
        self.calibration_stats = None
        self.is_calibrated = False
        
        logger.info(f"VAEAnomalyDetector initialized on {self.device}")
    
    def _load_model(self, model_path: str) -> Tuple[torch.nn.Module, VAEConfig]:
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Recréer la config
        config_dict = checkpoint.get('config', {})
        config = VAEConfig(**{k: v for k, v in config_dict.items() 
                             if k in VAEConfig.__dataclass_fields__})
        
        # Créer et charger le modèle
        model = VAE(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        logger.info(f"Model loaded from {model_path}")
        return model, config
    
    def compute_errors(
        self,
        dataloader: DataLoader,
        method: str = 'mse',
        return_reconstructions: bool = False
    ) -> Tuple[np.ndarray, Optional[List]]:
        """
        Calcule les erreurs de reconstruction pour un dataloader
        
        Args:
            dataloader: DataLoader d'images
            method: Méthode de calcul ('mse', 'mae', 'combined')
            return_reconstructions: Retourner aussi les reconstructions
            
        Returns:
            errors: Array des erreurs
            reconstructions: Liste des reconstructions (optionnel)
        """
        self.model.eval()
        all_errors = []
        all_reconstructions = [] if return_reconstructions else None
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing errors"):
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                images = images.to(self.device)
                errors = calculate_reconstruction_error(self.model, images, method)
                all_errors.extend(errors.cpu().numpy())
                
                if return_reconstructions:
                    recon = self.model.reconstruct(images)
                    all_reconstructions.append(recon.cpu())
        
        errors = np.array(all_errors)
        
        if return_reconstructions:
            reconstructions = torch.cat(all_reconstructions, dim=0)
            return errors, reconstructions
        
        return errors, None
    
    def calibrate(
        self,
        val_dataloader: DataLoader,
        labels: np.ndarray = None,
        percentile: float = 95.0,
        method: str = 'percentile',
        error_method: str = 'mse'
    ) -> float:
        """
        Calibre le seuil de détection
        
        Args:
            val_dataloader: DataLoader de validation
            labels: Labels (0=bénin, 1=malin) pour chaque image
            percentile: Percentile pour la méthode percentile
            method: 'percentile', 'optimal_f1', 'youden'
            error_method: Méthode de calcul d'erreur
            
        Returns:
            threshold: Seuil calibré
        """
        logger.info("Calibrating anomaly detection threshold...")
        
        # Calculer les erreurs
        errors, _ = self.compute_errors(val_dataloader, method=error_method)
        
        if labels is None:
            # Pas de labels, utiliser tous les erreurs comme "bénin"
            errors_benign = errors
            errors_malignant = None
        else:
            labels = np.array(labels)
            errors_benign = errors[labels == 0]
            errors_malignant = errors[labels == 1] if np.sum(labels == 1) > 0 else None
        
        # Calibrer le seuil
        self.threshold, self.calibration_stats = calibrate_threshold(
            errors_benign,
            errors_malignant,
            percentile=percentile,
            method=method
        )
        
        self.is_calibrated = True
        
        logger.info(f"Calibration complete. Threshold: {self.threshold:.6f}")
        logger.info(f"Benign errors - Mean: {self.calibration_stats['benign_mean']:.6f}, "
                   f"Std: {self.calibration_stats['benign_std']:.6f}")
        
        if errors_malignant is not None:
            logger.info(f"Malignant errors - Mean: {self.calibration_stats['malignant_mean']:.6f}, "
                       f"Std: {self.calibration_stats['malignant_std']:.6f}")
        
        return self.threshold
    
    def predict(
        self,
        images: torch.Tensor = None,
        dataloader: DataLoader = None,
        threshold: float = None,
        error_method: str = 'mse'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédit si les images sont des anomalies
        
        Args:
            images: Batch d'images [B, C, H, W]
            dataloader: Alternative - DataLoader d'images
            threshold: Seuil à utiliser (ou utiliser le calibré)
            error_method: Méthode de calcul d'erreur
            
        Returns:
            predictions: 1 = anomalie, 0 = normal
            scores: Erreurs de reconstruction (anomaly scores)
        """
        if threshold is None:
            if not self.is_calibrated:
                raise ValueError("Model not calibrated. Call calibrate() first or provide a threshold.")
            threshold = self.threshold
        
        if images is not None:
            # Single batch
            images = images.to(self.device)
            scores = calculate_reconstruction_error(self.model, images, error_method)
            scores = scores.cpu().numpy()
        elif dataloader is not None:
            # Full dataloader
            scores, _ = self.compute_errors(dataloader, method=error_method)
        else:
            raise ValueError("Either images or dataloader must be provided")
        
        predictions = (scores > threshold).astype(int)
        
        return predictions, scores
    
    def predict_single(
        self,
        image_path: str = None,
        image: torch.Tensor = None,
        threshold: float = None
    ) -> Dict:
        """
        Prédit pour une seule image
        
        Args:
            image_path: Chemin vers l'image
            image: Tensor de l'image (alternative)
            threshold: Seuil à utiliser
            
        Returns:
            result: Dict avec la prédiction et les détails
        """
        if threshold is None:
            threshold = self.threshold
        
        # Charger et préprocesser l'image
        transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor()
        ])
        
        if image_path is not None:
            img = Image.open(image_path).convert('RGB')
            image = transform(img).unsqueeze(0)
        
        image = image.to(self.device)
        
        # Calculer l'erreur et reconstruire
        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model.reconstruct(image)
            error = F.mse_loss(reconstruction, image, reduction='none')
            error = error.mean().item()
        
        is_anomaly = error > threshold
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': error,
            'threshold': threshold,
            'confidence': abs(error - threshold) / threshold,
            'reconstruction': reconstruction.cpu()
        }
    
    def evaluate(
        self,
        dataloader: DataLoader,
        labels: np.ndarray,
        threshold: float = None
    ) -> Dict:
        """
        Évalue les performances du détecteur
        
        Args:
            dataloader: DataLoader de test
            labels: Vrais labels (0=bénin, 1=malin)
            threshold: Seuil à utiliser
            
        Returns:
            metrics: Dict avec toutes les métriques
        """
        predictions, scores = self.predict(dataloader=dataloader, threshold=threshold)
        labels = np.array(labels)
        
        # Métriques de base
        accuracy = accuracy_score(labels, predictions)
        
        # Matrice de confusion
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        # Sensibilité (Recall pour les malins)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Spécificité
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # F1 Score
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # PR AUC
        precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall_curve, precision_curve)
        
        metrics = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,  # Recall
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'threshold': threshold or self.threshold
        }
        
        return metrics
    
    def plot_error_distribution(
        self,
        errors_benign: np.ndarray,
        errors_malignant: np.ndarray = None,
        threshold: float = None,
        save_path: str = None
    ):
        """
        Visualise la distribution des erreurs
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Distribution des bénins
        ax.hist(errors_benign, bins=50, alpha=0.7, label='Bénin', color='green', density=True)
        
        # Distribution des malins
        if errors_malignant is not None:
            ax.hist(errors_malignant, bins=50, alpha=0.7, label='Malin', color='red', density=True)
        
        # Seuil
        threshold = threshold or self.threshold
        if threshold is not None:
            ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2, 
                      label=f'Seuil: {threshold:.4f}')
        
        ax.set_xlabel('Erreur de Reconstruction (MSE)')
        ax.set_ylabel('Densité')
        ax.set_title('Distribution des Erreurs de Reconstruction - VAE Anomaly Detection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Distribution plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_roc_curve(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        save_path: str = None
    ):
        """
        Trace la courbe ROC
        """
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taux de Faux Positifs')
        ax.set_ylabel('Taux de Vrais Positifs')
        ax.set_title('Courbe ROC - VAE Anomaly Detection')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        plt.show()
        return fig, roc_auc
    
    def visualize_reconstructions(
        self,
        images: torch.Tensor,
        num_samples: int = 8,
        save_path: str = None
    ):
        """
        Visualise les reconstructions vs originaux
        """
        self.model.eval()
        images = images[:num_samples].to(self.device)
        
        with torch.no_grad():
            reconstructions = self.model.reconstruct(images)
            errors = calculate_reconstruction_error(self.model, images)
        
        fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
        
        for i in range(num_samples):
            # Original
            axes[0, i].imshow(images[i].cpu().permute(1, 2, 0).numpy())
            axes[0, i].set_title(f'Original')
            axes[0, i].axis('off')
            
            # Reconstruction
            axes[1, i].imshow(reconstructions[i].cpu().permute(1, 2, 0).numpy())
            axes[1, i].set_title(f'Recon (MSE: {errors[i]:.4f})')
            axes[1, i].axis('off')
        
        plt.suptitle('Originaux vs Reconstructions')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        plt.show()
        return fig
    
    def generate_anomaly_heatmaps(
        self,
        images: torch.Tensor,
        num_samples: int = 8,
        save_path: str = None,
        colormap: str = 'hot'
    ):
        """
        Génère des heatmaps d'anomalie pixel-par-pixel
        
        Synergie 2: XAI - Visualise exactement les zones que le VAE 
        n'arrive pas à reconstruire (zones pathologiques)
        
        Args:
            images: Images à analyser [B, C, H, W]
            num_samples: Nombre d'images à visualiser
            save_path: Chemin pour sauvegarder la figure
            colormap: Colormap matplotlib ('hot', 'jet', 'viridis')
            
        Returns:
            fig: Figure matplotlib
            heatmaps: Tensor des heatmaps [B, H, W]
        """
        self.model.eval()
        images = images[:num_samples].to(self.device)
        
        with torch.no_grad():
            reconstructions = self.model.reconstruct(images)
            
            # Calculer la différence pixel-par-pixel
            # |Image Originale - Image Reconstruite|
            diff = torch.abs(images - reconstructions)
            
            # Moyenner sur les canaux RGB pour obtenir une heatmap 2D
            heatmaps = diff.mean(dim=1)  # [B, H, W]
            
            # Normaliser pour la visualisation
            heatmaps_vis = heatmaps.cpu().numpy()
        
        # Visualisation
        fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))
        
        for i in range(num_samples):
            # Original
            img_np = images[i].cpu().permute(1, 2, 0).numpy()
            axes[0, i].imshow(img_np)
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstruction
            recon_np = reconstructions[i].cpu().permute(1, 2, 0).numpy()
            axes[1, i].imshow(recon_np)
            axes[1, i].set_title('Reconstruction')
            axes[1, i].axis('off')
            
            # Heatmap d'anomalie
            im = axes[2, i].imshow(heatmaps_vis[i], cmap=colormap)
            axes[2, i].set_title(f'Anomaly Heatmap')
            axes[2, i].axis('off')
            plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
        
        plt.suptitle('Heatmaps d\'Anomalie - Zones que le VAE ne peut pas reconstruire')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Anomaly heatmaps saved to {save_path}")
        
        plt.show()
        return fig, heatmaps
    
    def generate_overlay_heatmap(
        self,
        image: torch.Tensor,
        alpha: float = 0.5,
        colormap: str = 'jet',
        threshold_percentile: float = 70
    ) -> Tuple[np.ndarray, float]:
        """
        Génère une heatmap d'anomalie superposée sur l'image originale
        Très intuitif pour les médecins - montre les zones suspectes
        
        Args:
            image: Image unique [C, H, W] ou [1, C, H, W]
            alpha: Transparence de la heatmap (0=invisible, 1=opaque)
            colormap: Colormap pour la heatmap
            threshold_percentile: Percentile pour filtrer le bruit de fond
            
        Returns:
            overlay: Image avec heatmap superposée [H, W, 3]
            anomaly_score: Score d'anomalie global
        """
        self.model.eval()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            reconstruction = self.model.reconstruct(image)
            
            # Différence absolue
            diff = torch.abs(image - reconstruction)
            heatmap = diff.mean(dim=1).squeeze().cpu().numpy()  # [H, W]
            
            # Score global
            anomaly_score = float(heatmap.mean())
            
            # Seuiller pour ne montrer que les zones les plus anormales
            threshold = np.percentile(heatmap, threshold_percentile)
            heatmap_thresholded = np.where(heatmap > threshold, heatmap, 0)
            
            # Normaliser entre 0 et 1
            if heatmap_thresholded.max() > 0:
                heatmap_thresholded = heatmap_thresholded / heatmap_thresholded.max()
        
        # Convertir l'image en numpy
        img_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
        
        # Créer la heatmap colorée
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_thresholded)[:, :, :3]  # RGB seulement
        
        # Superposer
        overlay = (1 - alpha) * img_np + alpha * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        
        return overlay, anomaly_score
    
    def save_calibration(self, path: str):
        """Sauvegarde les paramètres de calibration"""
        calibration_data = {
            'threshold': self.threshold,
            'stats': self.calibration_stats,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        }
        
        with open(path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        logger.info(f"Calibration saved to {path}")
    
    def load_calibration(self, path: str):
        """Charge les paramètres de calibration"""
        with open(path, 'r') as f:
            calibration_data = json.load(f)
        
        self.threshold = calibration_data['threshold']
        self.calibration_stats = calibration_data['stats']
        self.is_calibrated = True
        
        logger.info(f"Calibration loaded from {path}. Threshold: {self.threshold}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='VAE Anomaly Detection Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained VAE model')
    parser.add_argument('--data_csv', type=str, required=True,
                       help='CSV file with images and labels')
    parser.add_argument('--img_dir', type=str, required=True,
                       help='Directory with images')
    parser.add_argument('--output_dir', type=str, default='./vae_results',
                       help='Output directory')
    parser.add_argument('--percentile', type=float, default=95.0,
                       help='Percentile for threshold calibration')
    
    args = parser.parse_args()
    
    # Load detector
    detector = VAEAnomalyDetector(model_path=args.model_path)
    
    print("VAE Anomaly Detector loaded successfully!")
    print(f"Calibration needed before predictions")
