"""
Classificateur Hybride: VAE + Supervisé (DenseNet)
==================================================

Ce module combine le VAE (détection d'anomalies) avec le classificateur
supervisé (DenseNet) pour une classification plus robuste.

Architecture Hybride:
    1. VAE: Calcule l'anomaly score (erreur de reconstruction)
    2. DenseNet: Prédit la classe (Bénin/Malin) avec probabilité
    3. Fusion: Combine les deux scores pour la décision finale

Stratégies de Fusion:
    - 'voting': Vote majoritaire
    - 'weighted': Moyenne pondérée des probabilités
    - 'cascade': VAE en premier filtre, DenseNet en second
    - 'ensemble': Combinaison sophistiquée avec calibration

Avantages:
    - Meilleure détection des cas Out-of-Distribution (OOD)
    - Réduction des faux négatifs critiques
    - Robustesse face aux données jamais vues

Usage:
    from hybrid_classifier import HybridClassifier
    
    hybrid = HybridClassifier(
        vae_model_path='vae_output/checkpoints/best_model.pth',
        classifier_model_path='models/densenet_best.pth'
    )
    hybrid.calibrate(val_loader, val_labels)
    predictions, confidence = hybrid.predict(test_loader)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
from tqdm import tqdm
import logging
import json
import matplotlib.pyplot as plt

# Imports locaux
from VAE_model import VAE, VAEConfig
from inference_vae import VAEAnomalyDetector, calculate_reconstruction_error

# Ajouter le chemin vers les classifiers
sys.path.insert(0, str(Path(__file__).parent.parent / 'classifiers'))
try:
    from densenet121 import create_densenet121_model, DenseNet121Classifier
except ImportError:
    DenseNet121Classifier = None
    create_densenet121_model = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridClassifier:
    """
    Classificateur hybride combinant VAE et classificateur supervisé
    
    Le VAE sert de filet de sécurité pour détecter les anomalies
    (images très différentes des données d'entraînement).
    Le classificateur supervisé fournit la classification principale.
    """
    
    def __init__(
        self,
        vae_model_path: str = None,
        classifier_model_path: str = None,
        vae_model: nn.Module = None,
        classifier_model: nn.Module = None,
        device: torch.device = None,
        fusion_strategy: str = 'weighted'
    ):
        """
        Initialize le classificateur hybride
        
        Args:
            vae_model_path: Chemin vers le modèle VAE sauvegardé
            classifier_model_path: Chemin vers le classificateur
            vae_model: Modèle VAE pré-chargé (alternative)
            classifier_model: Classificateur pré-chargé (alternative)
            device: Device (GPU/CPU)
            fusion_strategy: 'voting', 'weighted', 'cascade', 'ensemble'
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fusion_strategy = fusion_strategy
        
        # Charger le VAE
        if vae_model is not None:
            self.vae = vae_model.to(self.device)
            self.vae_config = VAEConfig()
        elif vae_model_path is not None:
            self._load_vae(vae_model_path)
        else:
            self.vae = None
            logger.warning("No VAE model provided")
        
        # Charger le classificateur
        if classifier_model is not None:
            self.classifier = classifier_model.to(self.device)
        elif classifier_model_path is not None:
            self._load_classifier(classifier_model_path)
        else:
            self.classifier = None
            logger.warning("No classifier model provided")
        
        # Paramètres de fusion
        self.vae_threshold = None
        self.vae_weight = 0.3  # Poids du VAE dans la fusion
        self.classifier_weight = 0.7  # Poids du classificateur
        
        # Calibration
        self.is_calibrated = False
        self.calibration_params = {}
        
        logger.info(f"HybridClassifier initialized on {self.device}")
        logger.info(f"Fusion strategy: {fusion_strategy}")
    
    def _load_vae(self, model_path: str):
        """Charge le modèle VAE"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        config_dict = checkpoint.get('config', {})
        self.vae_config = VAEConfig(**{k: v for k, v in config_dict.items() 
                                       if k in VAEConfig.__dataclass_fields__})
        
        self.vae = VAE(self.vae_config)
        self.vae.load_state_dict(checkpoint['model_state_dict'])
        self.vae = self.vae.to(self.device)
        self.vae.eval()
        
        logger.info(f"VAE loaded from {model_path}")
    
    def _load_classifier(self, model_path: str):
        """Charge le classificateur supervisé"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Créer le modèle DenseNet
        if create_densenet121_model is not None:
            self.classifier = create_densenet121_model(
                num_classes=2,
                pretrained=False,
                device=self.device
            )
        else:
            # Fallback - créer un DenseNet basique
            from torchvision import models
            self.classifier = models.densenet121(pretrained=False)
            self.classifier.classifier = nn.Linear(
                self.classifier.classifier.in_features, 2
            )
            self.classifier = self.classifier.to(self.device)
        
        # Charger les poids
        if 'model_state_dict' in checkpoint:
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.classifier.load_state_dict(checkpoint)
        
        self.classifier.eval()
        logger.info(f"Classifier loaded from {model_path}")
    
    def calibrate(
        self,
        val_dataloader: DataLoader,
        val_labels: np.ndarray,
        vae_percentile: float = 95.0
    ):
        """
        Calibre les seuils et poids pour la fusion
        
        Args:
            val_dataloader: DataLoader de validation
            val_labels: Labels (0=bénin, 1=malin)
            vae_percentile: Percentile pour le seuil VAE
        """
        logger.info("Calibrating hybrid classifier...")
        
        self.vae.eval()
        self.classifier.eval()
        
        all_vae_scores = []
        all_classifier_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Calibrating")):
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                images = images.to(self.device)
                batch_size = images.size(0)
                
                # VAE anomaly scores
                vae_errors = calculate_reconstruction_error(self.vae, images)
                all_vae_scores.extend(vae_errors.cpu().numpy())
                
                # Classifier probabilities
                logits = self.classifier(images)
                probs = F.softmax(logits, dim=1)[:, 1]  # P(malin)
                all_classifier_probs.extend(probs.cpu().numpy())
                
                # Labels pour ce batch
                start_idx = batch_idx * val_dataloader.batch_size
                end_idx = min(start_idx + batch_size, len(val_labels))
                all_labels.extend(val_labels[start_idx:end_idx])
        
        all_vae_scores = np.array(all_vae_scores)
        all_classifier_probs = np.array(all_classifier_probs)
        all_labels = np.array(all_labels)
        
        # Calibrer le seuil VAE sur les bénins
        benign_scores = all_vae_scores[all_labels == 0]
        self.vae_threshold = np.percentile(benign_scores, vae_percentile)
        
        # Normaliser les scores VAE
        self.vae_score_mean = np.mean(all_vae_scores)
        self.vae_score_std = np.std(all_vae_scores)
        
        # Optimiser les poids de fusion
        if self.fusion_strategy in ['weighted', 'ensemble']:
            self._optimize_weights(all_vae_scores, all_classifier_probs, all_labels)
        
        self.is_calibrated = True
        
        # Sauvegarder les paramètres
        self.calibration_params = {
            'vae_threshold': float(self.vae_threshold),
            'vae_score_mean': float(self.vae_score_mean),
            'vae_score_std': float(self.vae_score_std),
            'vae_weight': float(self.vae_weight),
            'classifier_weight': float(self.classifier_weight),
            'vae_percentile': vae_percentile,
            'num_samples': len(all_labels)
        }
        
        logger.info(f"Calibration complete:")
        logger.info(f"  VAE threshold: {self.vae_threshold:.6f}")
        logger.info(f"  VAE weight: {self.vae_weight:.3f}")
        logger.info(f"  Classifier weight: {self.classifier_weight:.3f}")
    
    def _optimize_weights(
        self,
        vae_scores: np.ndarray,
        classifier_probs: np.ndarray,
        labels: np.ndarray
    ):
        """Optimise les poids de fusion par grid search"""
        from sklearn.metrics import f1_score
        
        # Normaliser les scores VAE
        vae_normalized = (vae_scores - self.vae_score_mean) / (self.vae_score_std + 1e-8)
        vae_probs = 1 / (1 + np.exp(-vae_normalized))  # Sigmoid
        
        best_f1 = 0
        best_weights = (0.3, 0.7)
        
        # Grid search sur les poids
        for vae_w in np.arange(0.1, 0.6, 0.05):
            clf_w = 1.0 - vae_w
            
            # Combiner les probabilités
            combined = vae_w * vae_probs + clf_w * classifier_probs
            predictions = (combined > 0.5).astype(int)
            
            f1 = f1_score(labels, predictions)
            
            if f1 > best_f1:
                best_f1 = f1
                best_weights = (vae_w, clf_w)
        
        self.vae_weight, self.classifier_weight = best_weights
        logger.info(f"Optimized weights: VAE={self.vae_weight:.3f}, Classifier={self.classifier_weight:.3f}")
    
    def predict(
        self,
        images: torch.Tensor = None,
        dataloader: DataLoader = None,
        return_details: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Prédit la classe des images
        
        Args:
            images: Batch d'images [B, C, H, W]
            dataloader: DataLoader (alternative)
            return_details: Retourner les détails de la fusion
            
        Returns:
            predictions: Prédictions (0=bénin, 1=malin)
            details: Dict avec les scores détaillés (optionnel)
        """
        if not self.is_calibrated:
            logger.warning("Model not calibrated. Using default parameters.")
            if self.vae_threshold is None:
                self.vae_threshold = 0.1  # Default
        
        self.vae.eval()
        self.classifier.eval()
        
        if images is not None:
            # Single batch
            return self._predict_batch(images, return_details)
        elif dataloader is not None:
            # Full dataloader
            return self._predict_dataloader(dataloader, return_details)
        else:
            raise ValueError("Either images or dataloader must be provided")
    
    def _predict_batch(
        self,
        images: torch.Tensor,
        return_details: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Prédit pour un batch d'images"""
        images = images.to(self.device)
        
        with torch.no_grad():
            # VAE scores
            vae_errors = calculate_reconstruction_error(self.vae, images)
            vae_scores = vae_errors.cpu().numpy()
            
            # Classifier predictions
            logits = self.classifier(images)
            probs = F.softmax(logits, dim=1)
            classifier_probs = probs[:, 1].cpu().numpy()  # P(malin)
        
        # Fusion selon la stratégie
        predictions, fusion_scores = self._fuse_predictions(
            vae_scores, classifier_probs
        )
        
        if return_details:
            details = {
                'vae_scores': vae_scores,
                'classifier_probs': classifier_probs,
                'fusion_scores': fusion_scores,
                'vae_anomaly': (vae_scores > self.vae_threshold).astype(int)
            }
            return predictions, details
        
        return predictions
    
    def _predict_dataloader(
        self,
        dataloader: DataLoader,
        return_details: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Prédit pour un dataloader complet"""
        all_vae_scores = []
        all_classifier_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                images = images.to(self.device)
                
                # VAE
                vae_errors = calculate_reconstruction_error(self.vae, images)
                all_vae_scores.extend(vae_errors.cpu().numpy())
                
                # Classifier
                logits = self.classifier(images)
                probs = F.softmax(logits, dim=1)[:, 1]
                all_classifier_probs.extend(probs.cpu().numpy())
        
        vae_scores = np.array(all_vae_scores)
        classifier_probs = np.array(all_classifier_probs)
        
        predictions, fusion_scores = self._fuse_predictions(vae_scores, classifier_probs)
        
        if return_details:
            details = {
                'vae_scores': vae_scores,
                'classifier_probs': classifier_probs,
                'fusion_scores': fusion_scores,
                'vae_anomaly': (vae_scores > self.vae_threshold).astype(int)
            }
            return predictions, details
        
        return predictions
    
    def _fuse_predictions(
        self,
        vae_scores: np.ndarray,
        classifier_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fusionne les prédictions du VAE et du classificateur
        
        Returns:
            predictions: Prédictions finales
            fusion_scores: Scores de fusion (probabilité de malin)
        """
        if self.fusion_strategy == 'voting':
            # Vote majoritaire simple
            vae_pred = (vae_scores > self.vae_threshold).astype(int)
            clf_pred = (classifier_probs > 0.5).astype(int)
            
            # Si l'un des deux dit "malin", on dit "malin" (conservative)
            predictions = np.maximum(vae_pred, clf_pred)
            fusion_scores = np.maximum(
                vae_scores / (self.vae_threshold * 2),  # Normalize
                classifier_probs
            )
        
        elif self.fusion_strategy == 'weighted':
            # Moyenne pondérée des probabilités
            vae_normalized = (vae_scores - self.vae_score_mean) / (self.vae_score_std + 1e-8)
            vae_probs = 1 / (1 + np.exp(-vae_normalized))  # Sigmoid
            
            fusion_scores = self.vae_weight * vae_probs + self.classifier_weight * classifier_probs
            predictions = (fusion_scores > 0.5).astype(int)
        
        elif self.fusion_strategy == 'cascade':
            # VAE comme premier filtre
            # Si le VAE dit "anomalie", on prédit "malin"
            # Sinon, on utilise le classificateur
            vae_anomaly = vae_scores > self.vae_threshold
            clf_pred = (classifier_probs > 0.5).astype(int)
            
            predictions = np.where(vae_anomaly, 1, clf_pred)
            fusion_scores = np.where(
                vae_anomaly,
                np.maximum(0.5, classifier_probs),  # Au moins 0.5 si anomalie VAE
                classifier_probs
            )
        
        elif self.fusion_strategy == 'ensemble':
            # Combinaison sophistiquée
            # - Fort signal VAE (très anormal) → boost la prob malin
            # - Accord VAE+Clf → haute confiance
            
            vae_normalized = (vae_scores - self.vae_score_mean) / (self.vae_score_std + 1e-8)
            vae_probs = 1 / (1 + np.exp(-vae_normalized))
            
            # Boost si VAE détecte une forte anomalie
            anomaly_boost = np.clip(vae_normalized, 0, 2) / 2  # 0 à 1
            boosted_probs = classifier_probs + anomaly_boost * (1 - classifier_probs) * 0.3
            
            fusion_scores = np.clip(boosted_probs, 0, 1)
            predictions = (fusion_scores > 0.5).astype(int)
        
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        return predictions, fusion_scores
    
    def evaluate(
        self,
        dataloader: DataLoader,
        labels: np.ndarray
    ) -> Dict:
        """
        Évalue le classificateur hybride
        
        Returns:
            metrics: Dict avec les métriques
        """
        from sklearn.metrics import (
            accuracy_score, confusion_matrix, classification_report,
            roc_auc_score, f1_score, precision_score, recall_score
        )
        
        predictions, details = self.predict(dataloader=dataloader, return_details=True)
        labels = np.array(labels)
        
        # Métriques principales
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),  # Sensibilité
            'f1_score': f1_score(labels, predictions),
            'roc_auc': roc_auc_score(labels, details['fusion_scores']),
        }
        
        # Matrice de confusion
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Métriques par composant
        clf_pred = (details['classifier_probs'] > 0.5).astype(int)
        vae_pred = details['vae_anomaly']
        
        metrics['classifier_accuracy'] = accuracy_score(labels, clf_pred)
        metrics['vae_accuracy'] = accuracy_score(labels, vae_pred)
        
        # Amélioration apportée par le VAE
        metrics['improvement_over_classifier'] = metrics['accuracy'] - metrics['classifier_accuracy']
        
        return metrics
    
    def analyze_disagreements(
        self,
        dataloader: DataLoader,
        labels: np.ndarray,
        save_path: str = None
    ) -> Dict:
        """
        Analyse les cas où le VAE et le classificateur sont en désaccord
        
        Returns:
            analysis: Dict avec les statistiques de désaccord
        """
        _, details = self.predict(dataloader=dataloader, return_details=True)
        labels = np.array(labels)
        
        clf_pred = (details['classifier_probs'] > 0.5).astype(int)
        vae_pred = details['vae_anomaly']
        
        # Identifier les désaccords
        disagree_mask = clf_pred != vae_pred
        
        analysis = {
            'total_samples': len(labels),
            'disagreements': int(np.sum(disagree_mask)),
            'agreement_rate': float(1 - np.mean(disagree_mask)),
        }
        
        if np.sum(disagree_mask) > 0:
            # Analyser les désaccords
            disagree_labels = labels[disagree_mask]
            disagree_clf = clf_pred[disagree_mask]
            disagree_vae = vae_pred[disagree_mask]
            
            # VAE correct, Clf incorrect
            vae_correct = (disagree_vae == disagree_labels)
            clf_correct = (disagree_clf == disagree_labels)
            
            analysis['vae_wins'] = int(np.sum(vae_correct & ~clf_correct))
            analysis['clf_wins'] = int(np.sum(clf_correct & ~vae_correct))
            analysis['both_wrong'] = int(np.sum(~vae_correct & ~clf_correct))
            
            # Types de désaccords
            analysis['vae_flagged_benign_as_malignant'] = int(
                np.sum((vae_pred == 1) & (clf_pred == 0) & (labels == 0))
            )
            analysis['vae_caught_missed_malignant'] = int(
                np.sum((vae_pred == 1) & (clf_pred == 0) & (labels == 1))
            )
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(analysis, f, indent=2)
        
        return analysis
    
    def plot_fusion_analysis(
        self,
        dataloader: DataLoader,
        labels: np.ndarray,
        save_path: str = None
    ):
        """Visualise l'analyse de fusion"""
        _, details = self.predict(dataloader=dataloader, return_details=True)
        labels = np.array(labels)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Scatter plot VAE vs Classifier
        ax = axes[0, 0]
        scatter = ax.scatter(
            details['classifier_probs'],
            details['vae_scores'],
            c=labels,
            cmap='RdYlGn_r',
            alpha=0.5
        )
        ax.axhline(y=self.vae_threshold, color='red', linestyle='--', label='VAE Threshold')
        ax.axvline(x=0.5, color='blue', linestyle='--', label='Clf Threshold')
        ax.set_xlabel('Classifier P(Malignant)')
        ax.set_ylabel('VAE Reconstruction Error')
        ax.set_title('VAE vs Classifier Scores')
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='True Label')
        
        # 2. Distribution des scores de fusion
        ax = axes[0, 1]
        ax.hist(details['fusion_scores'][labels == 0], bins=30, alpha=0.7, 
               label='Benign', color='green')
        ax.hist(details['fusion_scores'][labels == 1], bins=30, alpha=0.7,
               label='Malignant', color='red')
        ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        ax.set_xlabel('Fusion Score')
        ax.set_ylabel('Count')
        ax.set_title('Fusion Score Distribution')
        ax.legend()
        
        # 3. ROC Curves comparison
        from sklearn.metrics import roc_curve, auc
        ax = axes[1, 0]
        
        for name, scores in [('Classifier', details['classifier_probs']),
                            ('VAE', details['vae_scores']),
                            ('Fusion', details['fusion_scores'])]:
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend()
        
        # 4. Confusion matrices
        ax = axes[1, 1]
        from sklearn.metrics import confusion_matrix
        
        predictions = (details['fusion_scores'] > 0.5).astype(int)
        cm = confusion_matrix(labels, predictions)
        
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix (Hybrid)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Fusion analysis saved to {save_path}")
        
        plt.show()
        return fig
    
    def save(self, output_dir: str):
        """Sauvegarde le classificateur hybride"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder les modèles
        torch.save(self.vae.state_dict(), output_dir / 'vae_model.pth')
        torch.save(self.classifier.state_dict(), output_dir / 'classifier_model.pth')
        
        # Sauvegarder la configuration
        config = {
            'fusion_strategy': self.fusion_strategy,
            'vae_threshold': self.vae_threshold,
            'vae_weight': self.vae_weight,
            'classifier_weight': self.classifier_weight,
            'vae_score_mean': getattr(self, 'vae_score_mean', None),
            'vae_score_std': getattr(self, 'vae_score_std', None),
            'calibration_params': self.calibration_params,
            'vae_config': self.vae_config.__dict__
        }
        
        with open(output_dir / 'hybrid_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Hybrid classifier saved to {output_dir}")
    
    @classmethod
    def load(cls, input_dir: str, device: torch.device = None) -> 'HybridClassifier':
        """Charge un classificateur hybride sauvegardé"""
        input_dir = Path(input_dir)
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Charger la configuration
        with open(input_dir / 'hybrid_config.json', 'r') as f:
            config = json.load(f)
        
        # Créer le VAE
        vae_config = VAEConfig(**config['vae_config'])
        vae = VAE(vae_config)
        vae.load_state_dict(torch.load(input_dir / 'vae_model.pth', map_location=device))
        
        # Créer le classificateur
        if create_densenet121_model is not None:
            classifier = create_densenet121_model(num_classes=2, pretrained=False, device='cpu')
        else:
            from torchvision import models
            classifier = models.densenet121(pretrained=False)
            classifier.classifier = nn.Linear(classifier.classifier.in_features, 2)
        
        classifier.load_state_dict(torch.load(input_dir / 'classifier_model.pth', map_location=device))
        
        # Créer l'instance
        instance = cls(
            vae_model=vae,
            classifier_model=classifier,
            device=device,
            fusion_strategy=config['fusion_strategy']
        )
        
        # Restaurer les paramètres de calibration
        instance.vae_threshold = config['vae_threshold']
        instance.vae_weight = config['vae_weight']
        instance.classifier_weight = config['classifier_weight']
        instance.vae_score_mean = config.get('vae_score_mean')
        instance.vae_score_std = config.get('vae_score_std')
        instance.calibration_params = config.get('calibration_params', {})
        instance.is_calibrated = True
        
        logger.info(f"Hybrid classifier loaded from {input_dir}")
        return instance


if __name__ == '__main__':
    print("Hybrid Classifier module loaded successfully!")
    print("Use HybridClassifier class to combine VAE and supervised classifier")
