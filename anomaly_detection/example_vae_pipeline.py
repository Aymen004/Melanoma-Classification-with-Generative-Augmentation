#!/usr/bin/env python3
"""
Example: Complete VAE Anomaly Detection Pipeline
================================================

Ce script démontre le pipeline complet de détection d'anomalies:
1. Entraînement du VAE sur images bénignes
2. Calibrage du seuil de détection
3. Évaluation sur données de test
4. Intégration avec le classificateur hybride

Usage:
    python example_vae_pipeline.py --data_dir ./data/isic2016
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Ajouter le chemin parent
sys.path.insert(0, str(Path(__file__).parent))

from VAE_model import VAE, VAEConfig, vae_loss_function, get_model_info
from train_vae import VAETrainer, BenignOnlyDataset, get_transforms
from inference_vae import VAEAnomalyDetector, calibrate_threshold
from hybrid_classifier import HybridClassifier


def create_sample_dataset(num_samples=1000, image_size=128):
    """Crée un dataset synthétique pour la démonstration"""
    print("Creating synthetic dataset for demonstration...")
    
    # Simuler des images "bénignes" (cercles réguliers, couleur uniforme)
    benign_images = []
    for i in range(num_samples):
        img = np.zeros((image_size, image_size, 3), dtype=np.float32)
        
        # Fond beige/rose (peau)
        img[:, :, 0] = 0.8 + np.random.uniform(-0.1, 0.1)
        img[:, :, 1] = 0.7 + np.random.uniform(-0.1, 0.1)
        img[:, :, 2] = 0.6 + np.random.uniform(-0.1, 0.1)
        
        # Lésion bénigne (cercle régulier, marron uniforme)
        center = (image_size // 2 + np.random.randint(-10, 10),
                 image_size // 2 + np.random.randint(-10, 10))
        radius = np.random.randint(20, 35)
        
        y, x = np.ogrid[:image_size, :image_size]
        mask = ((x - center[0])**2 + (y - center[1])**2) <= radius**2
        
        img[mask, 0] = 0.4 + np.random.uniform(-0.05, 0.05)
        img[mask, 1] = 0.3 + np.random.uniform(-0.05, 0.05)
        img[mask, 2] = 0.2 + np.random.uniform(-0.05, 0.05)
        
        benign_images.append(img)
    
    # Simuler des images "malignes" (formes irrégulières, couleurs variées)
    malignant_images = []
    for i in range(num_samples // 5):  # Moins d'images malignes (classe minoritaire)
        img = np.zeros((image_size, image_size, 3), dtype=np.float32)
        
        # Fond
        img[:, :, 0] = 0.8 + np.random.uniform(-0.1, 0.1)
        img[:, :, 1] = 0.7 + np.random.uniform(-0.1, 0.1)
        img[:, :, 2] = 0.6 + np.random.uniform(-0.1, 0.1)
        
        # Lésion maligne (forme irrégulière, couleurs variées)
        center = (image_size // 2, image_size // 2)
        
        # Créer une forme irrégulière
        y, x = np.ogrid[:image_size, :image_size]
        theta = np.arctan2(y - center[1], x - center[0])
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        # Rayon variable (irrégulier)
        base_radius = np.random.randint(25, 40)
        radius_var = base_radius + 10 * np.sin(5 * theta) + 5 * np.random.randn()
        mask = r < radius_var
        
        # Couleurs variées dans la lésion
        noise = np.random.randn(image_size, image_size) * 0.15
        img[mask, 0] = np.clip(0.2 + noise[mask], 0, 1)
        img[mask, 1] = np.clip(0.1 + noise[mask] * 0.5, 0, 1)
        img[mask, 2] = np.clip(0.15 + noise[mask] * 0.3, 0, 1)
        
        # Ajouter des zones plus foncées (asymétrie)
        dark_spot = np.random.rand(image_size, image_size) < 0.3
        img[mask & dark_spot, :] *= 0.5
        
        malignant_images.append(img)
    
    return np.array(benign_images), np.array(malignant_images)


class SyntheticDataset(Dataset):
    """Dataset pour les données synthétiques"""
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        # Convertir en tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            return image, self.labels[idx]
        return image


def run_demo(args):
    """Execute la démonstration complète"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("VAE Anomaly Detection - Complete Pipeline Demo")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # =========================================================================
    # 1. CRÉATION/CHARGEMENT DES DONNÉES
    # =========================================================================
    print("[1/5] Preparing data...")
    
    if args.use_synthetic:
        # Créer des données synthétiques
        benign_images, malignant_images = create_sample_dataset(
            num_samples=500, 
            image_size=args.image_size
        )
        
        # Split train/val/test pour les bénins
        benign_train, benign_temp = train_test_split(benign_images, test_size=0.3, random_state=42)
        benign_val, benign_test = train_test_split(benign_temp, test_size=0.5, random_state=42)
        
        # Split pour les malins (val et test uniquement)
        malignant_val, malignant_test = train_test_split(malignant_images, test_size=0.5, random_state=42)
        
        print(f"  Benign - Train: {len(benign_train)}, Val: {len(benign_val)}, Test: {len(benign_test)}")
        print(f"  Malignant - Val: {len(malignant_val)}, Test: {len(malignant_test)}")
        
        # Créer les datasets
        train_dataset = SyntheticDataset(benign_train)
        
        # Validation avec mix bénin/malin
        val_images = np.concatenate([benign_val, malignant_val])
        val_labels = np.concatenate([np.zeros(len(benign_val)), np.ones(len(malignant_val))])
        val_dataset = SyntheticDataset(val_images, val_labels)
        
        # Test avec mix bénin/malin
        test_images = np.concatenate([benign_test, malignant_test])
        test_labels = np.concatenate([np.zeros(len(benign_test)), np.ones(len(malignant_test))])
        test_dataset = SyntheticDataset(test_images, test_labels)
        
    else:
        print("  Loading data from directory...")
        train_transform, val_transform = get_transforms(args.image_size, 'standard')
        
        train_dataset = BenignOnlyDataset(
            csv_file=args.data_csv,
            img_dir=args.img_dir,
            transform=train_transform
        )
        
        # Pour les vrais datasets, diviser manuellement
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    if args.use_synthetic:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # =========================================================================
    # 2. CONFIGURATION ET CRÉATION DU VAE
    # =========================================================================
    print("\n[2/5] Creating VAE model...")
    
    config = VAEConfig(
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        hidden_dims=[32, 64, 128, 256],
        beta=args.beta,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs
    )
    
    model = VAE(config).to(device)
    get_model_info(model)
    
    # =========================================================================
    # 3. ENTRAÎNEMENT DU VAE
    # =========================================================================
    print("\n[3/5] Training VAE on benign images only...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer un val_loader sans labels pour le trainer
    class ImageOnlyDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            item = self.dataset[idx]
            if isinstance(item, tuple):
                return item[0]
            return item
    
    train_loader_images = DataLoader(
        ImageOnlyDataset(train_dataset), 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    trainer = VAETrainer(
        model=model,
        config=config,
        train_loader=train_loader_images,
        val_loader=None,  # Pas de validation pendant l'entraînement
        device=device,
        output_dir=str(output_dir)
    )
    
    history = trainer.train(num_epochs=args.epochs, save_every=5)
    
    print(f"  Final training loss: {history['train_loss'][-1]:.4f}")
    
    # =========================================================================
    # 4. CALIBRAGE DU SEUIL
    # =========================================================================
    print("\n[4/5] Calibrating anomaly detection threshold...")
    
    detector = VAEAnomalyDetector(model=model, config=config, device=device)
    
    if args.use_synthetic:
        # Calculer les erreurs pour calibration
        all_errors = []
        all_labels_list = []
        
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                errors = model.get_reconstruction_error(images)
                all_errors.extend(errors.cpu().numpy())
                all_labels_list.extend(labels.numpy())
        
        all_errors = np.array(all_errors)
        all_labels_arr = np.array(all_labels_list)
        
        # Calibrer
        errors_benign = all_errors[all_labels_arr == 0]
        errors_malignant = all_errors[all_labels_arr == 1]
        
        threshold, stats = calibrate_threshold(
            errors_benign, 
            errors_malignant,
            percentile=95.0,
            method='optimal_f1'
        )
        
        detector.threshold = threshold
        detector.calibration_stats = stats
        detector.is_calibrated = True
        
        print(f"  Threshold: {threshold:.6f}")
        print(f"  Benign errors - Mean: {stats['benign_mean']:.6f}, Std: {stats['benign_std']:.6f}")
        print(f"  Malignant errors - Mean: {stats['malignant_mean']:.6f}, Std: {stats['malignant_std']:.6f}")
        
        # Visualiser
        plt.figure(figsize=(10, 6))
        plt.hist(errors_benign, bins=30, alpha=0.7, label='Benign', color='green', density=True)
        plt.hist(errors_malignant, bins=30, alpha=0.7, label='Malignant', color='red', density=True)
        plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
        plt.xlabel('Reconstruction Error (MSE)')
        plt.ylabel('Density')
        plt.title('Distribution of Reconstruction Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'error_distribution.png', dpi=150)
        plt.close()
        print(f"  Distribution plot saved to {output_dir / 'error_distribution.png'}")
    
    # =========================================================================
    # 5. ÉVALUATION
    # =========================================================================
    print("\n[5/5] Evaluating on test set...")
    
    if args.use_synthetic:
        # Prédire sur le test set
        predictions, scores = detector.predict(dataloader=test_loader, threshold=threshold)
        
        # Évaluer
        metrics = detector.evaluate(test_loader, test_labels, threshold=threshold)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f} (Recall for malignant)")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  F1-Score:    {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        print("="*60)
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        
        # ROC Curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(test_labels, scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - VAE Anomaly Detection')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'roc_curve.png', dpi=150)
        plt.close()
        print(f"\nROC curve saved to {output_dir / 'roc_curve.png'}")
    
    # Sauvegarder la calibration
    detector.save_calibration(str(output_dir / 'calibration.json'))
    
    print(f"\n{'='*60}")
    print("Demo completed successfully!")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='VAE Anomaly Detection Demo')
    
    # Data
    parser.add_argument('--use_synthetic', action='store_true', default=True,
                       help='Use synthetic data for demo')
    parser.add_argument('--data_csv', type=str, default=None,
                       help='Path to CSV file with image info')
    parser.add_argument('--img_dir', type=str, default=None,
                       help='Directory containing images')
    
    # Model
    parser.add_argument('--image_size', type=int, default=64,
                       help='Image size (smaller for demo)')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta coefficient for KL')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs (fewer for demo)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./vae_demo_output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Si pas de données synthétiques, vérifier les chemins
    if not args.use_synthetic:
        if args.img_dir is None:
            parser.error("--img_dir is required when not using synthetic data")
    
    run_demo(args)


if __name__ == '__main__':
    main()
