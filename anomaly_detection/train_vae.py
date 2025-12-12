"""
Script d'Entraînement du VAE sur Données Bénignes
=================================================

Ce script entraîne le VAE exclusivement sur les lésions bénignes
pour apprendre la distribution de la "normalité".

Processus:
    1. Charger uniquement les images bénignes (nævus, kératose séborrhéique, etc.)
    2. Entraîner le VAE à reconstruire ces images
    3. Le VAE apprend ce qu'est une lésion "normale"
    4. À l'inférence, les anomalies auront une erreur de reconstruction élevée

Usage:
    python train_vae.py --data_dir ./data/isic2016 --epochs 100
    python train_vae.py --data_csv benign_images.csv --img_dir ./images
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import logging

from VAE_model import VAE, VAEConfig, ConvVAE, vae_loss_function, get_model_info

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenignOnlyDataset(Dataset):
    """
    Dataset contenant uniquement les images bénignes
    
    Ce dataset est utilisé pour entraîner le VAE à apprendre
    la distribution des lésions "normales".
    
    Args:
        csv_file: Fichier CSV avec les informations des images
        img_dir: Répertoire contenant les images
        transform: Transformations à appliquer
        benign_labels: Liste des labels considérés comme bénins
    """
    def __init__(
        self, 
        csv_file: str = None,
        img_dir: str = None,
        transform=None,
        benign_labels: list = None,
        image_list: list = None
    ):
        self.img_dir = Path(img_dir) if img_dir else None
        self.transform = transform
        
        if benign_labels is None:
            # Labels typiquement bénins dans ISIC
            benign_labels = [0, 'benign', 'Benign', 'nevus', 'Nevus', 
                           'seborrheic_keratosis', 'SK']
        
        if image_list is not None:
            # Liste directe d'images
            self.image_paths = image_list
        elif csv_file is not None:
            # Charger depuis CSV
            df = pd.read_csv(csv_file)
            
            # Auto-détecter la colonne d'image
            img_col = None
            for col in ['new_name', 'filename', 'image_name', 'image_id', 'image']:
                if col in df.columns:
                    img_col = col
                    break
            
            if img_col is None:
                raise ValueError("No valid image column found in CSV")
            
            # Auto-détecter la colonne de label
            label_col = None
            for col in ['label', 'target', 'class', 'diagnosis']:
                if col in df.columns:
                    label_col = col
                    break
            
            if label_col is None:
                logger.warning("No label column found, using all images")
                benign_df = df
            else:
                # Filtrer uniquement les bénins
                benign_df = df[df[label_col].isin(benign_labels)]
                logger.info(f"Filtered {len(benign_df)} benign images from {len(df)} total")
            
            # Construire les chemins d'images
            self.image_paths = []
            for _, row in benign_df.iterrows():
                img_name = row[img_col]
                img_path = self._find_image(img_name)
                if img_path:
                    self.image_paths.append(img_path)
        else:
            # Scanner le répertoire pour toutes les images
            self.image_paths = []
            if self.img_dir:
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    self.image_paths.extend(list(self.img_dir.glob(ext)))
        
        logger.info(f"BenignOnlyDataset created with {len(self.image_paths)} images")
    
    def _find_image(self, img_name: str) -> str:
        """Trouve le chemin complet de l'image"""
        if self.img_dir is None:
            return None
            
        extensions = ['', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for ext in extensions:
            test_path = self.img_dir / f"{img_name}{ext}"
            if test_path.exists():
                return str(test_path)
        
        # Essayer sans le répertoire (chemin absolu dans le CSV)
        if os.path.exists(img_name):
            return img_name
        
        return None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            # Retourner une image noire en cas d'erreur
            if self.transform:
                return torch.zeros(3, 128, 128)
            return Image.new('RGB', (128, 128), (0, 0, 0))


def get_transforms(image_size: int = 128, augmentation: str = 'standard'):
    """
    Retourne les transformations pour l'entraînement et la validation
    
    Args:
        image_size: Taille des images
        augmentation: 'none', 'light', 'standard', 'heavy'
    """
    # Normalisation vers [0, 1] pour la reconstruction
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    # Transformations de base
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    
    # Augmentations pour l'entraînement
    if augmentation == 'none':
        train_aug = []
    elif augmentation == 'light':
        train_aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
        ]
    elif augmentation == 'standard':
        train_aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    elif augmentation == 'heavy':
        train_aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]
    else:
        train_aug = []
    
    train_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size))] + 
        train_aug + 
        [transforms.ToTensor()]
    )
    
    val_transform = transforms.Compose(base_transforms)
    
    return train_transform, val_transform


class VAETrainer:
    """
    Trainer pour le Variational Autoencoder
    
    Gère l'entraînement, la validation, et la sauvegarde des checkpoints.
    """
    def __init__(
        self,
        model: nn.Module,
        config: VAEConfig,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        device: torch.device = None,
        output_dir: str = './vae_output'
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        
        # Créer les répertoires
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.samples_dir = self.output_dir / 'samples'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Déplacer le modèle sur le device
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10,
            verbose=True
        )
        
        # Historique
        self.history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'val_loss': [],
            'val_recon_loss': [],
            'val_kl_loss': []
        }
        
        # Meilleure loss
        self.best_val_loss = float('inf')
        
        # Images fixes pour visualiser la progression
        self.fixed_images = None
        
        logger.info(f"VAETrainer initialized on {self.device}")
    
    def train_epoch(self, epoch: int) -> dict:
        """Entraîne une époque"""
        self.model.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, images in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            recon, mu, logvar = self.model(images)
            
            # Loss
            loss, recon_loss, kl_loss = vae_loss_function(
                recon, images, mu, logvar,
                beta=self.config.beta,
                reconstruction_loss_type=self.config.reconstruction_loss
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })
            
            # Sauvegarder les images fixes pour la visualisation
            if self.fixed_images is None:
                self.fixed_images = images[:8].clone()
        
        num_batches = len(self.train_loader)
        return {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Valide le modèle"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for images in self.val_loader:
            images = images.to(self.device)
            
            recon, mu, logvar = self.model(images)
            
            loss, recon_loss, kl_loss = vae_loss_function(
                recon, images, mu, logvar,
                beta=self.config.beta,
                reconstruction_loss_type=self.config.reconstruction_loss
            )
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        
        num_batches = len(self.val_loader)
        return {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches
        }
    
    @torch.no_grad()
    def save_samples(self, epoch: int):
        """Sauvegarde des échantillons de reconstruction et de génération"""
        self.model.eval()
        
        # Reconstructions des images fixes
        if self.fixed_images is not None:
            fixed = self.fixed_images.to(self.device)
            recon, _, _ = self.model(fixed)
            
            # Concatener originales et reconstructions
            comparison = torch.cat([fixed, recon], dim=0)
            save_image(
                comparison.cpu(),
                self.samples_dir / f'reconstruction_epoch_{epoch}.png',
                nrow=8,
                normalize=True
            )
        
        # Générer de nouvelles images
        samples = self.model.sample(16, self.device)
        save_image(
            samples.cpu(),
            self.samples_dir / f'samples_epoch_{epoch}.png',
            nrow=4,
            normalize=True
        )
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Sauvegarde un checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        # Sauvegarder le checkpoint courant
        torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # Sauvegarder le meilleur modèle
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pth')
            logger.info(f"✓ New best model saved (val_loss: {self.best_val_loss:.4f})")
        
        # Garder seulement les N derniers checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        for ckpt in checkpoints[:-5]:  # Garder les 5 derniers
            ckpt.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Charge un checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def train(self, num_epochs: int = None, save_every: int = 10):
        """
        Boucle d'entraînement principale
        
        Args:
            num_epochs: Nombre d'époques (override config)
            save_every: Fréquence de sauvegarde des échantillons
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training on {len(self.train_loader.dataset)} images")
        if self.val_loader:
            logger.info(f"Validating on {len(self.val_loader.dataset)} images")
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon_loss'].append(train_metrics['recon_loss'])
            self.history['train_kl_loss'].append(train_metrics['kl_loss'])
            
            # Validation
            val_metrics = self.validate()
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_recon_loss'].append(val_metrics['recon_loss'])
                self.history['val_kl_loss'].append(val_metrics['kl_loss'])
                
                # Update scheduler
                self.scheduler.step(val_metrics['loss'])
                
                # Check for best model
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
            else:
                is_best = train_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = train_metrics['loss']
            
            # Logging
            log_msg = f"Epoch {epoch}/{num_epochs} | Train Loss: {train_metrics['loss']:.4f}"
            if val_metrics:
                log_msg += f" | Val Loss: {val_metrics['loss']:.4f}"
            logger.info(log_msg)
            
            # Save samples
            if epoch % save_every == 0 or epoch == 1:
                self.save_samples(epoch)
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        # Final save
        self.save_checkpoint(num_epochs)
        self.plot_training_curves()
        
        logger.info("Training completed!")
        return self.history
    
    def plot_training_curves(self):
        """Plot les courbes d'apprentissage"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Total Loss
        axes[0].plot(epochs, self.history['train_loss'], label='Train')
        if self.history['val_loss']:
            axes[0].plot(epochs, self.history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Reconstruction Loss
        axes[1].plot(epochs, self.history['train_recon_loss'], label='Train')
        if self.history['val_recon_loss']:
            axes[1].plot(epochs, self.history['val_recon_loss'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Reconstruction Loss')
        axes[1].set_title('Reconstruction Loss (MSE)')
        axes[1].legend()
        axes[1].grid(True)
        
        # KL Divergence
        axes[2].plot(epochs, self.history['train_kl_loss'], label='Train')
        if self.history['val_kl_loss']:
            axes[2].plot(epochs, self.history['val_kl_loss'], label='Validation')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('KL Divergence')
        axes[2].set_title('KL Divergence')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150)
        plt.close()
        
        logger.info(f"Training curves saved to {self.output_dir / 'training_curves.png'}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train VAE on Benign Images')
    
    # Data arguments
    parser.add_argument('--data_csv', type=str, default=None,
                       help='Path to CSV file with image info')
    parser.add_argument('--img_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--benign_only', action='store_true', default=True,
                       help='Train only on benign images')
    
    # Model arguments
    parser.add_argument('--image_size', type=int, default=128,
                       help='Image size')
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='Latent dimension')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Beta coefficient for KL divergence')
    parser.add_argument('--model_type', type=str, default='vae',
                       choices=['vae', 'conv_vae'],
                       help='VAE architecture type')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--augmentation', type=str, default='standard',
                       choices=['none', 'light', 'standard', 'heavy'],
                       help='Data augmentation level')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./vae_output',
                       help='Output directory')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save samples every N epochs')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Configuration
    config = VAEConfig(
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        beta=args.beta,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs
    )
    
    # Transforms
    train_transform, val_transform = get_transforms(
        image_size=args.image_size,
        augmentation=args.augmentation
    )
    
    # Dataset
    dataset = BenignOnlyDataset(
        csv_file=args.data_csv,
        img_dir=args.img_dir,
        transform=train_transform
    )
    
    # Train/Val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    
    # Model
    if args.model_type == 'vae':
        model = VAE(config)
    else:
        model = ConvVAE(config)
    
    get_model_info(model)
    
    # Trainer
    trainer = VAETrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir
    )
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs - start_epoch,
        save_every=args.save_every
    )
    
    # Save config
    config_path = Path(args.output_dir) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    logger.info(f"Training complete. Output saved to {args.output_dir}")


if __name__ == '__main__':
    main()
