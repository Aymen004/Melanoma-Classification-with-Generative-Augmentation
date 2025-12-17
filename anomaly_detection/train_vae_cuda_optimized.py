"""
Training Script pour VAE - Optimisé CUDA
=========================================

Ce script entraîne le VAE sur des images bénignes uniquement pour
apprendre la distribution "normale" et détecter les anomalies.

Optimisations:
- Mixed precision training (AMP)
- Gradient accumulation
- DataLoader optimisé
- Checkpointing intelligent

Usage:
    python train_vae_cuda_optimized.py \\
        --img_dir ./data/calibrage/calibrage_data \\
        --output_dir vae_output_noaug_optimized \\
        --epochs 1000 \\
        --batch_size 32 \\
        --lr 1e-4 \\
        --beta 0.01 \\
        --save_every 10
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

from VAE_model import ConvVAE, VAEConfig, vae_loss_function, get_model_info

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenignOnlyDataset(Dataset):
    """
    Dataset pour charger uniquement les images bénignes
    """
    
    def __init__(self, img_dir: str, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Charger toutes les images
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.image_paths = [
            p for p in self.img_dir.rglob('*') 
            if p.suffix.lower() in valid_extensions
        ]
        
        logger.info(f"Found {len(self.image_paths)} images in {img_dir}")
    
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
            return torch.zeros(3, 128, 128)


def get_transforms(image_size: int = 128, augmentation: str = 'none'):
    """
    ✅ SOLUTION 1 APPLIQUÉE: Pas de normalisation vers [-1, 1]
    ToTensor() seul met déjà les données entre [0, 1], parfait pour Sigmoid
    """
    
    base_transform = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()  # ✅ Convertit en [0, 1] directement
    ]
    
    if augmentation == 'standard':
        train_aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ]
    elif augmentation == 'heavy':
        train_aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
        ]
    else:  # 'none'
        train_aug = []
    
    train_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size))] + 
        train_aug + 
        [transforms.ToTensor()]
        # ✅ PAS de Normalize ici - on reste en [0, 1]
    )
    
    val_transform = transforms.Compose(base_transform)
    
    return train_transform, val_transform


class VAETrainer:
    """
    Trainer optimisé pour VAE avec:
    - Mixed precision
    - Gradient accumulation
    - Smart checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: dict,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=20
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        self.use_amp = config.get('use_amp', True)
        
        # Tracking
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'train_recon': [],
            'train_kl': [],
            'val_loss': [],
            'val_recon': [],
            'val_kl': []
        }
        
        # Output directories
        self.output_dir = Path(config['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.sample_dir = self.output_dir / 'samples'
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Entraîne une epoch"""
        self.model.train()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, images in enumerate(pbar):
            images = images.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward avec mixed precision
            if self.use_amp:
                with autocast():
                    reconstruction, mu, logvar = self.model(images)
                    loss, recon_loss, kl_loss = vae_loss_function(
                        reconstruction, images, mu, logvar,
                        beta=self.config['beta']
                    )
                
                # Backward avec gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                reconstruction, mu, logvar = self.model(images)
                loss, recon_loss, kl_loss = vae_loss_function(
                    reconstruction, images, mu, logvar,
                    beta=self.config['beta']
                )
                loss.backward()
                self.optimizer.step()
            
            # Tracking
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'kl': f"{kl_loss.item():.4f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_recon = total_recon / len(self.train_loader)
        avg_kl = total_kl / len(self.train_loader)
        
        return avg_loss, avg_recon, avg_kl
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """Valide le modèle"""
        if self.val_loader is None:
            return 0.0, 0.0, 0.0
        
        self.model.eval()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for images in self.val_loader:
            images = images.to(self.device)
            
            reconstruction, mu, logvar = self.model(images)
            loss, recon_loss, kl_loss = vae_loss_function(
                reconstruction, images, mu, logvar,
                beta=self.config['beta']
            )
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_recon = total_recon / len(self.val_loader)
        avg_kl = total_kl / len(self.val_loader)
        
        return avg_loss, avg_recon, avg_kl
    
    @torch.no_grad()
    def save_samples(self, epoch: int, num_samples: int = 8):
        """Sauvegarde des reconstructions"""
        self.model.eval()
        
        # Prendre un batch de validation
        if self.val_loader:
            images = next(iter(self.val_loader))[:num_samples]
        else:
            images = next(iter(self.train_loader))[:num_samples]
        
        images = images.to(self.device)
        reconstructions = self.model.reconstruct(images)
        
        # Créer la grille de comparaison
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
        
        for i in range(num_samples):
            # Original
            img_orig = images[i].cpu().permute(1, 2, 0).numpy()
            axes[0, i].imshow(img_orig)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)
            
            # Reconstruction
            img_recon = reconstructions[i].cpu().permute(1, 2, 0).numpy()
            axes[1, i].imshow(img_recon)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.sample_dir / f'reconstruction_epoch_{epoch:04d}.png', dpi=100)
        plt.close()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Sauvegarde un checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
            'config': self.config
        }
        
        # Checkpoint régulier
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"✅ New best model saved! Loss: {self.best_loss:.6f}")
    
    def train(self, num_epochs: int, save_every: int = 10, checkpoint_every: int = 25, resume_from: Optional[str] = None):
        """Boucle d'entraînement principale"""
        
        start_epoch = 0
        
        # Resume training
        if resume_from:
            logger.info(f"Resuming from {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint['best_loss']
            self.history = checkpoint['history']
        
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Total epochs: {num_epochs}")
        logger.info(f"Batch size: {self.config['batch_size']}")
        logger.info(f"Learning rate: {self.config['lr']}")
        logger.info(f"Beta (KL weight): {self.config['beta']}")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_recon, train_kl = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_recon, val_kl = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss if val_loss > 0 else train_loss)
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['train_recon'].append(train_recon)
            self.history['train_kl'].append(train_kl)
            self.history['val_loss'].append(val_loss)
            self.history['val_recon'].append(val_recon)
            self.history['val_kl'].append(val_kl)
            
            epoch_time = time.time() - epoch_start
            
            # Logging
            logger.info(
                f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_loss:.6f} (Recon: {train_recon:.6f}, KL: {train_kl:.6f}) - "
                f"Val Loss: {val_loss:.6f}"
            )
            
            # Check for best model
            current_loss = val_loss if val_loss > 0 else train_loss
            is_best = current_loss < self.best_loss
            if is_best:
                self.best_loss = current_loss
            
            # Save samples
            if (epoch + 1) % save_every == 0:
                self.save_samples(epoch + 1)
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_every == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best=is_best)
        
        logger.info("✅ Training completed!")
        logger.info(f"Best loss: {self.best_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Train VAE on benign images')
    
    # Data
    parser.add_argument('--img_dir', type=str, default='./data/calibrage/calibrage_data',
                       help='Directory containing benign images')
    parser.add_argument('--image_size', type=int, default=128,
                       help='Image size (default: 128)')
    parser.add_argument('--augmentation', type=str, default='none',
                       choices=['none', 'standard', 'heavy'],
                       help='Data augmentation level')
    
    # Model
    parser.add_argument('--model_type', type=str, default='conv_vae',
                       choices=['conv_vae'],
                       help='VAE architecture')
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='Latent dimension')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.01,
                       help='Beta for KL divergence (beta-VAE)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='vae_output_noaug_optimized',
                       help='Output directory')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save samples every N epochs')
    parser.add_argument('--checkpoint_every', type=int, default=25,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    # System
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Transforms
    train_transform, val_transform = get_transforms(args.image_size, args.augmentation)
    
    # Dataset
    full_dataset = BenignOnlyDataset(args.img_dir, transform=train_transform)
    
    # Split train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Update val dataset transform
    val_dataset.dataset.transform = val_transform
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    vae_config = VAEConfig(
        image_size=args.image_size,
        latent_dim=args.latent_dim
    )
    model = ConvVAE(vae_config).to(device)
    
    # Model info
    info = get_model_info(model)
    logger.info("=== Model Information ===")
    for k, v in info.items():
        logger.info(f"{k}: {v}")
    
    # Training config
    train_config = {
        'lr': args.lr,
        'beta': args.beta,
        'batch_size': args.batch_size,
        'output_dir': args.output_dir,
        'use_amp': True
    }
    
    # Trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device
    )
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume
    )


if __name__ == '__main__':
    main()
