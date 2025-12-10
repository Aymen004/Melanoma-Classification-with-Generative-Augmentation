import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging
from huggingface_hub import hf_hub_download
from diffusers import DDPMPipeline, UNet2DModel, DDPMScheduler

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DDPMPretrainedConfig:
    """Configuration pour le DDPM pré-entraîné"""
    def __init__(self):
        # Paramètres du modèle - optimisés avec pré-entraîné
        self.image_size = 64  # Compatible avec google/ddpm-ema-imagenet-64
        self.batch_size = 16  # Augmenté grâce au pré-entraîné
        self.learning_rate = 1e-5  # Garder faible pour fine-tuning
        self.num_epochs = 3000  # Moins d'époques nécessaires
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Optimisations CUDA
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
        # Chemins CORRIGÉS pour éviter les conflits
        self.data_dir = Path(__file__).parent / "train" / "ISBI2016_ISIC_Part3_Training_Data"
        self.train_csv_dir = Path(__file__).parent / "train"
        self.malignant_csv = self.train_csv_dir / "malignant_images.csv"
        
        # CORRECTION: Utiliser DDPM_pretrained pour éviter les conflits
        self.checkpoint_dir = Path(__file__).parent / "DDPM_pretrained" / "checkpoints"
        self.output_dir = Path(__file__).parent / "DDPM_pretrained" / "samples"
        
        # Créer les dossiers
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Modèle pré-entraîné
        self.pretrained_model = "google/ddpm-ema-imagenet-64"

class MalignantDataset(Dataset):
    """Dataset pour les images malignes - version optimisée"""
    def __init__(self, csv_file, data_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Vérifier les images existantes
        valid_images = []
        logger.info(f"Recherche d'images dans: {self.data_dir}")
        logger.info(f"Nombre d'entrées dans le CSV: {len(self.df)}")
        
        # Créer un mapping efficace des fichiers
        image_mapping = {}
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for img_file in self.data_dir.glob(ext):
                stem = img_file.stem
                image_mapping[stem] = img_file
        
        logger.info(f"Fichiers d'images trouvés: {len(image_mapping)}")
        
        # Associer les images du CSV
        for _, row in self.df.iterrows():
            image_name = row['image_name']
            if image_name in image_mapping:
                valid_images.append((image_name, str(image_mapping[image_name])))
            else:
                logger.warning(f"Image non trouvée: {image_name}")
        
        self.valid_images = valid_images
        logger.info(f"Dataset créé avec {len(self.valid_images)} images malignes")
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx):
        image_name, image_path = self.valid_images[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            logger.warning(f"Erreur lors du chargement de {image_path}: {e}")
            # Retourner une image vide correctement dimensionnée
            return torch.zeros(3, 64, 64)

class DDPMPretrainedTrainer:
    """Trainer pour fine-tuning d'un modèle DDPM pré-entraîné"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        logger.info("Chargement du modèle pré-entraîné...")
        
        try:
            # Charger le pipeline pré-entraîné
            self.pipeline = DDPMPipeline.from_pretrained(
                config.pretrained_model,
                torch_dtype=torch.float32  # Assurer la compatibilité
            ).to(self.device)
            
            # Extraire les composants
            self.unet = self.pipeline.unet
            self.scheduler = self.pipeline.scheduler
            
            logger.info(f"✓ Modèle pré-entraîné chargé: {config.pretrained_model}")
            logger.info(f"✓ Paramètres du modèle: {sum(p.numel() for p in self.unet.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle pré-entraîné: {e}")
            logger.info("Utilisation d'un modèle local...")
            
            # Fallback vers un modèle local si le téléchargement échoue
            self.unet = UNet2DModel(
                sample_size=64,
                in_channels=3,
                out_channels=3,
                layers_per_block=2,
                block_out_channels=(128, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D", 
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            ).to(self.device)
            
            self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        # Optimiseur pour fine-tuning
        self.optimizer = optim.AdamW(
            self.unet.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-6,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler de learning rate
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=1e-7
        )
        
        # Mixed precision pour efficacité
        try:
            from torch.amp import GradScaler
            self.scaler = GradScaler('cuda')
        except ImportError:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
    
    def train_step(self, batch):
        """Étape d'entraînement optimisée"""
        batch = batch.to(self.device)
        batch_size = batch.shape[0]
        
        # Échantillonner des timesteps aléatoires
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()
        
        # Ajouter du bruit aux images
        noise = torch.randn_like(batch)
        noisy_images = self.scheduler.add_noise(batch, noise, timesteps)
        
        # Prédiction avec mixed precision
        try:
            from torch.amp import autocast
            autocast_context = autocast('cuda')
        except ImportError:
            from torch.cuda.amp import autocast
            autocast_context = autocast()
        
        with autocast_context:
            # Prédire le bruit
            noise_pred = self.unet(noisy_images, timesteps).sample
            
            # Calculer la loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Vérifier la stabilité
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("Loss NaN détectée, skipping batch")
                return 0.0
        
        # Backward pass avec gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    @torch.no_grad()
    def generate_samples(self, num_samples=16):
        """Génération d'échantillons"""
        self.unet.eval()
        
        # Commencer avec du bruit pur
        shape = (num_samples, 3, self.config.image_size, self.config.image_size)
        image = torch.randn(shape, device=self.device)
        
        # Processus de débruitage
        self.scheduler.set_timesteps(50)  # Moins d'étapes pour plus de rapidité
        
        for t in tqdm(self.scheduler.timesteps, desc="Génération"):
            # Prédire le bruit
            noise_pred = self.unet(image, t).sample
            
            # Débruiter
            image = self.scheduler.step(noise_pred, t, image).prev_sample
        
        return image
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Sauvegarder un checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'loss': loss,
            'config': self.config.__dict__,
            'pretrained_model': self.config.pretrained_model
        }
        
        # Sauvegarder checkpoint normal
        checkpoint_path = self.config.checkpoint_dir / f"ddpm_pretrained_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Sauvegarder checkpoint "latest"
        latest_path = self.config.checkpoint_dir / "ddpm_pretrained_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Sauvegarder meilleur modèle
        if is_best:
            best_path = self.config.checkpoint_dir / "ddpm_pretrained_best.pt"
            torch.save(checkpoint, best_path)
        
        logger.info(f"Checkpoint sauvegardé: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Charger un checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as e:
            logger.warning(f"Chargement avec weights_only=False échoué: {e}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Charger les états
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except Exception as e:
                logger.warning(f"Impossible de charger scaler state: {e}")
        
        logger.info(f"Checkpoint chargé: {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['loss']
    
    def find_latest_checkpoint(self):
        """Trouver le checkpoint le plus récent"""
        if not self.config.checkpoint_dir.exists():
            return None
        
        # Chercher le checkpoint "latest"
        latest_path = self.config.checkpoint_dir / "ddpm_pretrained_latest.pt"
        if latest_path.exists():
            return latest_path
        
        # Sinon, chercher par numéro d'époque
        checkpoint_files = list(self.config.checkpoint_dir.glob("ddpm_pretrained_epoch_*.pt"))
        if not checkpoint_files:
            return None
        
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return checkpoint_files[-1]

def create_transforms(image_size):
    """Transformations optimisées pour le pré-entraîné"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
    ])

def denormalize(tensor):
    """Dénormaliser les images pour visualisation"""
    return (tensor + 1) / 2  # [-1,1] -> [0,1]

def visualize_samples(trainer, epoch, num_samples=16):
    """Générer et sauvegarder des échantillons"""
    logger.info(f"🎨 Génération d'échantillons à l'époque {epoch}...")
    
    try:
        # Générer les échantillons
        samples = trainer.generate_samples(num_samples)
        
        # Dénormaliser
        samples = denormalize(samples)
        samples = torch.clamp(samples, 0, 1)
        
        # Statistiques
        sample_std = samples.std().item()
        sample_mean = samples.mean().item()
        
        logger.info(f"  📊 Mean: {sample_mean:.3f}, Std: {sample_std:.3f}")
        
        if sample_std > 0.4:
            logger.warning(f"  ⚠️  Qualité médiocre (std={sample_std:.3f})")
        else:
            logger.info(f"  ✓ Qualité acceptable (std={sample_std:.3f})")
        
        # Sauvegarder grille
        grid = make_grid(samples, nrow=4, padding=2)
        save_path = trainer.config.output_dir / f"samples_epoch_{epoch}.png"
        save_image(grid, save_path)
        
        # Sauvegarder échantillons individuels
        individual_dir = trainer.config.output_dir / f"individual_epoch_{epoch}"
        individual_dir.mkdir(exist_ok=True)
        
        for i, sample in enumerate(samples):
            individual_path = individual_dir / f"sample_{i:03d}.png"
            save_image(sample, individual_path)
        
        logger.info(f"✓ Échantillons sauvegardés: {save_path}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")

def main():
    """Fonction principale avec modèle pré-entraîné"""
    # Configuration
    config = DDPMPretrainedConfig()
    
    logger.info(f"🚀 DDPM Fine-tuning avec modèle pré-entraîné")
    logger.info(f"Device: {config.device}")
    logger.info(f"Modèle pré-entraîné: {config.pretrained_model}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    
    # Vérifications des chemins
    logger.info(f"📁 Checkpoints: {config.checkpoint_dir}")
    logger.info(f"🖼️ Samples: {config.output_dir}")
    logger.info(f"📊 Dataset: {config.data_dir}")
    
    # Dataset
    transform = create_transforms(config.image_size)
    dataset = MalignantDataset(config.malignant_csv, config.data_dir, transform=transform)
    
    if len(dataset) == 0:
        logger.error("❌ Aucune image trouvée dans le dataset!")
        return
    
    logger.info(f"✓ Dataset chargé: {len(dataset)} images malignes")
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    # Vérification des données
    sample_batch = next(iter(dataloader))
    logger.info(f"✓ Données vérifiées: shape={sample_batch.shape}, range=[{sample_batch.min():.3f}, {sample_batch.max():.3f}]")
    
    # Sauvegarder échantillons du dataset
    dataset_samples = denormalize(sample_batch[:4])
    dataset_samples = torch.clamp(dataset_samples, 0, 1)
    save_image(dataset_samples, config.output_dir / "dataset_samples.png", nrow=2)
    logger.info(f"✓ Échantillons du dataset sauvegardés")
    
    # Initialiser le trainer
    trainer = DDPMPretrainedTrainer(config)
    
    # Recherche de checkpoint existant
    start_epoch = 0
    best_loss = float('inf')
    
    latest_checkpoint = trainer.find_latest_checkpoint()
    if latest_checkpoint:
        try:
            start_epoch, last_loss = trainer.load_checkpoint(latest_checkpoint)
            best_loss = last_loss
            logger.info(f"✅ Checkpoint chargé: époque {start_epoch}, loss {last_loss:.6f}")
        except Exception as e:
            logger.warning(f"❌ Erreur checkpoint: {e}")
            start_epoch = 0
            best_loss = float('inf')
    
    # Boucle d'entraînement
    logger.info(f"🎯 Début de l'entraînement: {start_epoch} → {config.num_epochs}")
    
    for epoch in range(start_epoch, config.num_epochs):
        trainer.unet.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in progress_bar:
            loss = trainer.train_step(batch)
            
            if loss > 0:
                epoch_loss += loss
                num_batches += 1
            
            # Mise à jour de la barre de progression
            current_lr = trainer.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Loss': f'{loss:.4f}',
                'Best': f'{best_loss:.4f}',
                'LR': f'{current_lr:.2e}'
            })
        
        # Mise à jour du learning rate
        trainer.lr_scheduler.step()
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            
            # Vérifier si c'est le meilleur modèle
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                logger.info(f"🏆 Nouveau meilleur modèle! Loss: {avg_loss:.6f}")
        
        # Sauvegardes périodiques
        if (epoch + 1) % 25 == 0 or is_best:
            trainer.save_checkpoint(epoch + 1, avg_loss, is_best=is_best)
            logger.info(f"💾 Checkpoint sauvegardé à l'époque {epoch + 1}")
        
        # Génération d'échantillons
        if (epoch + 1) % 25 == 0:
            visualize_samples(trainer, epoch + 1, num_samples=16)
    
    # Génération finale
    logger.info("🎨 Génération d'échantillons finaux...")
    try:
        # Différentes tailles d'échantillons
        for num_samples in [4, 16, 64]:
            samples = trainer.generate_samples(num_samples)
            samples = denormalize(samples)
            samples = torch.clamp(samples, 0, 1)
            
            # Grille
            nrow = 2 if num_samples == 4 else (4 if num_samples == 16 else 8)
            grid = make_grid(samples, nrow=nrow, padding=2)
            save_path = config.output_dir / f"final_generation_{num_samples}.png"
            save_image(grid, save_path)
            
            logger.info(f"✓ {num_samples} échantillons finaux sauvegardés")
        
    except Exception as e:
        logger.error(f"Erreur génération finale: {e}")
    
    logger.info("🏁 Entraînement terminé!")
    logger.info(f"📊 Meilleure loss: {best_loss:.6f}")
    logger.info(f"📁 Checkpoints: {config.checkpoint_dir}")
    logger.info(f"🖼️ Échantillons: {config.output_dir}")

if __name__ == "__main__":
    main()
