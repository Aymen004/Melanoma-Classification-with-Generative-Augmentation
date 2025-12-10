import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging
import os
import shutil
from diffusers import DDPMPipeline, UNet2DModel, DDPMScheduler

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DDPMGenerator128:
    """Générateur DDMP pour images 128x128 à partir du modèle pré-entraîné - CORRIGÉ"""
    
    def __init__(self, checkpoint_dir, output_dir="./generated_malignant_128"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Créer le dossier de sortie
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paramètres du modèle
        self.image_size = 128  # Taille cible
        self.original_size = 64  # Taille du modèle entraîné
        
        logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        logger.info(f"📁 Checkpoints: {self.checkpoint_dir}")
        logger.info(f"🖼️ Sortie: {self.output_dir}")
    
    def find_best_checkpoint(self):
        """Trouver le meilleur checkpoint disponible"""
        logger.info("🔍 Recherche du meilleur checkpoint...")
        
        # Priorité 1: Checkpoint "best"
        best_files = [
            "ddpm_pretrained_best.pt",
            "ddpm_best.pt", 
            "ddmp_best.pt"
        ]
        
        for best_file in best_files:
            best_path = self.checkpoint_dir / best_file
            if best_path.exists():
                logger.info(f"✅ Meilleur checkpoint trouvé: {best_file}")
                return best_path
        
        # Priorité 2: Checkpoint "latest"
        latest_files = [
            "ddpm_pretrained_latest.pt",
            "ddpm_latest.pt"
        ]
        
        for latest_file in latest_files:
            latest_path = self.checkpoint_dir / latest_file
            if latest_path.exists():
                logger.info(f"✅ Checkpoint latest trouvé: {latest_file}")
                return latest_path
        
        # Priorité 3: Checkpoint avec le numéro d'époque le plus élevé
        pretrained_files = list(self.checkpoint_dir.glob("ddpm_pretrained_epoch_*.pt"))
        if pretrained_files:
            pretrained_files.sort(key=lambda x: int(x.stem.split('_')[-1]), reverse=True)
            logger.info(f"✅ Checkpoint pré-entraîné le plus récent: {pretrained_files[0].name}")
            return pretrained_files[0]
        
        # Sinon, chercher les checkpoints normaux
        normal_files = list(self.checkpoint_dir.glob("ddpm_epoch_*.pt"))
        if normal_files:
            normal_files.sort(key=lambda x: int(x.stem.split('_')[-1]), reverse=True)
            logger.info(f"✅ Checkpoint normal le plus récent: {normal_files[0].name}")
            return normal_files[0]
        
        logger.error("❌ Aucun checkpoint trouvé!")
        return None
    
    def load_model(self, checkpoint_path):
        """Charger le modèle DDPM depuis un checkpoint"""
        logger.info(f"📥 Chargement du modèle depuis: {checkpoint_path.name}")
        
        try:
            # Charger le checkpoint avec compatibilité
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                logger.info("✓ Checkpoint chargé avec weights_only=False")
            except Exception as e:
                logger.warning(f"Tentative avec weights_only=True: {e}")
                
                # Fallback avec safe_globals pour PyTorch 2.6+
                import torch.serialization
                from pathlib import WindowsPath, PosixPath
                
                safe_globals = [
                    WindowsPath, 
                    PosixPath,
                    torch.torch_version.TorchVersion,
                    torch.Size,
                    torch.dtype,
                    torch.device,
                    torch.Tensor,
                ]
                
                with torch.serialization.safe_globals(safe_globals):
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                    logger.info("✓ Checkpoint chargé avec safe_globals")
            
            # Vérifier si c'est un modèle pré-entraîné (diffusers) ou custom
            if 'pretrained_model' in checkpoint:
                logger.info("🤖 Modèle pré-entraîné détecté")
                return self._load_pretrained_model(checkpoint)
            else:
                logger.info("🤖 Modèle custom détecté")
                return self._load_custom_model(checkpoint)
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            return None, None
    
    def _load_pretrained_model(self, checkpoint):
        """Charger un modèle pré-entraîné (diffusers)"""
        try:
            # Créer le modèle UNet
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
            
            # Charger les poids
            self.unet.load_state_dict(checkpoint['unet_state_dict'])
            
            # Créer le scheduler
            self.scheduler = DDPMScheduler(num_train_timesteps=1000)
            
            # Mode évaluation
            self.unet.eval()
            
            logger.info(f"✅ Modèle pré-entraîné chargé (époque {checkpoint.get('epoch', '?')})")
            
            return self.unet, self.scheduler
            
        except Exception as e:
            logger.error(f"❌ Erreur modèle pré-entraîné: {e}")
            return None, None
    
    def _load_custom_model(self, checkpoint):
        """Charger un modèle custom"""
        try:
            logger.warning("⚠️ Modèle custom détecté - utilisation d'une architecture par défaut")
            
            # Architecture par défaut compatible
            self.unet = UNet2DModel(
                sample_size=64,
                in_channels=3,
                out_channels=3,
                layers_per_block=2,
                block_out_channels=(64, 128, 256, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D", 
                    "DownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            ).to(self.device)
            
            # Essayer de charger les poids
            if 'model_state_dict' in checkpoint:
                self.unet.load_state_dict(checkpoint['model_state_dict'])
            elif 'unet_state_dict' in checkpoint:
                self.unet.load_state_dict(checkpoint['unet_state_dict'])
            else:
                # Essayer de charger directement
                self.unet.load_state_dict(checkpoint)
            
            # Scheduler par défaut
            self.scheduler = DDPMScheduler(num_train_timesteps=1000)
            
            # Mode évaluation
            self.unet.eval()
            
            logger.info(f"✅ Modèle custom chargé")
            
            return self.unet, self.scheduler
            
        except Exception as e:
            logger.error(f"❌ Erreur modèle custom: {e}")
            return None, None
    
    def setup_upsampler(self):
        """Configurer l'upsampler pour passer de 64x64 à 128x128"""
        self.upsampler = nn.Upsample(
            size=(self.image_size, self.image_size), 
            mode='bilinear', 
            align_corners=False
        ).to(self.device)
        
        logger.info(f"✅ Upsampler configuré: {self.original_size}x{self.original_size} → {self.image_size}x{self.image_size}")
    
    @torch.no_grad()
    def generate_batch(self, batch_size=8, num_inference_steps=50):
        """Générer un batch d'images"""
        self.unet.eval()
        
        # Commencer avec du bruit pur
        shape = (batch_size, 3, self.original_size, self.original_size)
        image = torch.randn(shape, device=self.device)
        
        # Processus de débruitage
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.scheduler.timesteps:
            # Prédire le bruit
            noise_pred = self.unet(image, t).sample
            
            # Débruiter
            image = self.scheduler.step(noise_pred, t, image).prev_sample
        
        # Upsampler vers 128x128
        if hasattr(self, 'upsampler'):
            image = self.upsampler(image)
        
        return image
    
    def denormalize(self, tensor):
        """Dénormaliser les images [-1,1] → [0,1]"""
        return (tensor + 1) / 2
    
    def save_image_pil(self, tensor, path, quality=95):
        """Sauvegarder une image avec PIL pour contrôler la qualité - CORRECTION"""
        # Convertir le tensor en PIL Image
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Assurer que les valeurs sont dans [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convertir en numpy
        np_img = tensor.cpu().numpy()
        np_img = (np_img * 255).astype(np.uint8)
        
        # Réorganiser les dimensions (C, H, W) -> (H, W, C)
        if np_img.shape[0] == 3:
            np_img = np_img.transpose(1, 2, 0)
        
        # Convertir en PIL et sauvegarder
        pil_img = Image.fromarray(np_img)
        pil_img.save(path, quality=quality, optimize=True)
    
    def generate_images(self, total_images=1000, batch_size=8, num_inference_steps=50):
        """Générer le nombre total d'images demandé - CORRIGÉ"""
        
        logger.info(f"🎨 Génération de {total_images} images {self.image_size}x{self.image_size}")
        logger.info(f"📦 Batch size: {batch_size}")
        logger.info(f"🔄 Étapes de débruitage: {num_inference_steps}")
        
        # Calculer le nombre de batches
        num_batches = (total_images + batch_size - 1) // batch_size
        
        # Créer les dossiers de sortie
        individual_dir = self.output_dir / "individual"
        grids_dir = self.output_dir / "grids"
        individual_dir.mkdir(exist_ok=True)
        grids_dir.mkdir(exist_ok=True)
        
        generated_count = 0
        all_images = []
        
        # Génération par batches
        for batch_idx in tqdm(range(num_batches), desc="Génération d'images"):
            try:
                # Calculer la taille du batch actuel
                remaining = total_images - generated_count
                current_batch_size = min(batch_size, remaining)
                
                if current_batch_size <= 0:
                    break
                
                # Générer le batch
                batch_images = self.generate_batch(current_batch_size, num_inference_steps)
                
                # Dénormaliser
                batch_images = self.denormalize(batch_images)
                batch_images = torch.clamp(batch_images, 0, 1)
                
                # Sauvegarder individuellement - CORRECTION: Utiliser save_image au lieu de save_image_pil
                for i, img in enumerate(batch_images):
                    img_idx = generated_count + i + 1
                    
                    # Vérifier la taille
                    if img.shape[-1] != self.image_size or img.shape[-2] != self.image_size:
                        logger.warning(f"⚠️ Taille incorrecte: {img.shape}, redimensionnement...")
                        img = F.interpolate(
                            img.unsqueeze(0), 
                            size=(self.image_size, self.image_size), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)
                    
                    # CORRECTION: Utiliser save_image sans le paramètre quality
                    img_path = individual_dir / f"malignant_ddpm_{img_idx:05d}.png"
                    save_image(img, img_path)
                    
                    # ALTERNATIVE: Utiliser PIL pour qualité élevée
                    # self.save_image_pil(img, img_path, quality=95)
                
                # Ajouter à la collection pour les grilles
                all_images.extend(batch_images.cpu())
                generated_count += current_batch_size
                
                # Nettoyer la mémoire périodiquement
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
                
                # Statistiques
                if batch_idx % 10 == 0:
                    batch_mean = batch_images.mean().item()
                    batch_std = batch_images.std().item()
                    logger.info(f"  📊 Batch {batch_idx}: mean={batch_mean:.3f}, std={batch_std:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Erreur batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Créer les grilles d'aperçu
        self._create_preview_grids(all_images[:64], grids_dir)
        
        # Statistiques finales
        logger.info(f"✅ Génération terminée!")
        logger.info(f"📊 {generated_count} images générées")
        logger.info(f"📁 Images individuelles: {individual_dir}")
        logger.info(f"🖼️ Grilles d'aperçu: {grids_dir}")
        
        return generated_count
    
    def _create_preview_grids(self, images, grids_dir):
        """Créer des grilles d'aperçu - CORRIGÉ"""
        if not images:
            return
        
        logger.info("🖼️ Création des grilles d'aperçu...")
        
        # Convertir en tenseur si nécessaire
        if isinstance(images, list):
            images = torch.stack(images[:64])
        
        # Différentes tailles de grilles
        grid_sizes = [16, 36, 64]
        
        for grid_size in grid_sizes:
            if len(images) >= grid_size:
                grid_images = images[:grid_size]
                nrow = int(np.sqrt(grid_size))
                
                # CORRECTION: Créer la grille sans paramètre quality
                grid = make_grid(grid_images, nrow=nrow, padding=2, normalize=False)
                
                # Sauvegarder
                grid_path = grids_dir / f"preview_grid_{grid_size}.png"
                save_image(grid, grid_path)  # CORRECTION: Pas de paramètre quality
                
                logger.info(f"  ✓ Grille {grid_size} sauvegardée")
    
    def run_generation(self, total_images=1000, batch_size=8, num_inference_steps=50):
        """Pipeline complet de génération"""
        
        print("🏥 GÉNÉRATEUR DDPM 128x128")
        print("=" * 50)
        
        try:
            # Étape 1: Trouver le meilleur checkpoint
            best_checkpoint = self.find_best_checkpoint()
            if best_checkpoint is None:
                logger.error("❌ Aucun checkpoint trouvé!")
                return False
            
            # Étape 2: Charger le modèle
            unet, scheduler = self.load_model(best_checkpoint)
            if unet is None:
                logger.error("❌ Impossible de charger le modèle!")
                return False
            
            # Étape 3: Configurer l'upsampler
            self.setup_upsampler()
            
            # Étape 4: Générer les images
            generated_count = self.generate_images(
                total_images=total_images,
                batch_size=batch_size, 
                num_inference_steps=num_inference_steps
            )
            
            logger.info(f"🎉 Génération réussie: {generated_count}/{total_images} images")
            return generated_count > 0  # CORRECTION: Retourner True seulement si des images ont été générées
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Nettoyer la mémoire
            if hasattr(self, 'unet'):
                del self.unet
            if hasattr(self, 'scheduler'):
                del self.scheduler
            if hasattr(self, 'upsampler'):
                del self.upsampler
            torch.cuda.empty_cache()

def main():
    """Fonction principale"""
    
    # Configuration
    CHECKPOINT_DIR = "./DDPM_pretrained/checkpoints"
    OUTPUT_DIR = "./generated_malignant_ddpm_128"
    TOTAL_IMAGES = 5900
    BATCH_SIZE = 4  # CORRECTION: Réduire pour éviter les erreurs mémoire
    INFERENCE_STEPS = 50
    
    print("🔍 Configuration:")
    print(f"📁 Dossier checkpoints: {CHECKPOINT_DIR}")
    print(f"📁 Dossier sortie: {OUTPUT_DIR}")
    print(f"🖼️ Nombre d'images: {TOTAL_IMAGES}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔄 Étapes d'inférence: {INFERENCE_STEPS}")
    print(f"📐 Taille finale: 128x128")
    
    # Créer le générateur
    generator = DDPMGenerator128(CHECKPOINT_DIR, OUTPUT_DIR)
    
    # Lancer la génération
    success = generator.run_generation(
        total_images=TOTAL_IMAGES,
        batch_size=BATCH_SIZE,
        num_inference_steps=INFERENCE_STEPS
    )
    
    if success:
        print("\n🎉 GÉNÉRATION TERMINÉE AVEC SUCCÈS!")
        print(f"📁 Vérifiez le dossier: {OUTPUT_DIR}")
        print(f"🖼️ Images individuelles: {OUTPUT_DIR}/individual/")
        print(f"🎯 Grilles d'aperçu: {OUTPUT_DIR}/grids/")
    else:
        print("\n❌ ÉCHEC DE LA GÉNÉRATION!")

if __name__ == "__main__":
    main()