"""
DDPM-Based Benign Data Augmentation for VAE Training
=====================================================

Synergie 3: Si vous manquez de données bénignes variées, utilisez le DDPM
(entraîné sur les classes bénignes) pour générer encore plus de données
d'entraînement saines pour le VAE.

Cette approche rend le VAE encore plus robuste à la diversité normale de la peau.

Workflow:
    1. Utiliser le DDPM existant (generators/ddpm/) pour générer des images bénignes
    2. Filtrer les images de haute qualité
    3. Les ajouter au dataset d'entraînement du VAE
    4. Entraîner le VAE sur ce dataset enrichi

Usage:
    # Générer 1000 images bénignes supplémentaires avec DDPM
    python ddpm_benign_augmentation.py \
        --ddpm_model_path ../generators/ddpm/checkpoints/best_model.pth \
        --num_samples 1000 \
        --output_dir ./benign_augmented
    
    # Entraîner le VAE sur le dataset enrichi
    python train_vae.py \
        --img_dir ./benign_augmented \
        --epochs 100
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import json

# Ajouter le chemin vers les générateurs
sys.path.insert(0, str(Path(__file__).parent.parent / 'generators' / 'ddpm'))

try:
    from DDPM_model import DDPMPretrainedTrainer, DDPMPretrainedConfig
    from DDPM_sampling import DDPMSampler
    DDPM_AVAILABLE = True
except ImportError:
    print("Warning: DDPM modules not found. This script requires the DDPM generator.")
    DDPM_AVAILABLE = False

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenignQualityFilter:
    """
    Filtre de qualité pour les images générées par DDPM
    
    Rejette les images de mauvaise qualité qui pourraient nuire
    à l'entraînement du VAE.
    """
    
    def __init__(self):
        self.checks = {
            'brightness': self._check_brightness,
            'contrast': self._check_contrast,
            'sharpness': self._check_sharpness,
            'artifacts': self._check_artifacts
        }
    
    def _check_brightness(self, image: np.ndarray) -> bool:
        """Vérifie que l'image n'est ni trop sombre ni trop claire"""
        mean_brightness = image.mean()
        return 0.2 < mean_brightness < 0.9
    
    def _check_contrast(self, image: np.ndarray) -> bool:
        """Vérifie un contraste suffisant"""
        std_contrast = image.std()
        return std_contrast > 0.05
    
    def _check_sharpness(self, image: np.ndarray) -> bool:
        """Vérifie la netteté (pas trop floue)"""
        # Gradient Laplacien
        from scipy import ndimage
        gray = image.mean(axis=2)
        laplacian = ndimage.laplace(gray)
        sharpness = laplacian.var()
        return sharpness > 0.001
    
    def _check_artifacts(self, image: np.ndarray) -> bool:
        """Vérifie l'absence d'artefacts évidents (pixels noirs/blancs extrêmes)"""
        # Pas trop de pixels complètement noirs ou blancs
        black_pixels = np.sum(image < 0.05) / image.size
        white_pixels = np.sum(image > 0.95) / image.size
        return black_pixels < 0.1 and white_pixels < 0.1
    
    def is_good_quality(self, image: np.ndarray, verbose: bool = False) -> bool:
        """
        Vérifie si l'image passe tous les critères de qualité
        
        Args:
            image: Image numpy [H, W, C] dans [0, 1]
            verbose: Afficher les résultats de chaque check
            
        Returns:
            bool: True si l'image est de bonne qualité
        """
        results = {}
        for name, check_func in self.checks.items():
            try:
                passed = check_func(image)
                results[name] = passed
                if verbose and not passed:
                    logger.debug(f"Quality check failed: {name}")
            except Exception as e:
                logger.warning(f"Error in {name} check: {e}")
                results[name] = False
        
        return all(results.values())


class DDPMBenignAugmenter:
    """
    Générateur de données bénignes augmentées avec DDPM
    """
    
    def __init__(
        self,
        ddpm_model_path: str = None,
        device: torch.device = None,
        quality_filter: bool = True
    ):
        """
        Initialize the augmenter
        
        Args:
            ddpm_model_path: Chemin vers le modèle DDPM entraîné sur bénins
            device: Device (GPU/CPU)
            quality_filter: Activer le filtre de qualité
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.quality_filter_enabled = quality_filter
        
        if quality_filter:
            self.quality_filter = BenignQualityFilter()
        
        # Charger le DDPM
        if ddpm_model_path and DDPM_AVAILABLE:
            self._load_ddpm(ddpm_model_path)
        else:
            self.ddpm_sampler = None
            logger.warning("DDPM model not loaded")
    
    def _load_ddpm(self, model_path: str):
        """Charge le modèle DDPM"""
        try:
            logger.info(f"Loading DDPM model from {model_path}")
            
            # Charger le config et le modèle
            config = DDPMPretrainedConfig()
            
            # Créer le sampler
            self.ddpm_sampler = DDPMSampler(config)
            
            # Charger les poids
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.ddpm_sampler.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.ddpm_sampler.model.load_state_dict(checkpoint)
            
            self.ddpm_sampler.model.eval()
            logger.info("DDPM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load DDPM model: {e}")
            self.ddpm_sampler = None
    
    def generate_benign_samples(
        self,
        num_samples: int,
        batch_size: int = 16,
        output_dir: str = './benign_augmented',
        save_metadata: bool = True
    ) -> dict:
        """
        Génère des échantillons bénigns avec DDPM
        
        Args:
            num_samples: Nombre d'images à générer
            batch_size: Taille des batchs pour la génération
            output_dir: Répertoire de sortie
            save_metadata: Sauvegarder les métadonnées
            
        Returns:
            stats: Statistiques de génération
        """
        if self.ddpm_sampler is None:
            logger.error("DDPM sampler not available")
            return {}
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'total_generated': 0,
            'accepted': 0,
            'rejected': 0,
            'rejection_rate': 0.0
        }
        
        accepted_images = []
        image_id = 0
        
        logger.info(f"Generating {num_samples} benign samples...")
        
        # Générer par batches
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
                current_batch_size = min(batch_size, num_samples - stats['total_generated'])
                
                # Générer avec DDPM
                samples = self.ddpm_sampler.sample(
                    num_samples=current_batch_size,
                    device=self.device
                )
                
                stats['total_generated'] += current_batch_size
                
                # Convertir en numpy
                samples_np = samples.cpu().permute(0, 2, 3, 1).numpy()
                samples_np = np.clip(samples_np, 0, 1)
                
                # Filtrer et sauvegarder
                for i in range(current_batch_size):
                    img_np = samples_np[i]
                    
                    # Vérifier la qualité
                    if self.quality_filter_enabled:
                        if not self.quality_filter.is_good_quality(img_np):
                            stats['rejected'] += 1
                            continue
                    
                    # Sauvegarder l'image
                    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                    img_path = output_dir / f'benign_ddpm_{image_id:06d}.png'
                    img_pil.save(img_path)
                    
                    accepted_images.append(str(img_path))
                    stats['accepted'] += 1
                    image_id += 1
        
        stats['rejection_rate'] = stats['rejected'] / stats['total_generated'] if stats['total_generated'] > 0 else 0
        
        logger.info(f"\nGeneration complete:")
        logger.info(f"  Total generated: {stats['total_generated']}")
        logger.info(f"  Accepted: {stats['accepted']}")
        logger.info(f"  Rejected: {stats['rejected']} ({stats['rejection_rate']*100:.1f}%)")
        
        # Sauvegarder les métadonnées
        if save_metadata:
            metadata = {
                'num_samples_requested': num_samples,
                'num_samples_accepted': stats['accepted'],
                'quality_filter_enabled': self.quality_filter_enabled,
                'image_paths': accepted_images,
                'stats': stats
            }
            
            metadata_path = output_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata saved to {metadata_path}")
        
        return stats
    
    def combine_with_real_data(
        self,
        real_data_dir: str,
        synthetic_data_dir: str,
        output_dir: str,
        synthetic_ratio: float = 0.3
    ):
        """
        Combine les données réelles avec les données synthétiques
        
        Args:
            real_data_dir: Répertoire des vraies images bénignes
            synthetic_data_dir: Répertoire des images synthétiques DDPM
            output_dir: Répertoire de sortie
            synthetic_ratio: Ratio de données synthétiques (0.3 = 30%)
        """
        import shutil
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copier les vraies images
        real_dir = Path(real_data_dir)
        real_images = list(real_dir.glob('*.jpg')) + list(real_dir.glob('*.png'))
        
        logger.info(f"Found {len(real_images)} real images")
        
        for img_path in tqdm(real_images, desc="Copying real images"):
            shutil.copy(img_path, output_dir / img_path.name)
        
        # Ajouter les images synthétiques
        synthetic_dir = Path(synthetic_data_dir)
        synthetic_images = list(synthetic_dir.glob('*.jpg')) + list(synthetic_dir.glob('*.png'))
        
        num_synthetic_to_add = int(len(real_images) * synthetic_ratio)
        synthetic_selected = np.random.choice(
            synthetic_images,
            size=min(num_synthetic_to_add, len(synthetic_images)),
            replace=False
        )
        
        logger.info(f"Adding {len(synthetic_selected)} synthetic images ({synthetic_ratio*100:.1f}%)")
        
        for img_path in tqdm(synthetic_selected, desc="Adding synthetic images"):
            shutil.copy(img_path, output_dir / img_path.name)
        
        total = len(real_images) + len(synthetic_selected)
        logger.info(f"\nCombined dataset created with {total} images")
        logger.info(f"  Real: {len(real_images)} ({len(real_images)/total*100:.1f}%)")
        logger.info(f"  Synthetic: {len(synthetic_selected)} ({len(synthetic_selected)/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate benign images with DDPM for VAE training augmentation'
    )
    
    # DDPM model
    parser.add_argument('--ddpm_model_path', type=str, required=True,
                       help='Path to trained DDPM model')
    
    # Generation
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for generation')
    parser.add_argument('--quality_filter', action='store_true', default=True,
                       help='Enable quality filtering')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./benign_augmented',
                       help='Output directory for generated images')
    
    # Optional: combine with real data
    parser.add_argument('--real_data_dir', type=str, default=None,
                       help='Directory with real benign images (optional)')
    parser.add_argument('--synthetic_ratio', type=float, default=0.3,
                       help='Ratio of synthetic to real data (0.3 = 30%)')
    parser.add_argument('--combined_output_dir', type=str, default='./benign_combined',
                       help='Output for combined dataset')
    
    args = parser.parse_args()
    
    if not DDPM_AVAILABLE:
        logger.error("DDPM modules not available. Cannot proceed.")
        return
    
    # Vérifier que le modèle existe
    if not Path(args.ddpm_model_path).exists():
        logger.error(f"DDPM model not found: {args.ddpm_model_path}")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Créer l'augmenter
    augmenter = DDPMBenignAugmenter(
        ddpm_model_path=args.ddpm_model_path,
        device=device,
        quality_filter=args.quality_filter
    )
    
    # Générer les échantillons
    stats = augmenter.generate_benign_samples(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    # Optionnel: combiner avec les vraies données
    if args.real_data_dir:
        logger.info("\nCombining with real data...")
        augmenter.combine_with_real_data(
            real_data_dir=args.real_data_dir,
            synthetic_data_dir=args.output_dir,
            output_dir=args.combined_output_dir,
            synthetic_ratio=args.synthetic_ratio
        )
        
        logger.info(f"\nNow you can train the VAE on the combined dataset:")
        logger.info(f"python train_vae.py --img_dir {args.combined_output_dir} --epochs 100")
    else:
        logger.info(f"\nGenerated images saved to: {args.output_dir}")
        logger.info(f"Train the VAE with: python train_vae.py --img_dir {args.output_dir} --epochs 100")


if __name__ == '__main__':
    main()
