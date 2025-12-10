"""
Générateur d'Images DCGAN Bénignes - Haute Qualité
==================================================
Génère des images 128×128 à partir du modèle DCGAN entraîné
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np


class Generator(nn.Module):
    """Générateur DCGAN - Architecture identique à l'entraînement"""
    
    def __init__(self, latent_dim=100, gen_features=64):
        super().__init__()
        
        ngf = gen_features
        nz = latent_dim
        
        self.main = nn.Sequential(
            # Entrée: (nz) -> (ngf*8, 4, 4)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # (ngf*8, 4, 4) -> (ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # (ngf*4, 8, 8) -> (ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # (ngf*2, 16, 16) -> (ngf, 32, 32)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # (ngf, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)


def load_generator(checkpoint_path, device='cuda'):
    """Charger le générateur depuis un checkpoint"""
    print(f"\n📂 Chargement du modèle: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Créer le générateur
    G = Generator(latent_dim=100, gen_features=64).to(device)
    
    # Charger les poids
    G.load_state_dict(checkpoint['G_state_dict'])
    G.eval()
    
    epoch = checkpoint.get('epoch', 'inconnu')
    g_loss = checkpoint.get('g_loss', 'inconnu')
    
    print(f"✅ Générateur chargé!")
    print(f"   Époque: {epoch}")
    print(f"   G_loss: {g_loss}")
    
    return G


def upscale_image(img_tensor, target_size=128):
    """
    Upscaler une image de 64×64 à 128×128 avec interpolation de haute qualité
    
    Args:
        img_tensor: Tensor (C, H, W) normalisé [-1, 1]
        target_size: Taille cible (128)
    
    Returns:
        Tensor (C, target_size, target_size) normalisé [-1, 1]
    """
    # Dénormaliser de [-1, 1] à [0, 1]
    img = (img_tensor + 1) / 2
    
    # Convertir en PIL pour upscaling de haute qualité
    img_pil = transforms.ToPILImage()(img)
    
    # Upscale avec LANCZOS (meilleure qualité)
    img_upscaled = img_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Reconvertir en tensor et renormaliser à [-1, 1]
    img_tensor = transforms.ToTensor()(img_upscaled)
    img_tensor = img_tensor * 2 - 1
    
    return img_tensor


def generate_images(
    checkpoint_path,
    output_dir,
    num_images=5000,
    batch_size=64,
    target_size=128,
    seed=None,
    device='cuda'
):
    """
    Générer des images synthétiques (ajoute seulement les images manquantes)
    
    Args:
        checkpoint_path: Chemin du checkpoint du générateur
        output_dir: Dossier de sortie
        num_images: Nombre TOTAL d'images désirées (existantes + nouvelles)
        batch_size: Taille de batch
        target_size: Taille finale des images (128)
        seed: Seed aléatoire (optionnel)
        device: Device CUDA ou CPU
    """
    
    # Créer le dossier de sortie
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Vérifier les images existantes
    existing_images = list(output_dir.glob("generated_benign_dcgan_*.png"))
    num_existing = len(existing_images)
    
    print("\n" + "=" * 60)
    print("🎨 GÉNÉRATION D'IMAGES DCGAN BÉNIGNES")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print(f"Images existantes: {num_existing}")
    print(f"Total désiré: {num_images}")
    
    # Calculer combien d'images à générer
    if num_existing >= num_images:
        print(f"\n✅ Objectif déjà atteint! ({num_existing} images présentes)")
        print("Aucune génération nécessaire.")
        return
    
    num_to_generate = num_images - num_existing
    print(f"À générer: {num_to_generate} nouvelles images")
    
    # Trouver le prochain numéro d'image
    if num_existing > 0:
        # Extraire les numéros des images existantes
        existing_numbers = []
        for img_file in existing_images:
            try:
                # Format: generated_benign_dcgan_0001.png
                num_str = img_file.stem.split('_')[-1]
                existing_numbers.append(int(num_str))
            except:
                continue
        
        start_number = max(existing_numbers) + 1 if existing_numbers else 1
    else:
        start_number = 1
    
    print(f"Numérotation à partir de: {start_number:04d}")
    print(f"Taille: {target_size}×{target_size}")
    print(f"Device: {device}")
    
    # Seed pour reproductibilité
    if seed is not None:
        torch.manual_seed(seed)
        print(f"Seed: {seed}")
    
    # Charger le générateur
    G = load_generator(checkpoint_path, device)
    
    print(f"\n🚀 Génération en cours...\n")
    
    image_counter = start_number
    num_batches = (num_to_generate + batch_size - 1) // batch_size
    images_generated = 0
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Génération"):
            # Calculer combien d'images générer dans ce batch
            remaining = num_to_generate - images_generated
            current_batch_size = min(batch_size, remaining)
            
            # Générer bruit latent
            z = torch.randn(current_batch_size, 100, device=device)
            
            # Générer images 64×64
            fake_imgs = G(z)
            
            # Upscaler à 128×128 et sauvegarder individuellement
            for i in range(current_batch_size):
                # Upscaler l'image
                img_upscaled = upscale_image(fake_imgs[i].cpu(), target_size)
                
                # Dénormaliser pour sauvegarde [0, 1]
                img_to_save = (img_upscaled + 1) / 2
                
                # Nom du fichier avec numérotation continue
                img_name = f"generated_benign_dcgan_{image_counter:04d}.png"
                img_path = output_dir / img_name
                
                # Sauvegarder
                save_image(img_to_save, img_path)
                
                image_counter += 1
                images_generated += 1
    
    print(f"\n✅ Génération terminée!")
    print(f"📁 {images_generated} nouvelles images générées")
    print(f"📊 Total dans le dossier: {num_existing + images_generated} images")
    print(f"📂 Dossier: {output_dir}")
    
    # Créer une grille d'aperçu des NOUVELLES images
    print("\n🖼️  Création d'une grille d'aperçu des nouvelles images...")
    create_preview_grid_range(output_dir, start_number, image_counter - 1, num_preview=min(64, images_generated))


def create_preview_grid_range(output_dir, start_num, end_num, num_preview=64):
    """Créer une grille d'aperçu pour une plage spécifique d'images"""
    output_dir = Path(output_dir)
    
    # Charger les images dans la plage spécifiée
    images = []
    count = 0
    
    for num in range(start_num, min(end_num + 1, start_num + num_preview)):
        img_name = f"generated_benign_dcgan_{num:04d}.png"
        img_path = output_dir / img_name
        
        if img_path.exists():
            img = Image.open(img_path)
            img_tensor = transforms.ToTensor()(img)
            images.append(img_tensor)
            count += 1
    
    if len(images) == 0:
        print("⚠️ Aucune nouvelle image trouvée pour l'aperçu")
        return
    
    # Créer la grille
    from torchvision.utils import make_grid
    
    nrow = 8 if len(images) >= 8 else len(images)
    grid = make_grid(images, nrow=nrow, padding=2)
    
    # Sauvegarder
    preview_path = output_dir / f"preview_new_images_{start_num:04d}_to_{end_num:04d}.png"
    save_image(grid, preview_path)
    
    print(f"✅ Grille d'aperçu des nouvelles images: {preview_path}")


def main():
    parser = argparse.ArgumentParser(description="Générer des images DCGAN bénignes haute qualité")
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./DCGAN_benign/checkpoints/checkpoint_best.pt',
        help='Chemin du checkpoint du générateur'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./generated_benign_dcgan',
        help='Dossier de sortie'
    )
    
    parser.add_argument(
        '--num_images',
        type=int,
        default=5000,
        help='Nombre TOTAL d\'images désirées (existantes + nouvelles)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Taille de batch pour la génération'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=128,
        help='Taille des images (128×128)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed aléatoire pour reproductibilité'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (cuda ou cpu)'
    )
    
    args = parser.parse_args()
    
    # Générer les images
    generate_images(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        num_images=args.num_images,
        batch_size=args.batch_size,
        target_size=args.size,
        seed=args.seed,
        device=args.device
    )


if __name__ == "__main__":
    main()
