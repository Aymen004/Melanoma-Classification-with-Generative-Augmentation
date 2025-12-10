import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torch.amp import autocast, GradScaler

# Importer les modèles
from DCGAN_model import Generator, Discriminator, weights_init

# Dataset optimisé pour GPU
class DCGANDataset(Dataset):
    def __init__(self, csv_file, image_dir, image_size=64):
        self.labels_df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] pour GAN
        ])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0] + ".jpg"
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image

def setup_gpu_optimization():
    """Configuration optimale pour RTX 3070"""
    if torch.cuda.is_available():
        # Optimisations CUDA
        torch.backends.cudnn.benchmark = True  # Optimise les convolutions
        torch.backends.cudnn.deterministic = False  # Pour la performance
        
        # Informations GPU
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"🚀 GPU détecté: {gpu_name}")
        print(f"💾 Mémoire GPU: {gpu_memory:.1f} GB")
        print(f"🔥 Optimisations CUDA activées")
        
        return device
    else:
        print("❌ CUDA non disponible")
        return torch.device('cpu')

def save_images(fake_images, epoch, output_dir="generated_images", nrow=8):
    """Sauvegarder les images générées"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Dénormaliser les images de [-1, 1] vers [0, 1]
    fake_images = (fake_images + 1) / 2
    
    # Sauvegarder la grille d'images
    filename = os.path.join(output_dir, f"fake_images_epoch_{epoch:03d}.png")
    vutils.save_image(fake_images, filename, nrow=nrow, normalize=True)
    
    return filename

def plot_losses(g_losses, d_losses, save_path="training_losses.png"):
    """Tracer les courbes de loss"""
    plt.figure(figsize=(12, 6))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="Generator", alpha=0.8)
    plt.plot(d_losses, label="Discriminator", alpha=0.8)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_dcgan():
    # Hyperparamètres optimisés pour RTX 3070
    lr = 0.0002         # Learning rate
    beta1 = 0.5         # Beta1 pour Adam optimizer
    nz = 100            # Taille du vecteur de bruit latent
    ngf = 64            # Features du générateur
    ndf = 64            # Features du discriminateur
    nc = 3              # Nombre de canaux (RGB)
    num_epochs = 10000  # Plus d'époques avec RTX 3070
    batch_size = 64     # Batch size plus élevé pour RTX 3070
    
    # Configuration GPU optimisée
    device = setup_gpu_optimization()
    
    # Chemins des données
    image_dir = "train/ISBI2016_ISIC_Part3_Training_Data"
    target_csv = "malignant_images.csv"  # Classe minoritaire à augmenter
    
    # Vérifier que les fichiers existent
    if not os.path.exists(target_csv):
        print(f"❌ Erreur: {target_csv} n'existe pas. Exécutez d'abord explore_data.ipynb")
        return
    
    if not os.path.exists(image_dir):
        print(f"❌ Erreur: {image_dir} n'existe pas. Vérifiez le chemin.")
        return
    
    # Créer le dataset et dataloader optimisé
    print("📊 Chargement des données...")
    dataset = DCGANDataset(target_csv, image_dir, image_size=64)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,  # Plus de workers pour RTX 3070
        pin_memory=True,  # Optimisation GPU
        persistent_workers=True  # Garde les workers en mémoire
    )
    
    print(f"Nombre d'images d'entraînement: {len(dataset)}")
    print(f"Nombre de batches: {len(dataloader)}")
    print(f"Batch size: {batch_size}")
    
    # Créer les modèles
    print("🏗️ Création des modèles...")
    generator = Generator(nz, ngf, nc).to(device)
    discriminator = Discriminator(nc, ndf).to(device)
    
    # Initialiser les poids
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Compilation des modèles pour RTX 3070 (PyTorch 2.0+)
    # Désactivé temporairement car Triton n'est pas disponible sur Windows Python 3.13
    # try:
    #     generator = torch.compile(generator)
    #     discriminator = torch.compile(discriminator)
    #     print("✅ Modèles compilés avec torch.compile")
    # except:
    #     print("⚠️ torch.compile non disponible, utilisation normale")
    print("⚠️ torch.compile désactivé (Triton non disponible sur Windows Python 3.13)")
    
    # Fonctions de loss (BCEWithLogitsLoss est compatible avec autocast)
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizers avec paramètres ajustés
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999), eps=1e-8)
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999), eps=1e-8)
    
    # Schedulers pour ajuster le learning rate
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=100, gamma=0.9)
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=100, gamma=0.9)
    
    # Mixed Precision pour RTX 3070
    scaler = GradScaler('cuda')
    
    # Labels pour l'entraînement
    real_label = 1.0
    fake_label = 0.0
    
    # Vecteur de bruit fixe pour visualisation
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    
    # Listes pour stocker les losses et métriques
    g_losses = []
    d_losses = []
    d_real_acc = []
    d_fake_acc = []
    
    # Variables pour le timing
    start_time = time.time()
    
    print("🚀 Début de l'entraînement optimisé RTX 3070...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        for i, data in enumerate(dataloader, 0):
            
            ############################
            # (1) Mise à jour du Discriminateur
            ############################
            discriminator.zero_grad()
            
            # Entraîner avec des vraies images
            real_data = data.to(device, non_blocking=True)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            
            with autocast('cuda'):
                output = discriminator(real_data).view(-1)
                err_d_real = criterion(output, label)
            
            scaler.scale(err_d_real).backward()
            d_x = output.mean().item()
            d_real_accuracy = (output > 0.5).float().mean().item()
            
            # Entraîner avec des fausses images
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            
            with autocast('cuda'):
                fake = generator(noise)
                label.fill_(fake_label)
                output = discriminator(fake.detach()).view(-1)
                err_d_fake = criterion(output, label)
            
            scaler.scale(err_d_fake).backward()
            d_g_z1 = output.mean().item()
            d_fake_accuracy = (output < 0.5).float().mean().item()
            
            err_d = err_d_real + err_d_fake
            scaler.step(optimizer_d)
            
            ############################
            # (2) Mise à jour du Générateur
            ############################
            generator.zero_grad()
            label.fill_(real_label)
            
            with autocast('cuda'):
                output = discriminator(fake).view(-1)
                err_g = criterion(output, label)
            
            scaler.scale(err_g).backward()
            d_g_z2 = output.mean().item()
            scaler.step(optimizer_g)
            
            # Mise à jour du scaler
            scaler.update()
            
            # Sauvegarder les métriques
            g_losses.append(err_g.item())
            d_losses.append(err_d.item())
            d_real_acc.append(d_real_accuracy)
            d_fake_acc.append(d_fake_accuracy)
            
            # Affichage des statistiques détaillées
            if i % 25 == 0:
                elapsed = time.time() - start_time
                images_per_sec = ((epoch * len(dataloader) + i + 1) * batch_size) / elapsed
                
                print(f'[{epoch:3d}/{num_epochs}][{i:3d}/{len(dataloader)}] '
                      f'Loss_D: {err_d.item():.4f} Loss_G: {err_g.item():.4f} '
                      f'D(x): {d_x:.3f} D(G(z)): {d_g_z1:.3f}/{d_g_z2:.3f} '
                      f'Acc_real: {d_real_accuracy:.3f} Acc_fake: {d_fake_accuracy:.3f} '
                      f'Speed: {images_per_sec:.1f} img/s')
        
        # Mise à jour des schedulers
        scheduler_d.step()
        scheduler_g.step()
        
        # Affichage du temps par époque
        epoch_time = time.time() - epoch_start_time
        print(f"Époque {epoch} terminée en {epoch_time:.1f}s")
        
        # Sauvegarder des images générées
        if epoch % 25 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
                img_path = save_images(fake, epoch)
                print(f"📸 Images sauvegardées: {img_path}")
        
        # Sauvegarder les modèles
        if epoch % 25 == 0 or epoch == num_epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'scheduler_g_state_dict': scheduler_g.state_dict(),
                'scheduler_d_state_dict': scheduler_d.state_dict(),
                'g_losses': g_losses,
                'd_losses': d_losses,
                'scaler_state_dict': scaler.state_dict(),
            }
            torch.save(checkpoint, f'dcgan_checkpoint_epoch_{epoch}.pth')
            print(f"💾 Checkpoint sauvegardé: dcgan_checkpoint_epoch_{epoch}.pth")
        
        # Monitoring de la mémoire GPU
        if epoch % 50 == 0:
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"🔍 Mémoire GPU: {memory_used:.2f}GB utilisée, {memory_cached:.2f}GB cachée")
    
    total_time = time.time() - start_time
    print(f"\n✅ Entraînement terminé en {total_time/3600:.1f}h!")
    
    # Sauvegarder les modèles finaux
    torch.save(generator.state_dict(), 'generator_final.pth')
    torch.save(discriminator.state_dict(), 'discriminator_final.pth')
    print("💾 Modèles finaux sauvegardés")
    
    # Tracer les courbes de loss et métriques
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(g_losses, label="Generator", alpha=0.8)
    plt.plot(d_losses, label="Discriminator", alpha=0.8)
    plt.title("Losses")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(d_real_acc, label="Real Images", alpha=0.8)
    plt.plot(d_fake_acc, label="Fake Images", alpha=0.8)
    plt.title("Discriminator Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Moyennes mobiles
    window = 100
    g_smooth = [np.mean(g_losses[max(0, i-window):i+1]) for i in range(len(g_losses))]
    d_smooth = [np.mean(d_losses[max(0, i-window):i+1]) for i in range(len(d_losses))]
    plt.plot(g_smooth, label="Generator (smooth)", alpha=0.8)
    plt.plot(d_smooth, label="Discriminator (smooth)", alpha=0.8)
    plt.title("Smoothed Losses")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Métriques sauvegardées: training_metrics.png")
    
    return generator, discriminator, g_losses, d_losses

if __name__ == "__main__":
    # Vérifier CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA non disponible!")
        exit(1)
    
    # Entraîner le DCGAN
    generator, discriminator, g_losses, d_losses = train_dcgan()
    
    print(f"\n📈 Résumé de l'entraînement:")
    print(f"Loss finale du générateur: {g_losses[-1]:.4f}")
    print(f"Loss finale du discriminateur: {d_losses[-1]:.4f}")
    print(f"Nombre total d'itérations: {len(g_losses):,}")
    print(f"🎯 Modèles prêts pour la génération d'images!")