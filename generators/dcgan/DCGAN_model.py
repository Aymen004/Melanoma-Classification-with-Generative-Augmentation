import torch
import torch.nn as nn
import torch.nn.functional as F

# Fonction d'initialisation des poids
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        """
        Générateur DCGAN
        Args:
            nz: taille du vecteur de bruit latent (100)
            ngf: nombre de features du générateur (64)
            nc: nombre de canaux de sortie (3 pour RGB)
        """
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        
        # Architecture du générateur
        # Input: vecteur de bruit (nz,) -> Output: image (3, 64, 64)
        self.main = nn.Sequential(
            # Couche 1: nz -> ngf*8
            # Input: (nz, 1, 1) -> Output: (ngf*8, 4, 4)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # Couche 2: ngf*8 -> ngf*4
            # Input: (ngf*8, 4, 4) -> Output: (ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # Couche 3: ngf*4 -> ngf*2
            # Input: (ngf*4, 8, 8) -> Output: (ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # Couche 4: ngf*2 -> ngf
            # Input: (ngf*2, 16, 16) -> Output: (ngf, 32, 32)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # Couche 5: ngf -> nc (sortie finale)
            # Input: (ngf, 32, 32) -> Output: (nc, 64, 64)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # Sortie entre [-1, 1]
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        """
        Discriminateur DCGAN
        Args:
            nc: nombre de canaux d'entrée (3 pour RGB)
            ndf: nombre de features du discriminateur (64)
        """
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        
        # Architecture du discriminateur
        # Input: image (3, 64, 64) -> Output: probabilité [0, 1]
        self.main = nn.Sequential(
            # Couche 1: nc -> ndf
            # Input: (nc, 64, 64) -> Output: (ndf, 32, 32)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Couche 2: ndf -> ndf*2
            # Input: (ndf, 32, 32) -> Output: (ndf*2, 16, 16)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Couche 3: ndf*2 -> ndf*4
            # Input: (ndf*2, 16, 16) -> Output: (ndf*4, 8, 8)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Couche 4: ndf*4 -> ndf*8
            # Input: (ndf*4, 8, 8) -> Output: (ndf*8, 4, 4)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Couche 5: ndf*8 -> 1 (sortie finale)
            # Input: (ndf*8, 4, 4) -> Output: (1, 1, 1)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
            # Pas de Sigmoid car BCEWithLogitsLoss l'applique automatiquement
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

def create_dcgan_models(nz=100, ngf=64, ndf=64, nc=3, device='cuda'):
    """
    Créer et initialiser les modèles Generator et Discriminator
    
    Args:
        nz: taille du vecteur de bruit latent
        ngf: nombre de features du générateur
        ndf: nombre de features du discriminateur
        nc: nombre de canaux (3 pour RGB)
        device: device (cuda ou cpu)
    
    Returns:
        generator, discriminator
    """
    # Créer les modèles
    generator = Generator(nz, ngf, nc).to(device)
    discriminator = Discriminator(nc, ndf).to(device)
    
    # Initialiser les poids
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    return generator, discriminator

def print_model_info(generator, discriminator):
    """
    Afficher des informations sur les modèles
    """
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=== MODÈLES DCGAN ===")
    print(f"Generator:")
    print(f"  - Paramètres: {count_parameters(generator):,}")
    print(f"  - Architecture: {len(list(generator.modules()))} couches")
    
    print(f"\nDiscriminator:")
    print(f"  - Paramètres: {count_parameters(discriminator):,}")
    print(f"  - Architecture: {len(list(discriminator.modules()))} couches")
    
    print(f"\nTotal paramètres: {count_parameters(generator) + count_parameters(discriminator):,}")

# Test des modèles
if __name__ == "__main__":
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nz = 100  # Taille du vecteur de bruit
    batch_size = 32
    
    print(f"Device: {device}")
    
    # Créer les modèles
    generator, discriminator = create_dcgan_models(device=device)
    
    # Afficher les informations
    print_model_info(generator, discriminator)
    
    # Test du générateur
    print("\n=== TEST GÉNÉRATEUR ===")
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake_images = generator(noise)
    print(f"Input noise shape: {noise.shape}")
    print(f"Generated images shape: {fake_images.shape}")
    print(f"Generated images range: [{fake_images.min():.3f}, {fake_images.max():.3f}]")
    
    # Test du discriminateur
    print("\n=== TEST DISCRIMINATEUR ===")
    real_images = torch.randn(batch_size, 3, 64, 64, device=device)
    real_output = discriminator(real_images)
    fake_output = discriminator(fake_images.detach())
    
    print(f"Real images shape: {real_images.shape}")
    print(f"Real output shape: {real_output.shape}")
    print(f"Real output range: [{real_output.min():.3f}, {real_output.max():.3f}]")
    print(f"Fake output range: [{fake_output.min():.3f}, {fake_output.max():.3f}]")
    
    print("\n✅ Modèles DCGAN créés avec succès!")