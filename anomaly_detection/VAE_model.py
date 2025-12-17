"""
Variational Autoencoder (VAE) Model
====================================

Ce module définit l'architecture du VAE pour la détection d'anomalies dans les images médicales.

Architecture:
- Encoder: ConvNet -> mu, logvar (espace latent)
- Reparameterization trick: z = mu + sigma * epsilon
- Decoder: ConvNet transposé -> reconstruction

Perte: Reconstruction Loss (MSE) + KL Divergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class VAEConfig:
    """Configuration pour le modèle VAE"""
    image_size: int = 128
    in_channels: int = 3
    latent_dim: int = 256
    hidden_dims: list = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256]


class ConvVAE(nn.Module):
    """
    VAE Convolutionnel pour images médicales
    
    Architecture optimisée pour la reconstruction d'images de lésions cutanées.
    Utilise des couches convolutionnelles pour préserver la structure spatiale.
    """
    
    def __init__(self, config: VAEConfig):
        super(ConvVAE, self).__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        
        # Encoder
        modules = []
        in_channels = config.in_channels
        
        for h_dim in config.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculer la taille après convolutions
        # Pour 128x128: 128 -> 64 -> 32 -> 16 -> 8
        self.feature_size = config.image_size // (2 ** len(config.hidden_dims))
        self.flatten_size = config.hidden_dims[-1] * self.feature_size * self.feature_size
        
        # Couches pour mu et logvar
        self.fc_mu = nn.Linear(self.flatten_size, config.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, config.latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(config.latent_dim, self.flatten_size)
        
        # ✅ ÉTAPE 2: Decoder amélioré avec Upsample + Conv2d (pas de damier)
        # Architecture plus douce pour éviter l'effet mosaïque de ConvTranspose2d
        hidden_dims_reversed = config.hidden_dims[::-1]
        
        self.decoder = nn.Sequential(
            # Bloc 1: 8x8 -> 16x16 (256 -> 128 channels)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_dims_reversed[0], hidden_dims_reversed[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims_reversed[1]),
            nn.LeakyReLU(0.2),
            
            # Bloc 2: 16x16 -> 32x32 (128 -> 64 channels)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_dims_reversed[1], hidden_dims_reversed[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims_reversed[2]),
            nn.LeakyReLU(0.2),
            
            # Bloc 3: 32x32 -> 64x64 (64 -> 32 channels)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_dims_reversed[2], hidden_dims_reversed[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dims_reversed[3]),
            nn.LeakyReLU(0.2),
            
            # Bloc 4: 64x64 -> 128x128 (garde 32 channels)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_dims_reversed[3], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Sortie finale: 32 -> 3 channels (RGB)
            nn.Conv2d(32, config.in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Sortie [0, 1]
        )
        
        # Plus besoin de final_layer séparé
        self.final_layer = None
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode l'image en paramètres de distribution latente
        
        Args:
            x: Image [B, C, H, W]
            
        Returns:
            mu: Moyenne de la distribution latente [B, latent_dim]
            logvar: Log de la variance [B, latent_dim]
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon
        
        Args:
            mu: Moyenne [B, latent_dim]
            logvar: Log variance [B, latent_dim]
            
        Returns:
            z: Échantillon latent [B, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Décode un vecteur latent en image
        
        Args:
            z: Vecteur latent [B, latent_dim]
            
        Returns:
            reconstruction: Image reconstruite [B, C, H, W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.config.hidden_dims[-1], self.feature_size, self.feature_size)
        result = self.decoder(result)
        # Plus de final_layer - tout est intégré dans decoder maintenant
        return result
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass complet
        
        Args:
            x: Image d'entrée [B, C, H, W]
            
        Returns:
            reconstruction: Image reconstruite
            mu: Moyenne latente
            logvar: Log variance latente
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction sans le reparameterization trick (utilise mu directement)"""
        mu, _ = self.encode(x)
        return self.decode(mu)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Génère de nouvelles images en échantillonnant l'espace latent
        
        Args:
            num_samples: Nombre d'échantillons à générer
            device: Device (CPU ou CUDA)
            
        Returns:
            samples: Images générées [num_samples, C, H, W]
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


# Alias pour compatibilité
VAE = ConvVAE


def vae_loss_function(
    reconstruction: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    reduction: str = 'mean'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fonction de perte VAE: Reconstruction Loss + Beta * KL Divergence
    
    Args:
        reconstruction: Image reconstruite [B, C, H, W]
        x: Image originale [B, C, H, W]
        mu: Moyenne latente [B, latent_dim]
        logvar: Log variance latente [B, latent_dim]
        beta: Poids de la perte KL (beta-VAE)
        reduction: 'mean' ou 'sum'
        
    Returns:
        total_loss: Perte totale
        recon_loss: Perte de reconstruction (L1 pour netteté)
        kl_loss: Divergence KL
    """
    # ✅ SOLUTION 3: Reconstruction loss (L1) - Plus net que MSE
    # L1 Loss préserve mieux les contours et textures fines
    recon_loss = F.l1_loss(reconstruction, x, reduction=reduction)
    
    # KL Divergence
    # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    if reduction == 'mean':
        kl_loss = kl_loss / x.size(0)
    
    # Perte totale
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def get_model_info(model: nn.Module) -> dict:
    """Retourne des informations sur le modèle"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 ** 2)  # Assume float32
    }


if __name__ == '__main__':
    # Test du modèle
    config = VAEConfig(image_size=128, latent_dim=256)
    model = ConvVAE(config)
    
    print("=== VAE Model Information ===")
    info = get_model_info(model)
    for k, v in info.items():
        print(f"{k}: {v}")
    
    # Test forward pass
    x = torch.randn(4, 3, 128, 128)
    reconstruction, mu, logvar = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Test perte
    loss, recon_loss, kl_loss = vae_loss_function(reconstruction, x, mu, logvar)
    print(f"\nTotal loss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    print("\n✅ Model test passed!")
