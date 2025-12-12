"""
Variational Autoencoder (VAE) pour la Détection d'Anomalies
============================================================

Architecture VAE complète pour apprendre la distribution des images "normales"
(lésions bénignes) et détecter les anomalies via l'erreur de reconstruction.

Théorie:
    - Encodeur: Compresse l'image en vecteur latent (μ, σ) définissant une distribution normale
    - Espace Latent: Représentation compressée de la "normalité"
    - Décodeur: Reconstruit l'image depuis l'espace latent
    
Loss Function:
    Loss = Reconstruction Loss (MSE/BCE) + β * KL Divergence
    - Reconstruction: Force le modèle à bien recréer les images saines
    - KL Divergence: Régularise l'espace latent vers N(0,1)

Usage:
    from VAE_model import VAE, VAEConfig
    config = VAEConfig()
    model = VAE(config)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, List
import math


@dataclass
class VAEConfig:
    """Configuration du Variational Autoencoder"""
    # Architecture
    image_size: int = 128  # Taille des images en entrée
    in_channels: int = 3  # Canaux RGB
    latent_dim: int = 256  # Dimension de l'espace latent
    hidden_dims: List[int] = None  # Dimensions des couches cachées
    
    # Training
    beta: float = 1.0  # Coefficient KL divergence (β-VAE)
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    
    # Regularization
    dropout: float = 0.2
    use_batch_norm: bool = True
    
    # Loss
    reconstruction_loss: str = 'mse'  # 'mse' ou 'bce'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256, 512]


class ResidualBlock(nn.Module):
    """Bloc résiduel pour améliorer le flux de gradient"""
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = True):
        super().__init__()
        self.use_bn = use_bn
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        
        out = out + identity
        out = self.activation(out)
        
        return out


class Encoder(nn.Module):
    """
    Encodeur CNN pour VAE
    Compresse l'image vers un vecteur latent (μ, log_σ²)
    """
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        # Première convolution
        modules = [
            nn.Conv2d(config.in_channels, config.hidden_dims[0], 
                     kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(config.hidden_dims[0]) if config.use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2)
        ]
        
        # Convolutions successives avec downsampling
        in_channels = config.hidden_dims[0]
        for h_dim in config.hidden_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim) if config.use_batch_norm else nn.Identity(),
                    nn.LeakyReLU(0.2),
                    nn.Dropout2d(config.dropout)
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculer la taille après encodage
        self.encoded_size = config.image_size // (2 ** len(config.hidden_dims))
        self.flat_size = config.hidden_dims[-1] * self.encoded_size * self.encoded_size
        
        # Couches pour μ et log(σ²)
        self.fc_mu = nn.Linear(self.flat_size, config.latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, config.latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode l'image en paramètres de la distribution latente
        
        Args:
            x: Image d'entrée [B, C, H, W]
            
        Returns:
            mu: Moyenne de la distribution [B, latent_dim]
            logvar: Log-variance de la distribution [B, latent_dim]
        """
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class Decoder(nn.Module):
    """
    Décodeur CNN pour VAE
    Reconstruit l'image depuis l'espace latent
    """
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        # Taille encodée
        self.encoded_size = config.image_size // (2 ** len(config.hidden_dims))
        self.flat_size = config.hidden_dims[-1] * self.encoded_size * self.encoded_size
        
        # Projection depuis l'espace latent
        self.fc_decode = nn.Linear(config.latent_dim, self.flat_size)
        
        # Convolutions transposées (upsampling)
        modules = []
        hidden_dims_reversed = list(reversed(config.hidden_dims))
        
        for i in range(len(hidden_dims_reversed) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims_reversed[i],
                        hidden_dims_reversed[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims_reversed[i + 1]) if config.use_batch_norm else nn.Identity(),
                    nn.LeakyReLU(0.2),
                    nn.Dropout2d(config.dropout)
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        
        # Couche finale pour reconstruire l'image
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims_reversed[-1],
                hidden_dims_reversed[-1],
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(hidden_dims_reversed[-1]) if config.use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dims_reversed[-1], config.in_channels, 
                     kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Normaliser la sortie entre [0, 1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Décode le vecteur latent en image
        
        Args:
            z: Vecteur latent [B, latent_dim]
            
        Returns:
            reconstruction: Image reconstruite [B, C, H, W]
        """
        h = self.fc_decode(z)
        h = h.view(-1, self.config.hidden_dims[-1], 
                   self.encoded_size, self.encoded_size)
        
        h = self.decoder(h)
        reconstruction = self.final_layer(h)
        
        return reconstruction


class VAE(nn.Module):
    """
    Variational Autoencoder complet
    
    Combine l'encodeur et le décodeur avec le reparameterization trick
    pour permettre la backpropagation à travers l'échantillonnage stochastique.
    """
    def __init__(self, config: VAEConfig = None):
        super().__init__()
        
        if config is None:
            config = VAEConfig()
        
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        # Initialisation des poids
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialisation Xavier pour les poids"""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization Trick
        
        Permet la backpropagation à travers l'échantillonnage:
        z = μ + σ * ε, où ε ~ N(0, I)
        
        Args:
            mu: Moyenne de la distribution latente
            logvar: Log-variance de la distribution latente
            
        Returns:
            z: Échantillon de l'espace latent
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # En mode évaluation, retourner directement la moyenne
            return mu
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass complet
        
        Args:
            x: Image d'entrée [B, C, H, W]
            
        Returns:
            reconstruction: Image reconstruite
            mu: Moyenne latente
            logvar: Log-variance latente
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, logvar
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode une image vers l'espace latent"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Décode un vecteur latent en image"""
        return self.decoder(z)
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruit une image (mode déterministe)"""
        mu, logvar = self.encoder(x)
        z = mu  # Utiliser la moyenne, pas d'échantillonnage
        return self.decoder(z)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Génère des échantillons depuis l'espace latent
        
        Args:
            num_samples: Nombre d'images à générer
            device: Device pour les tenseurs
            
        Returns:
            samples: Images générées [num_samples, C, H, W]
        """
        z = torch.randn(num_samples, self.config.latent_dim, device=device)
        samples = self.decoder(z)
        return samples
    
    def get_reconstruction_error(self, x: torch.Tensor, 
                                  reduction: str = 'none') -> torch.Tensor:
        """
        Calcule l'erreur de reconstruction (MSE par pixel)
        
        Args:
            x: Image d'entrée
            reduction: 'none' pour erreur par image, 'mean' pour moyenne
            
        Returns:
            error: Erreur de reconstruction
        """
        reconstruction = self.reconstruct(x)
        
        if reduction == 'none':
            # MSE par image (moyenne sur C, H, W)
            error = F.mse_loss(reconstruction, x, reduction='none')
            error = error.mean(dim=[1, 2, 3])  # [B]
        else:
            error = F.mse_loss(reconstruction, x, reduction=reduction)
        
        return error


def vae_loss_function(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    reconstruction_loss_type: str = 'mse'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calcule la loss du VAE
    
    Loss = Reconstruction Loss + β * KL Divergence
    
    Args:
        reconstruction: Image reconstruite [B, C, H, W]
        original: Image originale [B, C, H, W]
        mu: Moyenne latente [B, latent_dim]
        logvar: Log-variance latente [B, latent_dim]
        beta: Coefficient pour la KL divergence (β-VAE)
        reconstruction_loss_type: 'mse' ou 'bce'
        
    Returns:
        total_loss: Loss totale
        recon_loss: Loss de reconstruction
        kl_loss: KL divergence
    """
    batch_size = original.size(0)
    
    # Reconstruction Loss
    if reconstruction_loss_type == 'mse':
        # Mean Squared Error
        recon_loss = F.mse_loss(reconstruction, original, reduction='sum') / batch_size
    elif reconstruction_loss_type == 'bce':
        # Binary Cross Entropy (pour images normalisées [0, 1])
        recon_loss = F.binary_cross_entropy(reconstruction, original, reduction='sum') / batch_size
    else:
        raise ValueError(f"Unknown reconstruction loss type: {reconstruction_loss_type}")
    
    # KL Divergence
    # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    
    # Total Loss avec coefficient β
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


class ConvVAE(nn.Module):
    """
    VAE avec architecture convolutionnelle plus profonde
    Meilleure pour les images haute résolution
    """
    def __init__(self, config: VAEConfig = None):
        super().__init__()
        
        if config is None:
            config = VAEConfig()
            config.hidden_dims = [32, 64, 128, 256]
        
        self.config = config
        
        # Encodeur avec blocs résiduels
        encoder_layers = []
        in_ch = config.in_channels
        
        for h_dim in config.hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_ch, h_dim, 4, 2, 1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2),
                ResidualBlock(h_dim, h_dim, config.use_batch_norm)
            ])
            in_ch = h_dim
        
        self.encoder_conv = nn.Sequential(*encoder_layers)
        
        # Taille après convolutions
        self.encoded_size = config.image_size // (2 ** len(config.hidden_dims))
        self.flat_size = config.hidden_dims[-1] * self.encoded_size ** 2
        
        # Projection vers l'espace latent
        self.fc_mu = nn.Linear(self.flat_size, config.latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, config.latent_dim)
        
        # Projection depuis l'espace latent
        self.fc_decode = nn.Linear(config.latent_dim, self.flat_size)
        
        # Décodeur avec blocs résiduels
        decoder_layers = []
        hidden_dims_reversed = list(reversed(config.hidden_dims))
        
        for i in range(len(hidden_dims_reversed) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(hidden_dims_reversed[i], hidden_dims_reversed[i+1], 4, 2, 1),
                nn.BatchNorm2d(hidden_dims_reversed[i+1]),
                nn.LeakyReLU(0.2),
                ResidualBlock(hidden_dims_reversed[i+1], hidden_dims_reversed[i+1], config.use_batch_norm)
            ])
        
        # Couche finale
        decoder_layers.extend([
            nn.ConvTranspose2d(hidden_dims_reversed[-1], config.in_channels, 4, 2, 1),
            nn.Sigmoid()
        ])
        
        self.decoder_conv = nn.Sequential(*decoder_layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, self.config.hidden_dims[-1], self.encoded_size, self.encoded_size)
        return self.decoder_conv(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def get_model_info(model: nn.Module) -> dict:
    """Retourne les informations sur le modèle VAE"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / 1024**2,
        'config': model.config.__dict__ if hasattr(model, 'config') else {}
    }
    
    print(f"\n{'='*60}")
    print(f"VAE Model Information:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Model Size: {info['model_size_mb']:.2f} MB (FP32)")
    if hasattr(model, 'config'):
        print(f"  Latent Dimension: {model.config.latent_dim}")
        print(f"  Image Size: {model.config.image_size}")
        print(f"  Hidden Dims: {model.config.hidden_dims}")
    print(f"{'='*60}\n")
    
    return info


if __name__ == "__main__":
    # Test du modèle VAE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing VAE on device: {device}")
    
    # Configuration
    config = VAEConfig(
        image_size=128,
        latent_dim=256,
        hidden_dims=[32, 64, 128, 256, 512],
        beta=1.0
    )
    
    # Créer le modèle
    model = VAE(config).to(device)
    get_model_info(model)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, config.image_size, config.image_size).to(device)
    
    recon, mu, logvar = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Test loss
    total_loss, recon_loss, kl_loss = vae_loss_function(recon, x, mu, logvar, beta=config.beta)
    print(f"\nLosses:")
    print(f"  Total Loss: {total_loss.item():.4f}")
    print(f"  Reconstruction Loss: {recon_loss.item():.4f}")
    print(f"  KL Divergence: {kl_loss.item():.4f}")
    
    # Test reconstruction error
    error = model.get_reconstruction_error(x)
    print(f"\nReconstruction errors per image: {error.tolist()}")
    
    # Test sampling
    samples = model.sample(4, device)
    print(f"Generated samples shape: {samples.shape}")
    
    print("\n✓ VAE model test passed!")
