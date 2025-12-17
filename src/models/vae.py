"""
Convolutional Variational Autoencoder (ConvVAE) for Anomaly Detection
Trained on benign skin lesion images to learn "normal" skin appearance.
Reconstruction error (L1 Loss) serves as anomaly score for melanoma detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder optimized for 128x128 RGB images.
    
    Architecture:
        - Encoder: 5 convolutional layers progressively reducing spatial dimensions
        - Latent space: 512-dimensional Gaussian distribution
        - Decoder: 5 transposed convolutional layers reconstructing input
    
    Loss: L1 reconstruction loss + KL divergence (β=0.0001 for stable training)
    
    Args:
        latent_dim (int): Dimensionality of latent space. Default: 512
        beta (float): Weight for KL divergence term. Default: 0.0001
    """
    
    def __init__(self, latent_dim=512, beta=0.0001):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        
        # ==================== ENCODER ====================
        # Input: [B, 3, 128, 128]
        self.encoder = nn.Sequential(
            # Conv1: 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv4: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv5: 8x8 -> 4x4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Latent space projection: [B, 512, 4, 4] -> [B, latent_dim]
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        
        # Reverse projection: [B, latent_dim] -> [B, 512, 4, 4]
        self.fc_decode = nn.Linear(latent_dim, 512 * 4 * 4)
        
        # ==================== DECODER ====================
        self.decoder = nn.Sequential(
            # Deconv1: 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Deconv2: 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Deconv3: 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Deconv4: 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Deconv5: 64x64 -> 128x128
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # [B, 3, 128, 128]
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def encode(self, x):
        """
        Encode input image to latent distribution parameters.
        
        Args:
            x (torch.Tensor): Input images [B, 3, 128, 128] normalized to [0, 1]
        
        Returns:
            tuple: (mu, logvar) both of shape [B, latent_dim]
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten: [B, 512*4*4]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)
        
        Args:
            mu (torch.Tensor): Mean of latent distribution [B, latent_dim]
            logvar (torch.Tensor): Log variance of latent distribution [B, latent_dim]
        
        Returns:
            torch.Tensor: Sampled latent vector [B, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, use mean only (deterministic)
            return mu
    
    def decode(self, z):
        """
        Decode latent vector to reconstructed image.
        
        Args:
            z (torch.Tensor): Latent vector [B, latent_dim]
        
        Returns:
            torch.Tensor: Reconstructed image [B, 3, 128, 128] in [0, 1] range
        """
        h = self.fc_decode(z)
        h = h.view(h.size(0), 512, 4, 4)  # Reshape: [B, 512, 4, 4]
        return self.decoder(h)
    
    def forward(self, x):
        """
        Full forward pass: encode -> reparameterize -> decode
        
        Args:
            x (torch.Tensor): Input images [B, 3, 128, 128]
        
        Returns:
            tuple: (reconstructed_x, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar
    
    def compute_loss(self, x, reconstructed_x, mu, logvar):
        """
        Compute VAE loss: L1 reconstruction + β-weighted KL divergence
        
        Args:
            x (torch.Tensor): Original images [B, 3, 128, 128]
            reconstructed_x (torch.Tensor): Reconstructed images [B, 3, 128, 128]
            mu (torch.Tensor): Latent mean [B, latent_dim]
            logvar (torch.Tensor): Latent log variance [B, latent_dim]
        
        Returns:
            dict: {'total_loss', 'recon_loss', 'kl_loss'}
        """
        # L1 Reconstruction Loss (critical for anomaly detection)
        recon_loss = F.l1_loss(reconstructed_x, x, reduction='mean')
        
        # KL Divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # Formula: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Total loss with β weighting for stable training
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def get_anomaly_score(self, x):
        """
        Compute anomaly score based on L1 reconstruction error.
        Higher score indicates more anomalous (potentially malignant) sample.
        
        CRITICAL: Input must be normalized to [0, 1] range, NOT ImageNet normalized!
        
        Args:
            x (torch.Tensor): Input images [B, 3, 128, 128] in [0, 1] range
        
        Returns:
            torch.Tensor: Per-sample L1 reconstruction error [B]
        """
        self.eval()
        with torch.no_grad():
            reconstructed_x, _, _ = self.forward(x)
            # Compute per-sample L1 loss
            l1_error = torch.abs(x - reconstructed_x).view(x.size(0), -1).mean(dim=1)
        return l1_error


def load_vae_checkpoint(checkpoint_path, device='cuda', latent_dim=512):
    """
    Load pre-trained VAE model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to .pth checkpoint file
        device (str): Device to load model on ('cuda' or 'cpu')
        latent_dim (int): Latent dimension (must match training config)
    
    Returns:
        ConvVAE: Loaded model in eval mode
    """
    model = ConvVAE(latent_dim=latent_dim)
    
    # Load checkpoint with device mapping
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test model architecture
    print("=" * 60)
    print("ConvVAE Architecture Test")
    print("=" * 60)
    
    model = ConvVAE(latent_dim=512)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 128, 128)  # Batch of 4 images
    reconstructed, mu, logvar = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Test loss computation
    loss_dict = model.compute_loss(dummy_input, reconstructed, mu, logvar)
    print(f"\nLoss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.6f}")
    
    # Test anomaly score
    scores = model.get_anomaly_score(dummy_input)
    print(f"\nAnomaly scores shape: {scores.shape}")
    print(f"Sample anomaly scores: {scores.cpu().numpy()}")
    
    print("\n✓ All tests passed!")
