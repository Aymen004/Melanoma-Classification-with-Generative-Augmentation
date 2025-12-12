"""
Anomaly Detection Module - VAE-based Detection for Melanoma
============================================================

Ce module implémente un système de détection d'anomalies basé sur un
Variational Autoencoder (VAE) pour identifier les lésions cutanées anormales.

L'approche VAE renverse le problème de classification :
- Au lieu d'apprendre à quoi ressemble un cancer, on apprend ce qu'est la "normalité"
- Tout ce qui s'éloigne trop de cette normalité est signalé comme anomalie

Avantages:
- Indépendance vis-à-vis des données rares
- Filet de sécurité Out-of-Distribution (OOD)
- Complémentaire au classificateur supervisé (DenseNet)

Components:
    - VAE_model.py: Architecture du Variational Autoencoder
    - train_vae.py: Script d'entraînement sur données bénignes
    - inference_vae.py: Inférence et calibrage du seuil
    - hybrid_classifier.py: Fusion VAE + DenseNet
"""

from .VAE_model import (
    VAE,
    VAEConfig,
    Encoder,
    Decoder,
    vae_loss_function
)

from .train_vae import (
    VAETrainer,
    BenignOnlyDataset
)

from .inference_vae import (
    VAEAnomalyDetector,
    calculate_reconstruction_error,
    calibrate_threshold
)

__version__ = '1.0.0'
__author__ = 'Melanoma Classification Project'

__all__ = [
    'VAE',
    'VAEConfig',
    'Encoder',
    'Decoder',
    'vae_loss_function',
    'VAETrainer',
    'BenignOnlyDataset',
    'VAEAnomalyDetector',
    'calculate_reconstruction_error',
    'calibrate_threshold'
]
