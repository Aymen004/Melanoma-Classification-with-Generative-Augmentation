"""Model architectures for melanoma detection."""

from .vae import ConvVAE
from .densenet import DenseNetClassifier

__all__ = ['ConvVAE', 'DenseNetClassifier']
