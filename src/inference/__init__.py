"""Inference utilities."""

from .hybrid_system import HybridMelanomaDetector, predict_single_image, predict_batch

__all__ = ['HybridMelanomaDetector', 'predict_single_image', 'predict_batch']
