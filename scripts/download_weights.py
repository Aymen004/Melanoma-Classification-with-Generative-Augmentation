#!/usr/bin/env python3
"""
Download Pre-trained Model Weights

This script downloads pre-trained model checkpoints from Hugging Face Hub or Google Drive.

Usage:
    # Download specific model
    python download_weights.py --model densenet_ddpm
    
    # Download all essential models
    python download_weights.py --all
    
    # Download hybrid system (DenseNet + VAE)
    python download_weights.py --hybrid
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("❌ huggingface_hub not installed. Install with:")
    print("   pip install huggingface_hub")
    sys.exit(1)


# Model registry
MODELS = {
    'densenet_ddpm': {
        'repo': 'Mustapha03/melanoma-models',
        'filename': 'DenseNet_DDPM.pth',
        'size': '~30MB',
        'description': 'DenseNet-121 trained on DDPM-augmented data'
    },
    'vae_best': {
        'repo': 'Mustapha03/melanoma-models',
        'filename': 'VAE_best.pth',
        'size': '~99MB',
        'description': 'ConvVAE for anomaly detection (trained on benign only)'
    },
    'resnet_ddpm': {
        'repo': 'Mustapha03/melanoma-models',
        'filename': 'ResNet_DDPM.pth',
        'size': '~90MB',
        'description': 'ResNet-50 trained on DDPM-augmented data'
    },
    'vit_ddpm': {
        'repo': 'Mustapha03/melanoma-models',
        'filename': 'ViT_DDPM.pth',
        'size': '~330MB',
        'description': 'Vision Transformer trained on DDPM-augmented data'
    },
    'swin_ddpm': {
        'repo': 'Mustapha03/melanoma-models',
        'filename': 'Swin_DDPM.pth',
        'size': '~330MB',
        'description': 'Swin Transformer trained on DDPM-augmented data'
    }
}


def download_model(model_name: str, save_dir: str = 'checkpoints') -> Optional[Path]:
    """
    Download a specific model from Hugging Face Hub.
    
    Args:
        model_name: Name of the model (e.g., 'densenet_ddpm', 'vae_best')
        save_dir: Directory to save the model
        
    Returns:
        Path to downloaded model or None if failed
    """
    if model_name not in MODELS:
        print(f"❌ Unknown model: {model_name}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return None
    
    model_info = MODELS[model_name]
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"📦 Downloading: {model_name}")
    print(f"📄 Description: {model_info['description']}")
    print(f"💾 Size: {model_info['size']}")
    print(f"{'='*60}\n")
    
    try:
        local_path = hf_hub_download(
            repo_id=model_info['repo'],
            filename=model_info['filename'],
            local_dir=str(save_path),
            local_dir_use_symlinks=False
        )
        print(f"✅ Downloaded to: {local_path}\n")
        return Path(local_path)
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}\n")
        return None


def download_hybrid_system(save_dir: str = 'checkpoints') -> bool:
    """
    Download both models needed for hybrid system (DenseNet + VAE).
    
    Args:
        save_dir: Directory to save models
        
    Returns:
        True if both downloaded successfully
    """
    print("\n🛡️  HYBRID SYSTEM SETUP")
    print("="*60)
    print("Downloading essential models for hybrid melanoma detection:")
    print("  1. DenseNet classifier")
    print("  2. VAE anomaly detector")
    print("="*60)
    
    densenet = download_model('densenet_ddpm', save_dir)
    vae = download_model('vae_best', save_dir)
    
    if densenet and vae:
        print("\n✅ Hybrid system ready!")
        print("\nQuick start:")
        print("  python inference.py --image lesion.jpg")
        return True
    else:
        print("\n❌ Failed to download hybrid system components")
        return False


def download_all(save_dir: str = 'checkpoints') -> None:
    """Download all available models."""
    print("\n📦 DOWNLOADING ALL MODELS")
    print("="*60)
    print(f"Total models: {len(MODELS)}")
    print("This may take several minutes...")
    print("="*60)
    
    success = []
    failed = []
    
    for model_name in MODELS.keys():
        result = download_model(model_name, save_dir)
        if result:
            success.append(model_name)
        else:
            failed.append(model_name)
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"✅ Successful: {len(success)}/{len(MODELS)}")
    if success:
        for model in success:
            print(f"   • {model}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(MODELS)}")
        for model in failed:
            print(f"   • {model}")
    print("="*60)


def list_models() -> None:
    """Display all available models."""
    print("\n📚 AVAILABLE MODELS")
    print("="*60)
    for name, info in MODELS.items():
        print(f"\n🔹 {name}")
        print(f"   Description: {info['description']}")
        print(f"   Size: {info['size']}")
        print(f"   Filename: {info['filename']}")
    print("="*60)
    print(f"\nTotal: {len(MODELS)} models available")
    print("\nUsage:")
    print(f"  python {sys.argv[0]} --model <model_name>")


def main():
    parser = argparse.ArgumentParser(
        description='Download pre-trained melanoma detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download specific model
  python download_weights.py --model densenet_ddpm
  
  # Download hybrid system (recommended)
  python download_weights.py --hybrid
  
  # Download all models
  python download_weights.py --all
  
  # List available models
  python download_weights.py --list
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Specific model to download (e.g., densenet_ddpm, vae_best)'
    )
    parser.add_argument(
        '--hybrid',
        action='store_true',
        help='Download both models for hybrid system (DenseNet + VAE)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available models'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available models'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints',
        help='Directory to save models (default: checkpoints/)'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
    elif args.hybrid:
        download_hybrid_system(args.save_dir)
    elif args.all:
        download_all(args.save_dir)
    elif args.model:
        download_model(args.model, args.save_dir)
    else:
        parser.print_help()
        print("\n💡 TIP: Start with --hybrid for quick setup")


if __name__ == '__main__':
    main()
