#!/usr/bin/env python3
"""
Simple inference script for Hybrid Melanoma Detection

Usage:
    python inference.py --image path/to/lesion.jpg
    python inference.py --image path/to/lesion.jpg --save-output results.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference.hybrid_system import load_hybrid_detector


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Melanoma Detection - Zero-Miss Architecture"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        required=True,
        help="Path to input lesion image"
    )
    parser.add_argument(
        "--densenet-checkpoint",
        type=str,
        default="checkpoints/DenseNet_DDPM.pth",
        help="Path to DenseNet weights"
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default="checkpoints/VAE_best.pth",
        help="Path to VAE weights"
    )
    parser.add_argument(
        "--densenet-threshold",
        type=float,
        default=0.3,
        help="DenseNet classification threshold"
    )
    parser.add_argument(
        "--vae-threshold",
        type=float,
        default=0.136,
        help="VAE anomaly detection threshold"
    )
    parser.add_argument(
        "--save-output",
        type=str,
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Validate image path
    if not Path(args.image).exists():
        print(f"❌ Error: Image not found: {args.image}")
        sys.exit(1)
    
    print("🔬 Loading Hybrid Melanoma Detector...")
    print(f"   DenseNet: {args.densenet_checkpoint}")
    print(f"   VAE: {args.vae_checkpoint}")
    
    # Load detector
    try:
        detector = load_hybrid_detector(
            densenet_checkpoint=args.densenet_checkpoint,
            vae_checkpoint=args.vae_checkpoint,
            densenet_threshold=args.densenet_threshold,
            vae_threshold=args.vae_threshold
        )
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        sys.exit(1)
    
    # Run prediction
    print(f"\n🔍 Analyzing image: {args.image}")
    try:
        results = detector.predict_single(args.image)
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        sys.exit(1)
    
    # Display results
    detector.print_prediction(results)
    
    # Save to file if requested
    if args.save_output:
        with open(args.save_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.save_output}")


if __name__ == "__main__":
    main()
