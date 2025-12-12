#!/bin/bash
# =============================================================================
# VAE Anomaly Detection Training Script
# =============================================================================
# 
# This script trains a VAE on benign images only for anomaly detection
# 
# Usage:
#   ./train_vae.sh [OPTIONS]
#
# Options:
#   --data_csv      Path to CSV file with image info
#   --img_dir       Directory containing images  
#   --epochs        Number of training epochs (default: 100)
#   --beta          Beta coefficient for KL divergence (default: 1.0)
#   --latent_dim    Latent space dimension (default: 256)
#
# =============================================================================

set -e  # Exit on error

# Default parameters
DATA_CSV=""
IMG_DIR=""
EPOCHS=100
BATCH_SIZE=32
LATENT_DIM=256
BETA=1.0
IMAGE_SIZE=128
OUTPUT_DIR="./vae_output"
AUGMENTATION="standard"

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --data_csv)
            DATA_CSV="$2"
            shift 2
            ;;
        --img_dir)
            IMG_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --latent_dim)
            LATENT_DIM="$2"
            shift 2
            ;;
        --beta)
            BETA="$2"
            shift 2
            ;;
        --image_size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --augmentation)
            AUGMENTATION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./train_vae.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data_csv      Path to CSV file with image info"
            echo "  --img_dir       Directory containing images (required)"
            echo "  --epochs        Number of training epochs (default: 100)"
            echo "  --batch_size    Batch size (default: 32)"
            echo "  --latent_dim    Latent space dimension (default: 256)"
            echo "  --beta          Beta coefficient for KL divergence (default: 1.0)"
            echo "  --image_size    Image size (default: 128)"
            echo "  --output_dir    Output directory (default: ./vae_output)"
            echo "  --augmentation  Augmentation level: none, light, standard, heavy (default: standard)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$IMG_DIR" ]; then
    echo "Error: --img_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

# Print configuration
echo "=============================================="
echo "VAE Anomaly Detection Training"
echo "=============================================="
echo "Configuration:"
echo "  Image directory: $IMG_DIR"
echo "  CSV file: ${DATA_CSV:-'None (using all images)'}"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Latent dim: $LATENT_DIM"
echo "  Beta: $BETA"
echo "  Image size: $IMAGE_SIZE"
echo "  Augmentation: $AUGMENTATION"
echo "  Output: $OUTPUT_DIR"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python anomaly_detection/train_vae.py \
    --img_dir $IMG_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --latent_dim $LATENT_DIM \
    --beta $BETA \
    --image_size $IMAGE_SIZE \
    --output_dir $OUTPUT_DIR \
    --augmentation $AUGMENTATION"

# Add optional CSV if provided
if [ -n "$DATA_CSV" ]; then
    CMD="$CMD --data_csv $DATA_CSV"
fi

# Run training
echo ""
echo "Starting training..."
echo "Command: $CMD"
echo ""

eval $CMD

echo ""
echo "=============================================="
echo "Training complete!"
echo "Output saved to: $OUTPUT_DIR"
echo "=============================================="
