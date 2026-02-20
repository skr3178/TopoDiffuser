#!/usr/bin/env bash
# TopoDiffuser Training & Sampling Commands
# Commands actually used from bash history

echo "=========================================="
echo "TopoDiffuser Training Commands"
echo "=========================================="

# =============================================================================
# 1. ENCODER TRAINING (LiDAR-only)
# =============================================================================
echo -e "\n### 1. ENCODER TRAINING ###"
echo "Train encoder on full dataset (all sequences 00-10)"
echo ""
echo "Command:"
echo "  nohup python -u train_encoder_expanded.py --epochs 50 --batch_size 128 > train_encoder_expanded.log 2>&1 &"
echo ""
echo "Output:"
echo "  - checkpoints/encoder_expanded_best.pth"
echo "  - checkpoints/encoder_expanded_latest.pth"
echo ""

# =============================================================================
# 2. DIFFUSION MODEL TRAINING
# =============================================================================
echo -e "\n### 2. DIFFUSION MODEL TRAINING ###"
echo "Train diffusion model with frozen encoder (resume from checkpoint)"
echo ""
echo "Command:"
echo "  nohup python -u train_diffusion_only.py \\"
echo "      --encoder_ckpt checkpoints/encoder_expanded_best.pth \\"
echo "      --resume checkpoints/diffusion_unet_latest.pth \\"
echo "      --epochs 500 \\"
echo "      --batch_size 64 \\"
echo "      --lr 1e-4 \\"
echo "      --noise_schedule cosine > train_diffusion.log 2>&1 &"
echo ""
echo "Output:"
echo "  - checkpoints/diffusion_unet_best.pth"
echo "  - checkpoints/diffusion_unet_latest.pth"
echo "  - checkpoints/diffusion_history.json"
echo ""

# =============================================================================
# 3. SAMPLE DIFFUSION TRAJECTORIES
# =============================================================================
echo -e "\n### 3. SAMPLE DIFFUSION TRAJECTORIES ###"
echo ""
echo "Single-frame inference:"
echo "  python run_diffusion.py --mode infer \\"
echo "      --encoder_ckpt checkpoints/encoder_expanded_best.pth \\"
echo "      --diffusion_ckpt checkpoints/diffusion_unet_best.pth \\"
echo "      --sequence 00 --frame 1000"
echo ""
echo "Batch evaluation on test sequences:"
echo "  python run_diffusion.py --mode eval \\"
echo "      --encoder_ckpt checkpoints/encoder_expanded_best.pth \\"
echo "      --diffusion_ckpt checkpoints/diffusion_unet_best.pth \\"
echo "      --val_sequences 08 09 10"
echo ""

echo "=========================================="
echo "Training Commands Ready!"
echo "=========================================="
