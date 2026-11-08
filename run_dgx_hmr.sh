#!/bin/bash
# ==============================================================================
# 🚀 DGX Production Launcher for Parametric Volumetric HMR (Iterasi 2)
# ==============================================================================
# Usage:
#   ./run_dgx_hmr.sh        # Uses GPU 1 by default
#   ./run_dgx_hmr.sh 2      # Uses GPU 2
# ==============================================================================

set -e

GPU_ID=${1:-1}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "=================================================================="
echo "👑 Neural Gesture Mannequin — Iterasi 2 (3D Volumetric HMR SMPL)"
echo "   Target GPU      : $GPU_ID"
echo "   Directory       : $REPO_DIR"
echo "=================================================================="

# 1. Activate Conda Environment (hp_env)
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate hp_env 2>/dev/null || echo "ℹ️ Running in current environment..."
fi

# 2. Check and Download EFT 3D Annotations if not present (~174 MB)
EFT_JSON="./dataset_coco/eft/COCO2014-Part-ver01.json"
if [ ! -f "$EFT_JSON" ]; then
    echo "📥 EFT annotations not detected. Downloading from Facebook AI Research CDN..."
    python3 download_eft.py
else
    echo "✅ EFT annotations detected: $EFT_JSON ($(du -h "$EFT_JSON" | cut -f1))"
fi

# 3. Check Pretrained Backbone from Iterasi 1
PRETRAINED_ARG=""
if [ -f "./checkpoints_dgx/volumetric_posenet_best.pth" ]; then
    echo "📦 Transfer Learning Active: Found Iterasi 1 weights (volumetric_posenet_best.pth)!"
    PRETRAINED_ARG="--pretrained-base ./checkpoints_dgx/volumetric_posenet_best.pth"
else
    echo "ℹ️ Iterasi 1 checkpoint not found in ./checkpoints_dgx/. Starting from ImageNet weights."
fi

# 4. Check Local COCO Images
if [ ! -d "./dataset_coco/train2017" ]; then
    echo "❌ Error: ./dataset_coco/train2017 directory not found!"
    echo "Please ensure the COCO dataset is present in ./dataset_coco."
    exit 1
fi

LOG_FILE="train_dgx_hmr.log"
OUTPUT_DIR="./checkpoints_dgx_hmr"
mkdir -p "$OUTPUT_DIR"

echo "🚀 Launching HMR Training on GPU $GPU_ID in background (nohup)..."
echo "   Log File: $LOG_FILE"

CUDA_VISIBLE_DEVICES=$GPU_ID nohup python3 src/train_hmr.py \
    --eft-json "$EFT_JSON" \
    --coco-train-dir "./dataset_coco/train2017" \
    --coco-val-dir "./dataset_coco/val2017" \
    $PRETRAINED_ARG \
    --backbone mobilenet_v3_large \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.0005 \
    --num-workers 6 \
    --output-dir "$OUTPUT_DIR" > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ Training successfully launched with PID: $PID"
echo "------------------------------------------------------------------"
echo "📋 Monitoring Commands:"
echo "   tail -f $LOG_FILE"
echo "   ps aux | grep train_hmr"
echo "=================================================================="
