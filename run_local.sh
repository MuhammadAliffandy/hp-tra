#!/bin/bash
# ==============================================================================
# NEURAL GESTURE MANNEQUIN — LOCAL WORKSTATION LAUNCHER (macOS / Local Linux)
# Ideal for rapid smoke testing and local validation
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit 1

DATA_DIR="$PROJECT_DIR/dataset_coco"
SAVE_DIR="$PROJECT_DIR/checkpoints_local"

echo "======================================================================"
echo "💻 LAUNCHING VOLUMETRIC POSENET (LOCAL WORKSTATION MODE)"
echo "======================================================================"
echo "📂 Working Dir  : $PROJECT_DIR"
echo "📦 Data Root    : $DATA_DIR"
echo "💾 Checkpoints  : $SAVE_DIR"
echo "======================================================================"

# Determine best local device
DEVICE="cpu"
if python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    DEVICE="mps"
    echo "🍏 Apple Silicon MPS (Metal Performance Shaders) detected!"
elif python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
    echo "🎮 NVIDIA CUDA detected!"
else
    echo "🖥️ Running on CPU mode."
fi

# 1. Download minimal validation set if missing
if [ ! -f "$DATA_DIR/annotations/person_keypoints_val2017.json" ]; then
    echo "📥 Downloading COCO validation set for local testing..."
    python3 download_coco.py --data-dir "$DATA_DIR" --split val
fi

# 2. Run Local Training Smoke Test
python3 src/train.py \
    --data-dir     "$DATA_DIR" \
    --save-dir     "$SAVE_DIR" \
    --backbone     "mobilenet_v3_small" \
    --epochs       5 \
    --batch-size   16 \
    --lr           5e-4 \
    --num-workers  0 \
    --device       "$DEVICE"
