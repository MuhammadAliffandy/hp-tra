#!/bin/bash
# ==============================================================================
# NEURAL GESTURE MANNEQUIN — DGX LAUNCHER & AUTO-CONDA SETUP SCRIPT
# Automated Environment, Dataset Verification, and Background Execution on DGX
# ==============================================================================
# Usage:
#   ./run_dgx.sh                  # Default: GPU 1, Batch 64, 80 Epochs
#   ./run_dgx.sh 2                # Run on GPU 2
#   ./run_dgx.sh 1 128 100        # Run on GPU 1, Batch Size 128, 100 Epochs
# ==============================================================================

# Default GPU is explicitly set to Device 1
GPU_ID=${1:-1}
BATCH_SIZE=${2:-64}
EPOCHS=${3:-80}

export CUDA_VISIBLE_DEVICES=$GPU_ID
export TF_CPP_MIN_LOG_LEVEL=2

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit 1

ENV_NAME="hp_env"
DATA_DIR="$PROJECT_DIR/dataset_coco"
SAVE_DIR="$PROJECT_DIR/checkpoints_dgx"
LOG_FILE="$PROJECT_DIR/train_dgx.log"

echo "======================================================================"
echo "🚀 LAUNCHING MULTI-TASK VOLUMETRIC POSENET ON DGX CLUSTER"
echo "======================================================================"
echo "📂 Project Root  : $PROJECT_DIR"
echo "🐍 Target Conda  : $ENV_NAME (Python 3.10.2)"
echo "🎮 Default GPU   : CUDA_VISIBLE_DEVICES=$GPU_ID (Device 1)"
echo "📦 Batch Size    : $BATCH_SIZE"
echo "🔄 Total Epochs  : $EPOCHS"
echo "📁 Dataset Dir   : $DATA_DIR (Contained in project for easy cleanup)"
echo "💾 Checkpoints   : $SAVE_DIR"
echo "📝 Master Log    : $LOG_FILE"
echo "🛡️ Mode          : 100% DISCONNECT-PROOF (nohup + disown)"
echo "======================================================================"

# ------------------------------------------------------------------------------
# 1. Automated Conda Environment Setup (hp_env with Python 3.10.2)
# ------------------------------------------------------------------------------
echo "🔍 Checking Conda environment '$ENV_NAME'..."

CONDA_INITIALIZED=false
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    CONDA_INITIALIZED=true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    CONDA_INITIALIZED=true
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    CONDA_INITIALIZED=true
fi

if [ "$CONDA_INITIALIZED" = true ]; then
    if conda env list | grep -E "^${ENV_NAME}\s" > /dev/null 2>&1; then
        echo "✅ Conda environment '$ENV_NAME' already exists."
    else
        echo "📦 Creating fresh Conda environment '$ENV_NAME' with Python 3.10.2..."
        conda create -y -n "$ENV_NAME" python=3.10.2
    fi
    echo "⚡ Activating environment '$ENV_NAME'..."
    conda activate "$ENV_NAME"
else
    echo "⚠️ Conda executable not detected in standard paths. Continuing with current Python: $(python3 --version)"
fi

# ------------------------------------------------------------------------------
# 2. Dependency Verification & ABI Protection (NumPy < 2)
# ------------------------------------------------------------------------------
echo "📦 Verifying dependencies from requirements.txt..."
pip install -r requirements.txt --quiet

# Strict check for NumPy < 2 to prevent PyTorch / C-Extension ABI crashes
python3 -c "import numpy as np; exit(0 if int(np.__version__.split('.')[0]) < 2 else 1)" 2>/dev/null || {
    echo "⚠️ NumPy 2.x detected! Enforcing 'numpy<2' for PyTorch / DGX stability..."
    pip install "numpy<2" --quiet || true
}

# ------------------------------------------------------------------------------
# 3. Disconnect-Proof Execution (nohup + disown on BOTH Download & Training)
# ------------------------------------------------------------------------------
# We wrap the entire process (dataset download if missing + training) into
# a background runner so even if SSH drops or laptop disconnects, the
# pipeline WILL NEVER DIE!
# ------------------------------------------------------------------------------

echo "🛡️ Launching complete pipeline under nohup (immune to SSH disconnect)..."

nohup bash -c "
    cd '$PROJECT_DIR' || exit 1
    
    # Step A: Automated Complete Dataset Download (If missing)
    if [ ! -d '$DATA_DIR/train2017' ] || [ ! -f '$DATA_DIR/annotations/person_keypoints_train2017.json' ]; then
        echo \"📥 [$(date '+%Y-%m-%d %H:%M:%S')] Complete COCO dataset not detected in $DATA_DIR.\"
        echo \"⬇️ [$(date '+%Y-%m-%d %H:%M:%S')] Initiating automated download of COMPLETE dataset (train + val + annotations)...\"
        python3 download_coco.py --data-dir '$DATA_DIR' --split all --clean-zip
    else
        echo \"✅ [$(date '+%Y-%m-%d %H:%M:%S')] Complete COCO 2017 dataset verified in $DATA_DIR.\"
    fi

    # Step B: Headless Multi-Task Volumetric Training
    echo \"🔥 [$(date '+%Y-%m-%d %H:%M:%S')] Starting training process on GPU $GPU_ID (CUDA_VISIBLE_DEVICES=$GPU_ID)...\"
    python3 -u src/train.py \
        --data-dir     '$DATA_DIR' \
        --save-dir     '$SAVE_DIR' \
        --backbone     'mobilenet_v3_large' \
        --epochs       '$EPOCHS' \
        --batch-size   '$BATCH_SIZE' \
        --lr           5e-4 \
        --num-workers  4 \
        --device       'cuda'
" > "$LOG_FILE" 2>&1 &

PID=$!
disown $PID 2>/dev/null || true

echo "======================================================================"
echo "✅ PIPELINE LAUNCHED IN BACKGROUND (DISCONNECT-PROOF)!"
echo "   PID: $PID"
echo "   GPU: Device $GPU_ID (CUDA_VISIBLE_DEVICES=$GPU_ID)"
echo ""
echo "💡 You can now safely close your terminal, shut down SSH, or disconnect."
echo "   The entire download and training will run continuously in the background!"
echo ""
echo "📊 How to monitor execution live:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🖼️ Visual snapshots per epoch:"
echo "   ls -l $SAVE_DIR/snapshots/"
echo ""
echo "🛑 How to stop process if needed:"
echo "   kill -9 $PID"
echo "======================================================================"
