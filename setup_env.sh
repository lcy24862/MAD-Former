#!/bin/bash
# =============================================================================
# setup_env.sh — One-click environment setup for MAD-Former experiments
#
# Usage:
#   bash setup_env.sh              # Auto-detect GPU, install optimal PyTorch
#   bash setup_env.sh --venv       # Create virtual environment first
#   bash setup_env.sh --cpu        # Force CPU-only PyTorch
# =============================================================================
set -e

echo "============================================"
echo " MAD-Former Environment Setup"
echo "============================================"

# ---- Detect Python ----
PYTHON=""
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "[ERROR] Python not found. Please install Python 3.10+ first."
    exit 1
fi
echo "[INFO] Python: $($PYTHON --version)"

# ---- Create virtual environment (optional) ----
FORCE_CPU=false
for arg in "$@"; do
    case $arg in
        --venv)
            VENV_DIR="./venv"
            if [ ! -d "$VENV_DIR" ]; then
                echo "[INFO] Creating virtual environment at $VENV_DIR ..."
                $PYTHON -m venv "$VENV_DIR"
            fi
            source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null
            echo "[INFO] Virtual environment activated."
            ;;
        --cpu) FORCE_CPU=true ;;
    esac
done

# ---- Upgrade pip ----
echo "[INFO] Upgrading pip..."
$PYTHON -m pip install --upgrade pip -q

# ---- Detect GPU & CUDA ----
GPU_NAME=""
CUDA_VER=""
INSTALL_CUDA_PYTORCH=false

if $FORCE_CPU; then
    echo "[INFO] Forcing CPU-only installation."
else
    # Try nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$GPU_INFO" ]; then
            GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
            COMPUTE_CAP=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
            echo "[INFO] GPU detected: $GPU_NAME (Compute Capability: $COMPUTE_CAP)"

            # Check CUDA version from nvidia-smi
            CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" 2>/dev/null || echo "")
            [ -n "$CUDA_VER" ] && echo "[INFO] Driver CUDA: $CUDA_VER"

            # Determine PyTorch CUDA index based on GPU architecture
            # Blackwell (RTX 5090/5080/5070): sm_120 → needs CUDA 12.8+
            # Ada Lovelace (RTX 4090/4080): sm_89 → CUDA 11.8+
            # Ampere (RTX 3090/3080/A100): sm_80/86 → CUDA 11.0+
            # Hopper (H100): sm_90 → CUDA 11.8+

            if [ "$COMPUTE_CAP" = "12.0" ] || echo "$GPU_NAME" | grep -qiE "5090|5080|5070|5060|blackwell"; then
                # Blackwell → CUDA 12.8 minimum
                PYTORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu128"
                echo "[INFO] Blackwell GPU → installing PyTorch with CUDA 12.8 (cu128)"
            elif [ "${COMPUTE_CAP%%.*}" -ge 9 2>/dev/null ]; then
                # Hopper (9.x) → CUDA 12.4
                PYTORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu124"
                echo "[INFO] Hopper GPU → installing PyTorch with CUDA 12.4 (cu124)"
            else
                # Older architectures → CUDA 12.4 (widely compatible)
                PYTORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu124"
                echo "[INFO] Installing PyTorch with CUDA 12.4 (cu124)"
            fi
            INSTALL_CUDA_PYTORCH=true
        fi
    else
        echo "[WARN] nvidia-smi not found. Installing CPU-only PyTorch."
        echo "[WARN] On a GPU server, ensure NVIDIA driver + CUDA are installed first!"
    fi
fi

# ---- Install PyTorch ----
echo ""
if $INSTALL_CUDA_PYTORCH; then
    echo "[INFO] Installing PyTorch with CUDA support..."
    $PYTHON -m pip install torch torchvision torchaudio --index-url "$PYTORCH_CUDA_INDEX" -q
else
    echo "[INFO] Installing PyTorch (CPU-only)..."
    $PYTHON -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
fi

# ---- Install core dependencies ----
echo "[INFO] Installing core dependencies..."
$PYTHON -m pip install \
    nibabel \
    monai \
    scipy \
    numpy \
    tensorboard \
    tqdm \
    scikit-learn \
    openpyxl \
    -q

# ---- Verify installation ----
echo ""
echo "============================================"
echo " Verifying Installation"
echo "============================================"

$PYTHON -c "
import sys
print(f'Python:  {sys.version.split()[0]}')

import torch
print(f'PyTorch: {torch.__version__}')

if torch.cuda.is_available():
    print(f'CUDA:    {torch.version.cuda}')
    print(f'GPU:     {torch.cuda.get_device_name(0)}')
    print(f'Memory:  {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
    print(f'Count:   {torch.cuda.device_count()}')
    print('')
    print('[OK] CUDA is available! GPU training ready.')
else:
    print('CUDA:    NOT AVAILABLE (CPU-only mode)')
    import warnings
    warnings.warn('GPU not detected — training will be very slow on CPU!')
"

echo ""
echo "============================================"
echo " Setup Complete!"
echo "============================================"
echo ""
echo " Installed packages:"
$PYTHON -m pip list 2>/dev/null | grep -iE "torch|nibabel|monai|scipy|tensorboard|tqdm|scikit|openpyxl|numpy"
echo ""
echo " Next steps:"
echo "   1. Verify data is in place:  ls data/"
echo "   2. Dry-run experiments:      bash run_experiments.sh --dry-run"
echo "   3. Run all experiments:      nohup bash run_experiments.sh > run.log 2>&1 &"
