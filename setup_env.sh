#!/bin/bash
# =============================================================================
# setup_env.sh — One-click environment setup for MAD-Former experiments
# Usage: bash setup_env.sh
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
    echo "[ERROR] Python not found. Please install Python 3.9+ first."
    exit 1
fi

echo "[INFO] Using Python: $($PYTHON --version)"

# ---- Create virtual environment (optional) ----
if [ "$1" == "--venv" ]; then
    VENV_DIR="./venv"
    if [ ! -d "$VENV_DIR" ]; then
        echo "[INFO] Creating virtual environment at $VENV_DIR ..."
        $PYTHON -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null
    echo "[INFO] Virtual environment activated."
fi

# ---- Upgrade pip ----
echo "[INFO] Upgrading pip..."
$PYTHON -m pip install --upgrade pip -q

# ---- Install PyTorch (CPU-only for setup; swap to CUDA version on GPU server) ----
echo "[INFO] Installing PyTorch (CPU version)..."
$PYTHON -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q

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

echo ""
echo "============================================"
echo " Setup Complete!"
echo "============================================"
echo ""
echo " Installed packages:"
$PYTHON -m pip list 2>/dev/null | grep -iE "torch|nibabel|monai|scipy|tensorboard|tqdm|scikit|openpyxl|numpy"
echo ""
echo " If using CUDA GPU, reinstall PyTorch with CUDA:"
echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
echo ""
echo " Next: bash run_experiments.sh"
