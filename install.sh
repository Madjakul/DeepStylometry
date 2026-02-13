#!/bin/bash

# Default Configuration
ENV_NAME="deep-stylometry"
PYTHON_VERSION="3.11"
TORCH_VERSION="2.6.0"
CUDA_VERSION_SHORT="124"         # 124 for cuda 12.4
FLASH_ATTN_VERSION="2.7.4.post1" # The stable version we confirmed works

# Help Function
usage() {
    echo "Usage: $0 [-n env_name] [-p python_version] [-c cuda_short_code]"
    echo "  -n: Conda environment name (default: $ENV_NAME)"
    echo "  -p: Python version (default: $PYTHON_VERSION)"
    echo "  -c: CUDA short code for PyTorch (default: $CUDA_VERSION_SHORT, e.g., 118, 121, 124)"
    exit 1
}

# Parse Flags
while getopts "n:p:c:h" opt; do
    case $opt in
    n) ENV_NAME="$OPTARG" ;;
    p) PYTHON_VERSION="$OPTARG" ;;
    c) CUDA_VERSION_SHORT="$OPTARG" ;;
    h) usage ;;
    *) usage ;;
    esac
done

echo "========================================================"
echo "Initializing Setup for Env: $ENV_NAME"
echo "Target: Python $PYTHON_VERSION | Torch $TORCH_VERSION | CUDA $CUDA_VERSION_SHORT"
echo "========================================================"

# 1. Create and Activate Environment
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "[!] Environment $ENV_NAME already exists. Skipping creation."
else
    echo "[*] Creating conda environment..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

echo "[*] Activating environment..."
conda activate $ENV_NAME

# 2. Base Dependencies
echo "[*] Installing base build tools..."
pip install --upgrade pip setuptools wheel packaging psutil ninja numpy

# 3. Install PyTorch
PYTORCH_INDEX="https://download.pytorch.org/whl/cu${CUDA_VERSION_SHORT}"
echo "[*] Installing PyTorch from $PYTORCH_INDEX..."

# We force reinstall to ensure we get the CUDA version, not the cached CPU version
pip install torch==${TORCH_VERSION} \
    --index-url $PYTORCH_INDEX

# 4. Runtime Detection
echo "[*] Detecting Runtime Configuration..."

DETECT_OUTPUT=$(python3 -c "
import torch
import sys
import platform

# Get Major.Minor version (e.g., 2.6.0 -> 2.6)
v_torch = '.'.join(torch.__version__.split('+')[0].split('.')[:2])

# Clean cuda version (strip + and letters)
v_cuda = torch.version.cuda.replace('.', '')

# Detect ABI (0 or 1)
v_cxx11 = 'TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE'

# Python tag (e.g., cp311)
v_py = f'cp{sys.version_info.major}{sys.version_info.minor}'
# Platform
v_plat = 'linux_x86_64'

print(f'{v_torch} {v_cuda} {v_cxx11} {v_py}')
")

read -r DET_TORCH DET_CUDA DET_ABI DET_PY <<<"$DETECT_OUTPUT"

echo "    - Installed Torch: $DET_TORCH"
echo "    - Built with CUDA: $DET_CUDA"
echo "    - CXX11 ABI:       $DET_ABI"
echo "    - Python Tag:      $DET_PY"

# 5. Construct Flash Attention Wheel URL
# Naming Convention: flash_attn-{ver}+cu{cuda}torch{torch}cxx11abi{BOOL}-{py}-{py}-{plat}.whl
# Example: flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Note: Flash Attention wheels usually drop the minor cuda version (cu124 -> cu12)
SHORT_CUDA=${DET_CUDA:0:2} # Takes first 2 chars of 124 -> 12

WHEEL_NAME="flash_attn-${FLASH_ATTN_VERSION}+cu${SHORT_CUDA}torch${DET_TORCH}cxx11abi${DET_ABI}-${DET_PY}-${DET_PY}-linux_x86_64.whl"
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/${WHEEL_NAME}"

echo "[*] Downloading and Installing Flash Attention..."
echo "    - Target Wheel: $WHEEL_NAME"

# We use --no-deps here to prevent dependency resolution issues
pip install $WHEEL_URL --no-build-isolation --no-cache-dir --no-deps

if [ $? -eq 0 ]; then
    echo "[✔] Flash Attention installed successfully!"
else
    echo "[✘] Failed to install Flash Attention. The wheel URL might be invalid for this combination."
    exit 1
fi

# 6. Generate Constraints File (The "Safety" Lock)
echo "[*] Generating constraints.txt to lock environment..."
CONSTRAINTS_FILE="constraints.txt"

# Get all installed nvidia packages and torch with their exact versions
pip freeze | grep -E "^(torch==|nvidia-)" >$CONSTRAINTS_FILE

echo "    - Generated $CONSTRAINTS_FILE with $(wc -l <$CONSTRAINTS_FILE) locked packages."

# 7. Final Verification
echo "[*] verifying installation..."
python3 -c "import torch; import flash_attn; print(f'Success! Flash Attn {flash_attn.__version__} loaded on CUDA {torch.version.cuda}')"

echo "========================================================"
echo "Setup Complete. To install future packages, run:"
echo "pip install <package> -c constraints.txt --extra-index-url $PYTORCH_INDEX"
echo "========================================================"
