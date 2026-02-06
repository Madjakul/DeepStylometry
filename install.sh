#!/bin/bash
set -e # Exit on error

# === Defaults ===
CUDA_VERSION="12.1"
ENV_NAME=""
INSTALL_DEEPSPEED=false
PYTHON_VERSION="3.11"

# === Help Function ===
print_usage() {
    echo "Usage: $0 --env <name> [OPTIONS]"
    echo "Options:"
    echo "  --env <name>       Name of the conda environment (Required)"
    echo "  --cuda <ver>       CUDA version (default: 12.1)"
    echo "  --python <ver>     Python version (default: 3.11)"
    echo "  --with-deepspeed   Install DeepSpeed"
    echo "  --help             Show this help"
    exit 1
}

# === Parse Arguments ===
while [[ "$#" -gt 0 ]]; do
    case $1 in
    --env)
        ENV_NAME="$2"
        shift
        ;;
    --cuda)
        CUDA_VERSION="$2"
        shift
        ;;
    --python)
        PYTHON_VERSION="$2"
        shift
        ;;
    --with-deepspeed) INSTALL_DEEPSPEED=true ;;
    --help) print_usage ;;
    *)
        echo "Unknown parameter: $1"
        print_usage
        ;;
    esac
    shift
done

if [[ -z "$ENV_NAME" ]]; then
    echo "ERROR: You must provide an environment name with --env"
    print_usage
fi

# === 1. Setup/Activate Conda Environment ===
setup_conda_env() {
    echo "=== Managing Conda Environment: $ENV_NAME ==="

    # Source conda.sh so we can use 'conda activate' inside this script
    # Try standard locations
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$WORK_DIR/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$WORK_DIR/miniconda3/etc/profile.d/conda.sh"
    else
        # Fallback: try to find where conda is
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [[ -n "$CONDA_BASE" ]]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        else
            echo "ERROR: Could not find conda.sh to enable activation."
            exit 1
        fi
    fi

    # Check if env exists
    if conda info --envs | grep -q "^$ENV_NAME "; then
        echo "Environment '$ENV_NAME' exists. Activating..."
        conda activate "$ENV_NAME"
    else
        echo "Creating environment '$ENV_NAME' with Python $PYTHON_VERSION..."
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
        conda activate "$ENV_NAME"
    fi
}

# === 2. Setup CUDA (The "Hybrid" Approach) ===
setup_cuda() {
    echo "=== Setting up CUDA $CUDA_VERSION ==="

    # We install the compiler into the environment to make it portable
    if ! command -v nvcc &>/dev/null; then
        echo "Installing CUDA Compiler (nvcc) via Conda..."
        # Install nvcc and dev headers
        conda install -y -c nvidia "cuda-nvcc=$CUDA_VERSION" "cuda-libraries-dev=$CUDA_VERSION"
    else
        echo "nvcc already found at $(which nvcc)"
    fi

    # Calculate paths based on current Conda env
    export CUDA_HOME=$CONDA_PREFIX
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

    # Verify
    echo "Active CUDA_HOME: $CUDA_HOME"
    nvcc --version | grep release
}

# === 3. Make Variables Permanent (The "Pro" Trick) ===
make_vars_permanent() {
    echo "=== Making Environment Variables Permanent for '$ENV_NAME' ==="

    ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
    DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"

    mkdir -p "$ACTIVATE_DIR"
    mkdir -p "$DEACTIVATE_DIR"

    # Write activation script
    cat >"$ACTIVATE_DIR/env_vars.sh" <<EOF
#!/bin/sh
export CUDA_HOME=$CONDA_PREFIX
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib:\$LD_LIBRARY_PATH
export DS_BUILD_OPS=0
EOF

    # Write deactivation script (cleanup)
    cat >"$DEACTIVATE_DIR/env_vars.sh" <<EOF
#!/bin/sh
unset CUDA_HOME
unset DS_BUILD_OPS
# Note: Cleaning up PATH/LD_LIBRARY_PATH cleanly is hard in shell, 
# usually we leave them or rely on conda deactivate to restore the previous state.
EOF

    echo "Saved variables to $ACTIVATE_DIR/env_vars.sh"
}

# === 4. Install Packages ===
install_packages() {
    # Install build tools
    pip install --upgrade packaging ninja setuptools wheel

    # Determine Torch Index based on CUDA version (simplified logic)
    # Removing dots for url (12.1 -> 121)
    CUDA_SHORT=${CUDA_VERSION//./}
    TORCH_INDEX="https://download.pytorch.org/whl/cu${CUDA_SHORT}"

    echo "Installing PyTorch from $TORCH_INDEX..."
    pip install --upgrade torch --index-url "$TORCH_INDEX"

    echo "Installing Flash Attention..."
    export MAX_JOBS=4
    # --no-build-isolation is KEY here
    pip install --no-build-isolation flash-attn

    if [ "$INSTALL_DEEPSPEED" = true ]; then
        echo "Installing DeepSpeed..."
        export DS_BUILD_OPS=0
        pip install deepspeed
    fi
}

# === Main Execution ===
setup_conda_env
setup_cuda
make_vars_permanent # Only works because we are inside the env
install_packages

echo ""
echo "======================================================="
echo "   INSTALLATION COMPLETE"
echo "======================================================="
echo "To start working, run:"
echo "   conda activate $ENV_NAME"
echo ""
echo "Note: CUDA_HOME and PATHs are now auto-configured"
echo "whenever you activate this environment."
echo "======================================================="
