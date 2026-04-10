#!/bin/bash
# =============================================================================
# env.sh - AutoMoMa environment setup script
#
# This script sets up the AutoMoMa development environment with Python 3.11
# =============================================================================

set -euo pipefail

# Detect conda and setup
if ! command -v conda &> /dev/null; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi

# Environment name
ENV_NAME="${ENV_NAME:-automoma}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# CUDA detection
detect_cuda_home() {
    if [ -n "${CUDA_HOME:-}" ]; then
        echo "Using CUDA_HOME: $CUDA_HOME"
        return 0
    fi

    # Common CUDA installation paths
    local cuda_paths=(
        "/usr/local/cuda"
        "/usr/local/cuda-12.1"
        "/usr/local/cuda-12.4"
        "/usr/local/cuda-12.6"
        "/opt/cuda"
    )

    for path in "${cuda_paths[@]}"; do
        if [ -d "$path" ] && [ -f "$path/bin/nvcc" ]; then
            echo "$path"
            return 0
        fi
    done

    echo ""
    return 1
}

# Main setup
main() {
    echo "=== AutoMoMa Environment Setup ==="
    echo ""

    # 1. Create conda environment
    echo "[1/5] Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    if conda env list | grep -q "^$ENV_NAME "; then
        echo "  Environment '$ENV_NAME' already exists."
    else
        conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
    fi

    # 2. Activate environment
    echo "[2/5] Activating environment..."
    conda activate "$ENV_NAME"

    # 3. Install flit_core (required for editable install)
    echo "[3/5] Installing flit_core..."
    pip install flit_core

    # 4. Install AutoMoMa base package
    echo "[4/5] Installing AutoMoMa base package..."
    pip install -e .

    # 5. Detect CUDA and install curobo if available
    echo "[5/5] Checking CUDA availability..."
    CUDA_HOME=$(detect_cuda_home)
    if [ -n "$CUDA_HOME" ]; then
        export CUDA_HOME
        echo "  CUDA detected at: $CUDA_HOME"
        echo "  Installing curobo with isaacsim support..."
        pip install tomli wheel ninja
        pip install -e "./third_party/curobo [isaacsim]" --no-build-isolation
    else
        echo "  WARNING: CUDA toolkit not found."
        echo "  curobo requires CUDA toolkit to build its CUDA extensions."
        echo "  To install curobo manually:"
        echo "    export CUDA_HOME=/path/to/cuda"
        echo "    pip install -e './third_party/curobo [isaacsim]' --no-build-isolation"
    fi

    echo ""
    echo "=== Setup Complete ==="
    echo ""
    echo "To activate the environment:"
    echo "  conda activate $ENV_NAME"
    echo ""
}

main "$@"
