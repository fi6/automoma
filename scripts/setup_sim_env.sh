#!/bin/bash
# =============================================================================
# setup_sim_env.sh - Setup environment for SIM mode (IsaacLab + IsaacLab-Arena)
#
# Source this script to set up environment variables for sim mode:
#   source scripts/setup_sim_env.sh
# =============================================================================

# IsaacLab paths
export ISAACLAB_PATH="/home/xinhai/projects/automoma/third_party/IsaacLab-Arena/submodules/IsaacLab"
export IsaacSim_ROOT="/home/xinhai/isaac-sim-5.1.0"

# AutoMoMa asset paths (adjust these to your local paths)
export AUTOMOMA_OBJECT_ROOT="/home/xinhai/projects/automoma/assets/object"
export AUTOMOMA_SCENE_ROOT="/home/xinhai/projects/automoma/assets/scene/infinigen/kitchen_1130"
export AUTOMOMA_ROBOT_ROOT="/home/xinhai/projects/automoma/assets/robot"

# Source Isaac Sim environment setup (must be sourced from _isaac_sim directory)
# This sets up PYTHONPATH and LD_LIBRARY_PATH correctly for Isaac Sim
if [ -f "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" ]; then
    # Save current directory and source the script
    _orig_dir="$(pwd)"
    cd "${ISAACLAB_PATH}/_isaac_sim"
    source "./setup_conda_env.sh"
    cd "${_orig_dir}"
    unset _orig_dir
else
    echo "[WARNING] Isaac Sim setup script not found at ${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh"
    echo "[WARNING] Falling back to manual path configuration..."

    # Add Isaac Sim Python packages and extensions to PYTHONPATH
    export PYTHONPATH="${ISAACLAB_PATH}:${PYTHONPATH:-}"
    export PYTHONPATH="${ISAACLAB_PATH}/_isaac_sim/python_packages:${PYTHONPATH}"
    export PYTHONPATH="${ISAACLAB_PATH}/_isaac_sim/exts/isaacsim.simulation_app:${PYTHONPATH}"
    export PYTHONPATH="${ISAACLAB_PATH}/_isaac_sim/extsDeprecated/omni.isaac.kit:${PYTHONPATH}"
    export PYTHONPATH="${ISAACLAB_PATH}/_isaac_sim/kit/kernel/py:${PYTHONPATH}"
    export PYTHONPATH="${ISAACLAB_PATH}/_isaac_sim/kit/plugins/bindings-python:${PYTHONPATH}"

    # Add Isaac Sim libraries to LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="${IsaacSim_ROOT}/kit/lib/linux-x86_64:${LD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${IsaacSim_ROOT}/exts/omni.graph/${PLATFORM}/:${LD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${IsaacSim_ROOT}/kit/lib/${PLATFORM}:${LD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${IsaacSim_ROOT}/bin/${PLATFORM}:${LD_LIBRARY_PATH:-}"
fi

# Accept NVIDIA EULA
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

echo "[sim env] Environment variables set for SIM mode"
echo "  ISAACLAB_PATH=$ISAACLAB_PATH"
echo "  IsaacSim_ROOT=$IsaacSim_ROOT"
echo "  PYTHONPATH includes Isaac Sim packages"
echo "  LD_LIBRARY_PATH includes Isaac Sim libraries"
