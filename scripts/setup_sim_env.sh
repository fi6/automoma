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

# Add Isaac Sim Python packages to PYTHONPATH
export PYTHONPATH="${ISAACLAB_PATH}:${ISAACLAB_PATH}/_isaac_sim/python_packages:${ISAACLAB_PATH}/_isaac_sim/exts/isaacsim.simulation_app:${PYTHONPATH:-}"

# Accept NVIDIA EULA
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

echo "[sim env] Environment variables set for SIM mode"
echo "  ISAACLAB_PATH=$ISAACLAB_PATH"
echo "  IsaacSim_ROOT=$IsaacSim_ROOT"
echo "  PYTHONPATH updated with Isaac Sim packages"
