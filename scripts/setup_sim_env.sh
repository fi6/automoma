#!/usr/bin/env bash
# Source this file to configure AutoMoMa simulation environment variables:
#   source scripts/setup_sim_env.sh

_SETUP_SIM_ENV_SOURCED=false
if [[ -n "${BASH_VERSION:-}" ]]; then
    _SETUP_SIM_ENV_SOURCE="${BASH_SOURCE[0]}"
    [[ "$_SETUP_SIM_ENV_SOURCE" != "$0" ]] && _SETUP_SIM_ENV_SOURCED=true
elif [[ -n "${ZSH_VERSION:-}" ]]; then
    _SETUP_SIM_ENV_SOURCE="${(%):-%N}"
    [[ ":${ZSH_EVAL_CONTEXT:-}:" == *":file:"* ]] && _SETUP_SIM_ENV_SOURCED=true
else
    _SETUP_SIM_ENV_SOURCE="$0"
fi

if [[ "$_SETUP_SIM_ENV_SOURCED" != "true" ]]; then
    echo "Error: source this script instead of executing it: source scripts/setup_sim_env.sh" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$_SETUP_SIM_ENV_SOURCE")/.." && pwd)"
unset _SETUP_SIM_ENV_SOURCE _SETUP_SIM_ENV_SOURCED

export ISAACLAB_PATH="${ISAACLAB_PATH:-$REPO_ROOT/third_party/IsaacLab-Arena/submodules/IsaacLab}"
export AUTOMOMA_OBJECT_ROOT="${AUTOMOMA_OBJECT_ROOT:-$REPO_ROOT/assets/object}"
export AUTOMOMA_SCENE_ROOT="${AUTOMOMA_SCENE_ROOT:-$REPO_ROOT/assets/scene/infinigen/kitchen_1130}"
export AUTOMOMA_ROBOT_ROOT="${AUTOMOMA_ROBOT_ROOT:-$REPO_ROOT/assets/robot}"

if [[ -f "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" ]]; then
    _orig_dir="$(pwd)"
    cd "${ISAACLAB_PATH}/_isaac_sim"
    source "./setup_conda_env.sh"
    cd "${_orig_dir}"
    unset _orig_dir
elif [[ -n "${IsaacSim_ROOT:-}" && -d "${IsaacSim_ROOT}" ]]; then
    echo "[sim env] Warning: Isaac Sim setup script not found at ${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" >&2
    echo "[sim env] Falling back to a minimal PYTHONPATH/LD_LIBRARY_PATH setup." >&2
    export PYTHONPATH="${ISAACLAB_PATH}:${ISAACLAB_PATH}/_isaac_sim/python_packages:${PYTHONPATH:-}"
    export LD_LIBRARY_PATH="${IsaacSim_ROOT}/kit/lib/linux-x86_64:${IsaacSim_ROOT}/kit/lib:${LD_LIBRARY_PATH:-}"
else
    echo "[sim env] IsaacSim_ROOT is not set; relying on the active Python environment for Isaac Sim packages." >&2
    export PYTHONPATH="${ISAACLAB_PATH}:${PYTHONPATH:-}"
fi

export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
export OMNI_KIT_ALLOW_ROOT="${OMNI_KIT_ALLOW_ROOT:-1}"

echo "[sim env] AutoMoMa simulation environment configured"
echo "  ISAACLAB_PATH=$ISAACLAB_PATH"
echo "  IsaacSim_ROOT=${IsaacSim_ROOT:-<active-python-env>}"
echo "  AUTOMOMA_OBJECT_ROOT=$AUTOMOMA_OBJECT_ROOT"
echo "  AUTOMOMA_SCENE_ROOT=$AUTOMOMA_SCENE_ROOT"
echo "  AUTOMOMA_ROBOT_ROOT=$AUTOMOMA_ROBOT_ROOT"
