#!/usr/bin/env bash
# Render actual planned-trajectory ghost panels for the poster comparison.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-$REPO_ROOT/third_party/IsaacLab-Arena/submodules/IsaacLab}"

if [[ -z "${AUTOMOMA_PYTHON:-}" && -x "$HOME/miniconda3/envs/automoma/bin/python" ]]; then
    AUTOMOMA_PYTHON="$HOME/miniconda3/envs/automoma/bin/python"
fi
PYTHON_BIN="${AUTOMOMA_PYTHON:-python}"
PYTHON_ENV_DIR="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
if [[ -d "$PYTHON_ENV_DIR/lib" ]]; then
    export CONDA_PREFIX="$PYTHON_ENV_DIR"
    export LD_LIBRARY_PATH="$PYTHON_ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export AUTOMOMA_OBJECT_ROOT="$REPO_ROOT/assets/object"
export AUTOMOMA_SCENE_ROOT="$REPO_ROOT/assets/scene/ithor"
export AUTOMOMA_ROBOT_ROOT="$REPO_ROOT/assets/robot"

source "$REPO_ROOT/scripts/setup_sim_env.sh"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/third_party/IsaacLab-Arena:$ISAACLAB_PATH/source/isaaclab:$ISAACLAB_PATH/source/isaaclab_assets:$ISAACLAB_PATH/source/isaaclab_tasks:$ISAACLAB_PATH/source/isaaclab_mimic${PYTHONPATH:+:$PYTHONPATH}"

args=("$@")
has_enable_cameras=false
has_headless=false
for arg in "${args[@]}"; do
    [[ "$arg" == "--enable_cameras" ]] && has_enable_cameras=true
    [[ "$arg" == "--headless" ]] && has_headless=true
done

if [[ "$has_enable_cameras" == false ]]; then
    args=(--enable_cameras "${args[@]}")
fi
if [[ "$has_headless" == false ]]; then
    args=(--headless "${args[@]}")
fi

cd "$REPO_ROOT"
"$PYTHON_BIN" "$REPO_ROOT/tools/paper/poster/render_actual_ghost_comparison.py" "${args[@]}"
