#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Unified pipeline for the lerobot-arena project.
#
# Supports: record, convert, train, eval
#
# Usage:
#   bash scripts/run_pipeline.sh <mode> [mode-specific args] [overrides...]
#
#   record:
#     bash scripts/run_pipeline.sh record <object_name> <scene_name> <num_episodes> [overrides...]
#     e.g.  bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 30
#           bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 30 --interpolated 2
#
#   convert:
#     bash scripts/run_pipeline.sh convert <object_name> <scene_name> <num_episodes> [overrides...]
#
#   train:
#     bash scripts/run_pipeline.sh train <policy> <object_name> <scene_name> <num_episodes> [overrides...]
#     e.g.  bash scripts/run_pipeline.sh train act microwave_7221 scene_0_seed_0 30 --steps=20000
#
#   eval:
#     bash scripts/run_pipeline.sh eval <policy> <object_name> <scene_name> <num_episodes> [overrides...]
#     e.g.  bash scripts/run_pipeline.sh eval act microwave_7221 scene_0_seed_0 30 --eval.n_episodes=50
#
# Override mechanism:
#   Extra flags after the positional arguments are appended to the underlying
#   command.  For record mode, overrides are inserted before the subcommand
#   name so argparse sees them as global flags (e.g. --interpolated, --device).
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ISAACLAB_ARENA="$REPO_ROOT/IsaacLab-Arena"

# ---------------------------------------------------------------------------
# Export env vars so IsaacLab-Arena resolves assets from lerobot-arena
# ---------------------------------------------------------------------------
export AUTOMOMA_OBJECT_ROOT="$REPO_ROOT/assets/object"
export AUTOMOMA_SCENE_ROOT="$REPO_ROOT/assets/scene/infinigen/kitchen_1130"
export AUTOMOMA_ROBOT_ROOT="$REPO_ROOT/assets/robot"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_pipeline.sh record  <object_name> <scene_name> <num_ep>  [overrides...]
  bash scripts/run_pipeline.sh convert <object_name> <scene_name> <num_ep>  [overrides...]
  bash scripts/run_pipeline.sh train   <policy> <object_name> <scene_name> <num_ep> [overrides...]
  bash scripts/run_pipeline.sh eval    <policy> <object_name> <scene_name> <num_ep> [overrides...]

Examples:
  bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 30
  bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 30 --interpolated 2
  bash scripts/run_pipeline.sh train act microwave_7221 scene_0_seed_0 30 --steps=20000
  bash scripts/run_pipeline.sh eval  act microwave_7221 scene_0_seed_0 30 --eval.n_episodes=50
EOF
    exit 1
}

timestamp() { date +"%Y%m%d_%H%M%S"; }

# Experiment name:  summit_franka_open-<object>-<scene>-<N>
mk_exp_name() { echo "summit_franka_open-${1}-${2}-${3}"; }

require_arg() {
    if [[ -z "${1:-}" ]]; then
        echo "Error: Missing required argument: $2" >&2
        usage
    fi
}

# Setup logging: tee stdout+stderr to a timestamped log file.
setup_log() {
    local mode="$1" obj="$2" scene="$3"
    LOG_DIR="$REPO_ROOT/logs/$mode/$obj/$scene"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/$(timestamp).log"
    echo "========================================"
    echo "  Log file: $LOG_FILE"
    echo "========================================"
}

run_logged() {
    "$@" 2>&1 | tee -a "$LOG_FILE"
    return "${PIPESTATUS[0]}"
}

# ---------------------------------------------------------------------------
# Mode: record
# ---------------------------------------------------------------------------
do_record() {
    local object_name="$1"; shift
    local scene_name="$1"; shift
    local num_episodes="$1"; shift
    # $@ = overrides

    local name; name="$(mk_exp_name "$object_name" "$scene_name" "$num_episodes")"
    local traj_file="$REPO_ROOT/data/trajs/summit_franka/${object_name}/${scene_name}/traj_data_train.pt"
    local dataset_file="$REPO_ROOT/data/automoma/${name}.hdf5"

    setup_log record "$object_name" "$scene_name"

    # Build command.  Overrides go before the subcommand so argparse
    # treats them as top-level flags (e.g. --interpolated, --device).
    local -a cmd=(
        python isaaclab_arena/scripts/record_automoma_demos.py
        --enable_cameras
        --mobile_base_relative
        --traj_file "$traj_file"
        --dataset_file "$dataset_file"
        --num_episodes "$num_episodes"
        "$@"                              # ← overrides (before subcommand)
        summit_franka_open_door           # subcommand
        --object_name "$object_name"
        --scene_name "$scene_name"
        --object_center
    )

    echo "Command: ${cmd[*]}"
    echo ""

    pushd "$ISAACLAB_ARENA" > /dev/null
    run_logged "${cmd[@]}"
    popd > /dev/null
}

# ---------------------------------------------------------------------------
# Mode: convert (convert_hdf5_to_lerobot)
# ---------------------------------------------------------------------------
do_convert() {
    local object_name="$1"; shift
    local scene_name="$1"; shift
    local num_episodes="$1"; shift

    local name; name="$(mk_exp_name "$object_name" "$scene_name" "$num_episodes")"

    setup_log convert "$object_name" "$scene_name"

    local -a cmd=(
        python isaaclab_arena_gr00t/data_utils/convert_hdf5_to_lerobot_v30.py
        --yaml_file isaaclab_arena_gr00t/config/summit_franka_manip_config.yaml
        --data_root "$REPO_ROOT/data/automoma"
        --hdf5_name "${name}.hdf5"
        --repo_id "automoma/${name}"
        --output_dir "$REPO_ROOT/data/lerobot/automoma/${name}"
        "$@"
    )

    echo "Command: ${cmd[*]}"
    echo ""

    pushd "$ISAACLAB_ARENA" > /dev/null
    run_logged "${cmd[@]}"
    popd > /dev/null
}

# ---------------------------------------------------------------------------
# Mode: train
# ---------------------------------------------------------------------------
do_train() {
    local policy="$1"; shift
    local object_name="$1"; shift
    local scene_name="$1"; shift
    local num_episodes="$1"; shift

    local name; name="$(mk_exp_name "$object_name" "$scene_name" "$num_episodes")"
    local dataset_root="$REPO_ROOT/data/lerobot/automoma/${name}"
    local output_dir="$REPO_ROOT/outputs/train/${policy}_${name}"

    setup_log train "$object_name" "$scene_name"

    # Interactive check for existing output
    if [[ -d "$output_dir" ]]; then
        echo "Warning: Output directory already exists:"
        echo "  $output_dir"
        read -r -p "Delete and retrain? [y/N] " answer
        if [[ "${answer,,}" == "y" ]]; then
            rm -rf "$output_dir"
            echo "Deleted."
        else
            echo "Keeping existing directory."
        fi
    fi

    local -a cmd=(
        lerobot-train
        --policy.type="$policy"
        --batch_size=128
        --steps=10000
        --log_freq=50
        --eval_freq=500
        --save_freq=1000
        --job_name="${policy}_${name}"
        --dataset.repo_id="$name"
        --dataset.root="$dataset_root"
        --policy.chunk_size=16
        --policy.n_action_steps=16
        --policy.optimizer_lr=1e-4
        --policy.push_to_hub=false
        --policy.device=cuda
        --wandb.enable=true
        --output_dir="$output_dir"
        --dataset.preload=true
        --dataset.preload_cache=true
        --dataset.filter_features_by_policy=true
        "$@"
    )

    echo "Command: ${cmd[*]}"
    echo ""

    run_logged "${cmd[@]}"
}

# ---------------------------------------------------------------------------
# Mode: eval
# ---------------------------------------------------------------------------
do_eval() {
    local policy="$1"; shift
    local object_name="$1"; shift
    local scene_name="$1"; shift
    local num_episodes="$1"; shift

    local name; name="$(mk_exp_name "$object_name" "$scene_name" "$num_episodes")"
    local policy_path="$REPO_ROOT/outputs/train/${policy}_${name}/checkpoints/last/pretrained_model"
    local traj_file="$REPO_ROOT/data/trajs/summit_franka/${object_name}/${scene_name}/traj_data_test.pt"

    setup_log eval "$object_name" "$scene_name"

    # Build env.kwargs JSON
    local env_kwargs="{\"object_name\": \"${object_name}\", \"scene_name\": \"${scene_name}\", \"object_center\": true, \"mobile_base_relative\": true, \"traj_file\": \"${traj_file}\", \"traj_seed\": 42}"

    local rename_map='{"observation.images.ego_topdown_rgb": "observation.images.ego_topdown", "observation.images.ego_wrist_rgb": "observation.images.ego_wrist", "observation.images.fix_local_rgb": "observation.images.fix_local"}'

    local -a cmd=(
        lerobot-eval
        --policy.path="$policy_path"
        --policy.device=cuda
        --env.type=isaaclab_arena
        --env.hub_path="$ISAACLAB_ARENA/isaaclab-arena-envs"
        --env.environment=summit_franka_open_door_eval
        --env.headless=false
        --env.enable_cameras=true
        --env.state_keys=joint_pos
        --env.camera_keys=ego_topdown_rgb,ego_wrist_rgb,fix_local_rgb
        --env.state_dim=12
        --env.action_dim=12
        --env.camera_height=240
        --env.camera_width=320
        --env.episode_length=300
        "--env.kwargs=$env_kwargs"
        "--rename_map=$rename_map"
        --trust_remote_code=true
        --eval.batch_size=1
        --eval.n_episodes=50
        "$@"
    )

    echo "Command: ${cmd[*]}"
    echo ""

    pushd "$ISAACLAB_ARENA" > /dev/null
    run_logged "${cmd[@]}"
    popd > /dev/null
}


# =============================================================================
# Entry point
# =============================================================================
[[ $# -lt 1 ]] && usage

MODE="$1"; shift

case "$MODE" in
    record)
        require_arg "${1:-}" "object_name"
        require_arg "${2:-}" "scene_name"
        require_arg "${3:-}" "num_episodes"
        OBJ="$1"; SCN="$2"; NEP="$3"; shift 3
        do_record "$OBJ" "$SCN" "$NEP" "$@"
        ;;
    convert|convert_hdf5_to_lerobot)
        require_arg "${1:-}" "object_name"
        require_arg "${2:-}" "scene_name"
        require_arg "${3:-}" "num_episodes"
        OBJ="$1"; SCN="$2"; NEP="$3"; shift 3
        do_convert "$OBJ" "$SCN" "$NEP" "$@"
        ;;
    train)
        require_arg "${1:-}" "policy"
        require_arg "${2:-}" "object_name"
        require_arg "${3:-}" "scene_name"
        require_arg "${4:-}" "num_episodes"
        POL="$1"; OBJ="$2"; SCN="$3"; NEP="$4"; shift 4
        do_train "$POL" "$OBJ" "$SCN" "$NEP" "$@"
        ;;
    eval)
        require_arg "${1:-}" "policy"
        require_arg "${2:-}" "object_name"
        require_arg "${3:-}" "scene_name"
        require_arg "${4:-}" "num_episodes"
        POL="$1"; OBJ="$2"; SCN="$3"; NEP="$4"; shift 4
        do_eval "$POL" "$OBJ" "$SCN" "$NEP" "$@"
        ;;
    *)
        echo "Error: Unknown mode '$MODE'" >&2
        usage
        ;;
esac
