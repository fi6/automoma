#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Unified pipeline for the lerobot-arena project.
#
# Supports: plan, record, convert, train, eval, debug
#
# Usage:
#   bash scripts/run_pipeline.sh <mode> [mode-specific args] [overrides...]
#
#   plan:
#     bash scripts/run_pipeline.sh plan <object_id> <scene_name> <split> [overrides...]
#     e.g.  bash scripts/run_pipeline.sh plan 7221 scene_0_seed_0 train
#           bash scripts/run_pipeline.sh plan 7221 scene_0_seed_0 test planner.traj.batch_size=16
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
#   debug:
#     bash scripts/run_pipeline.sh debug <object_name> <scene_name> [overrides...]
#     e.g.  bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train/grasp_0001/traj_data.pt
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
ISAACLAB_ARENA="$REPO_ROOT/third_party/IsaacLab-Arena"

# ---------------------------------------------------------------------------
# Export env vars so IsaacLab-Arena resolves assets from lerobot-arena
# ---------------------------------------------------------------------------
export AUTOMOMA_OBJECT_ROOT="$REPO_ROOT/assets/object"
export AUTOMOMA_SCENE_ROOT="$REPO_ROOT/assets/scene/infinigen/kitchen_1130"
export AUTOMOMA_ROBOT_ROOT="$REPO_ROOT/assets/robot"

echo "Using AUTOMOMA_OBJECT_ROOT=$AUTOMOMA_OBJECT_ROOT"
echo "Using AUTOMOMA_SCENE_ROOT=$AUTOMOMA_SCENE_ROOT"
echo "Using AUTOMOMA_ROBOT_ROOT=$AUTOMOMA_ROBOT_ROOT"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_pipeline.sh plan    <object_id> <scene_name> <split> [overrides...]
  bash scripts/run_pipeline.sh record  <object_name> <scene_name> <num_ep>  [--headless|--no-headless] [overrides...]
  bash scripts/run_pipeline.sh convert <object_name> <scene_name> <num_ep>  [overrides...]
  bash scripts/run_pipeline.sh train   <policy> <object_name> <scene_name> <num_ep> [overrides...]
  bash scripts/run_pipeline.sh eval    <policy> <object_name> <scene_name> <num_ep> [--headless|--no-headless] [overrides...]
  bash scripts/run_pipeline.sh debug   <object_name> <scene_name> [overrides...]

Examples:
  bash scripts/run_pipeline.sh plan 7221 scene_0_seed_0 train
  bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 30 --headless
  bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 30 --headless --interpolated 2
  bash scripts/run_pipeline.sh convert microwave_7221 scene_0_seed_0 30
  bash scripts/run_pipeline.sh train act microwave_7221 scene_0_seed_0 30 --steps=20000
  bash scripts/run_pipeline.sh eval  act microwave_7221 scene_0_seed_0 30 --headless --eval.n_episodes=50
  bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 --debug_file path/to/traj_data.pt
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
# Mode: plan
# ---------------------------------------------------------------------------
do_plan() {
    local object_id="$1"; shift
    local scene_name="$1"; shift
    local split="$1"; shift

    setup_log plan "$object_id" "$scene_name"

    local -a cmd=(
        python scripts/plan.py
        "scene_name=${scene_name}"
        "object_id=${object_id}"
        "mode=${split}"
        "$@"
    )

    echo "Command: ${cmd[*]}"
    echo ""

    pushd "$REPO_ROOT" > /dev/null
    run_logged "${cmd[@]}"
    popd > /dev/null
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
    local traj_file="$REPO_ROOT/data/trajs/summit_franka/${object_name}/${scene_name}/train/traj_data_train.pt"
    local dataset_file="$REPO_ROOT/data/automoma/${name}.hdf5"
    local headless_flag=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --headless)
                headless_flag="--headless"
                shift
                ;;
            --no-headless)
                headless_flag=""
                shift
                ;;
            --traj_file=*)
                traj_file="${1#*=}"
                shift
                ;;
            --traj_file)
                traj_file="$2"
                shift 2
                ;;
            --dataset_file=*)
                dataset_file="${1#*=}"
                shift
                ;;
            --dataset_file)
                dataset_file="$2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done

    setup_log record "$object_name" "$scene_name"

    # Build command.  Overrides go before the subcommand so argparse
    # treats them as top-level flags (e.g. --headless, --interpolated, --device).
    local -a cmd=(
        python isaaclab_arena/scripts/record_automoma_demos.py
        --enable_cameras
        --mobile_base_relative
        --traj_file "$traj_file"
        --dataset_file "$dataset_file"
        --num_episodes "$num_episodes"
    )

    if [[ -n "$headless_flag" ]]; then
        cmd+=("$headless_flag")
    fi

    cmd+=(
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
# Mode: debug
# ---------------------------------------------------------------------------
do_debug() {
    local object_name="$1"; shift
    local scene_name="$1"; shift
    # $@ = overrides (should include --debug_file)

    # Normalize --debug_file to an absolute path from repo root so it still
    # resolves correctly after we pushd into IsaacLab-Arena.
    local -a overrides=("$@")
    local debug_file=""
    local i
    for ((i = 0; i < ${#overrides[@]}; i++)); do
        if [[ "${overrides[$i]}" == "--debug_file" ]]; then
            if (( i + 1 < ${#overrides[@]} )); then
                debug_file="${overrides[$((i + 1))]}"
                if [[ "$debug_file" != /* ]]; then
                    debug_file="$REPO_ROOT/$debug_file"
                    overrides[$((i + 1))]="$debug_file"
                fi
            fi
            break
        elif [[ "${overrides[$i]}" == --debug_file=* ]]; then
            debug_file="${overrides[$i]#--debug_file=}"
            if [[ "$debug_file" != /* ]]; then
                debug_file="$REPO_ROOT/$debug_file"
                overrides[$i]="--debug_file=$debug_file"
            fi
            break
        fi
    done

    if [[ -z "$debug_file" ]]; then
        echo "Error: debug mode requires --debug_file <path/to/*.pt>." >&2
        echo "Example:" >&2
        echo "  bash scripts/run_pipeline.sh debug $object_name $scene_name --debug_file data/trajs/summit_franka/$object_name/$scene_name/train/grasp_0001/traj_data.pt" >&2
        exit 1
    fi

    if [[ ! -f "$debug_file" ]]; then
        echo "Error: debug file not found: $debug_file" >&2
        local candidate_root="$REPO_ROOT/data/trajs/summit_franka/$object_name/$scene_name/train"
        if [[ -d "$candidate_root" ]]; then
            echo "Available .pt files under $candidate_root:" >&2
            find "$candidate_root" -maxdepth 3 -type f -name "*.pt" | sort >&2
        fi
        exit 1
    fi

    local -a cmd=(
        python isaaclab_arena/scripts/debug_automoma_demos.py
        "${overrides[@]}"                 # ← overrides (before subcommand)
        summit_franka_open_door           # subcommand
        --object_name "$object_name"
        --scene_name "$scene_name"
        --object_center
    )

    echo "Command: ${cmd[*]}"
    echo ""

    pushd "$ISAACLAB_ARENA" > /dev/null
    "${cmd[@]}"
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
    local data_root="$REPO_ROOT/data/automoma"
    local hdf5_name="${name}.hdf5"
    local repo_id="automoma/${name}"
    local output_dir="$REPO_ROOT/data/lerobot/automoma/${name}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --data_root=*)
                data_root="${1#*=}"
                shift
                ;;
            --data_root)
                data_root="$2"
                shift 2
                ;;
            --hdf5_name=*)
                hdf5_name="${1#*=}"
                shift
                ;;
            --hdf5_name)
                hdf5_name="$2"
                shift 2
                ;;
            --repo_id=*)
                repo_id="${1#*=}"
                shift
                ;;
            --repo_id)
                repo_id="$2"
                shift 2
                ;;
            --output_dir=*)
                output_dir="${1#*=}"
                shift
                ;;
            --output_dir)
                output_dir="$2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done

    setup_log convert "$object_name" "$scene_name"

    local -a cmd=(
        python isaaclab_arena_gr00t/data_utils/convert_hdf5_to_lerobot_v30.py
        --yaml_file isaaclab_arena_gr00t/config/summit_franka_manip_config.yaml
        --data_root "$data_root"
        --hdf5_name "$hdf5_name"
        --repo_id "$repo_id"
        --output_dir "$output_dir"
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
    local dataset_repo_id="$name"
    local dataset_root="$REPO_ROOT/data/lerobot/automoma/${name}"
    local output_dir="$REPO_ROOT/outputs/train/${policy}_${name}"
    local job_name="${policy}_${name}"
    local force_overwrite="${FORCE_TRAIN_OVERWRITE:-0}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dataset.repo_id=*)
                dataset_repo_id="${1#*=}"
                shift
                ;;
            --dataset.repo_id)
                dataset_repo_id="$2"
                shift 2
                ;;
            --dataset.root=*)
                dataset_root="${1#*=}"
                shift
                ;;
            --dataset.root)
                dataset_root="$2"
                shift 2
                ;;
            --output_dir=*)
                output_dir="${1#*=}"
                shift
                ;;
            --output_dir)
                output_dir="$2"
                shift 2
                ;;
            --job_name=*)
                job_name="${1#*=}"
                shift
                ;;
            --job_name)
                job_name="$2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done

    setup_log train "$object_name" "$scene_name"

    if [[ -d "$output_dir" ]]; then
        echo "Warning: Output directory already exists:"
        echo "  $output_dir"
        if [[ "$force_overwrite" == "1" ]]; then
            rm -rf "$output_dir"
            echo "Deleted because FORCE_TRAIN_OVERWRITE=1"
        else
            read -r -p "Delete and retrain? [y/N] " answer
            if [[ "${answer,,}" == "y" ]]; then
                rm -rf "$output_dir"
                echo "Deleted."
            else
                echo "Keeping existing directory."
            fi
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
        --job_name="$job_name"
        --dataset.repo_id="$dataset_repo_id"
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
    local traj_file="$REPO_ROOT/data/trajs/summit_franka/${object_name}/${scene_name}/test/traj_data_test.pt"
    local traj_seed="${EVAL_TRAJ_SEED:-42}"
    local env_headless="false"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --headless)
                env_headless="true"
                shift
                ;;
            --no-headless)
                env_headless="false"
                shift
                ;;
            --policy.path=*)
                policy_path="${1#*=}"
                shift
                ;;
            --policy.path)
                policy_path="$2"
                shift 2
                ;;
            --traj_file=*)
                traj_file="${1#*=}"
                shift
                ;;
            --traj_file)
                traj_file="$2"
                shift 2
                ;;
            --traj_seed=*)
                traj_seed="${1#*=}"
                shift
                ;;
            --traj_seed)
                traj_seed="$2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done

    setup_log eval "$object_name" "$scene_name"

    if [[ ! -d "$policy_path" ]]; then
        echo "Error: policy checkpoint not found: $policy_path" >&2
        exit 1
    fi

    if [[ ! -f "$traj_file" ]]; then
        echo "Error: eval trajectory file not found: $traj_file" >&2
        local candidate_root="$REPO_ROOT/data/trajs/summit_franka/$object_name/$scene_name"
        if [[ -d "$candidate_root" ]]; then
            echo "Available trajectory files under $candidate_root:" >&2
            find "$candidate_root" -maxdepth 3 -type f -name "traj_data_*.pt" | sort >&2
        fi
        exit 1
    fi

    local openness_threshold="${OPENNESS_THRESHOLD:-0.3}"
    local proximity_threshold="${PROXIMITY_THRESHOLD:-0.12}"
    local proximity_window_steps="${PROXIMITY_WINDOW_STEPS:-8}"
    local proximity_required_steps="${PROXIMITY_REQUIRED_STEPS:-5}"
    local use_fingertips="${USE_FINGERTIP_PROXIMITY:-true}"
    local disable_fingertip_proximity="false"
    if [[ "$use_fingertips" == "false" ]]; then
        disable_fingertip_proximity="true"
    fi

    local debug_visualize_handle="${DEBUG_VISUALIZE_HANDLE:-false}"
    local debug_record_handle_diagnostics="${DEBUG_RECORD_HANDLE_DIAGNOSTICS:-false}"
    local debug_marker_scale="${DEBUG_MARKER_SCALE:-1.0}"

    local env_kwargs="{\"object_name\": \"${object_name}\", \"scene_name\": \"${scene_name}\", \"object_center\": true, \"mobile_base_relative\": true, \"traj_file\": \"${traj_file}\", \"traj_seed\": ${traj_seed}, \"openness_threshold\": ${openness_threshold}, \"proximity_threshold\": ${proximity_threshold}, \"proximity_window_steps\": ${proximity_window_steps}, \"proximity_required_steps\": ${proximity_required_steps}, \"disable_fingertip_proximity\": ${disable_fingertip_proximity}, \"debug_visualize_handle\": ${debug_visualize_handle}, \"debug_record_handle_diagnostics\": ${debug_record_handle_diagnostics}, \"debug_marker_scale\": ${debug_marker_scale}}"

    local rename_map='{"observation.images.ego_topdown_rgb": "observation.images.ego_topdown", "observation.images.ego_wrist_rgb": "observation.images.ego_wrist", "observation.images.fix_local_rgb": "observation.images.fix_local"}'

    local -a cmd=(
        lerobot-eval
        --policy.path="$policy_path"
        --policy.device=cuda
        --env.type=isaaclab_arena
        --env.hub_path="$ISAACLAB_ARENA/isaaclab-arena-envs"
        --env.environment=summit_franka_open_door_eval
        "--env.headless=${env_headless}"
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
    plan)
        require_arg "${1:-}" "object_id"
        require_arg "${2:-}" "scene_name"
        require_arg "${3:-}" "split"
        OBJ_ID="$1"; SCN="$2"; SPLIT="$3"; shift 3
        do_plan "$OBJ_ID" "$SCN" "$SPLIT" "$@"
        ;;
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
    debug)
        require_arg "${1:-}" "object_name"
        require_arg "${2:-}" "scene_name"
        OBJ="$1"; SCN="$2"; shift 2
        do_debug "$OBJ" "$SCN" "$@"
        ;;
    *)
        echo "Error: Unknown mode '$MODE'" >&2
        usage
        ;;
esac
