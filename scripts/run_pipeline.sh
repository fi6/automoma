#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Unified pipeline for the lerobot-arena project.
#
# Supports: plan, record, convert, train, eval, debug
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ISAACLAB_ARENA="$REPO_ROOT/third_party/IsaacLab-Arena"
ROBOTWIN_ROOT="$REPO_ROOT/third_party/RoboTwin"

export AUTOMOMA_OBJECT_ROOT="$REPO_ROOT/assets/object"
export AUTOMOMA_SCENE_ROOT="$REPO_ROOT/assets/scene/infinigen/kitchen_1130"
export AUTOMOMA_ROBOT_ROOT="$REPO_ROOT/assets/robot"

echo "Using AUTOMOMA_OBJECT_ROOT=$AUTOMOMA_OBJECT_ROOT"
echo "Using AUTOMOMA_SCENE_ROOT=$AUTOMOMA_SCENE_ROOT"
echo "Using AUTOMOMA_ROBOT_ROOT=$AUTOMOMA_ROBOT_ROOT"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_pipeline.sh plan    <object_id> <scene_name> <split> [overrides...]
  bash scripts/run_pipeline.sh record  <object_name> <scene_name> <num_ep> [--headless|--no-headless] [overrides...]
  bash scripts/run_pipeline.sh convert <benchmark> <object_name> <scene_name> <num_ep> [--policy=POLICY] [overrides...]
  bash scripts/run_pipeline.sh train   <benchmark> <policy> <object_name> <scene_name> <num_ep> [overrides...]
  bash scripts/run_pipeline.sh eval    <benchmark> <policy> <object_name> <scene_name> <num_ep> [--headless|--no-headless] [overrides...]
  bash scripts/run_pipeline.sh debug   <object_name> <scene_name> [overrides...]

Benchmarks:
  lerobot
  robotwin
EOF
    exit 1
}

timestamp() { date +"%Y%m%d_%H%M%S"; }
mk_exp_name() { echo "summit_franka_open-${1}-${2}-${3}"; }
require_arg() {
    if [[ -z "${1:-}" ]]; then
        echo "Error: Missing required argument: $2" >&2
        usage
    fi
}

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

normalize_benchmark() {
    local benchmark="$1"
    case "${benchmark,,}" in
        lerobot) echo "lerobot" ;;
        robotwin) echo "robotwin" ;;
        *)
            echo "Error: Unsupported benchmark '$benchmark'. Use lerobot or robotwin." >&2
            exit 1
            ;;
    esac
}

normalize_policy() {
    echo "$1" | tr '[:upper:]' '[:lower:]'
}

capitalize_policy() {
    case "$(normalize_policy "$1")" in
        dp3) echo "DP3" ;;
        dp) echo "DP" ;;
        act) echo "ACT" ;;
        *) echo "$1" ;;
    esac
}

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

do_record() {
    local object_name="$1"; shift
    local scene_name="$1"; shift
    local num_episodes="$1"; shift

    local name; name="$(mk_exp_name "$object_name" "$scene_name" "$num_episodes")"
    local traj_file="$REPO_ROOT/data/trajs/summit_franka/${object_name}/${scene_name}/train/traj_data_train.pt"
    local dataset_file="$REPO_ROOT/data/automoma/${name}.hdf5"
    local headless_flag=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --headless) headless_flag="--headless"; shift ;;
            --no-headless) headless_flag=""; shift ;;
            --traj_file=*) traj_file="${1#*=}"; shift ;;
            --traj_file) traj_file="$2"; shift 2 ;;
            --dataset_file=*) dataset_file="${1#*=}"; shift ;;
            --dataset_file) dataset_file="$2"; shift 2 ;;
            *) break ;;
        esac
    done

    setup_log record "$object_name" "$scene_name"

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
        "$@"
        summit_franka_open_door
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

do_debug() {
    local object_name="$1"; shift
    local scene_name="$1"; shift
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
        exit 1
    fi
    if [[ ! -f "$debug_file" ]]; then
        echo "Error: debug file not found: $debug_file" >&2
        exit 1
    fi

    local -a cmd=(
        python isaaclab_arena/scripts/debug_automoma_demos.py
        "${overrides[@]}"
        summit_franka_open_door
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

do_convert() {
    local benchmark="$(normalize_benchmark "$1")"; shift
    local object_name="$1"; shift
    local scene_name="$1"; shift
    local num_episodes="$1"; shift

    local name; name="$(mk_exp_name "$object_name" "$scene_name" "$num_episodes")"
    local policy=""
    local data_root="$REPO_ROOT/data/automoma"
    local hdf5_name="${name}.hdf5"
    local repo_id="automoma/${name}"
    local output_dir="$REPO_ROOT/data/lerobot/automoma/${name}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --policy=*) policy="${1#*=}"; shift ;;
            --policy) policy="$2"; shift 2 ;;
            --data_root=*) data_root="${1#*=}"; shift ;;
            --data_root) data_root="$2"; shift 2 ;;
            --hdf5_name=*) hdf5_name="${1#*=}"; shift ;;
            --hdf5_name) hdf5_name="$2"; shift 2 ;;
            --repo_id=*) repo_id="${1#*=}"; shift ;;
            --repo_id) repo_id="$2"; shift 2 ;;
            --output_dir=*) output_dir="${1#*=}"; shift ;;
            --output_dir) output_dir="$2"; shift 2 ;;
            *) break ;;
        esac
    done

    setup_log convert "$object_name" "$scene_name"

    if [[ "$benchmark" == "lerobot" ]]; then
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
        return
    fi

    if [[ -z "$policy" ]]; then
        echo "Error: robotwin convert requires --policy." >&2
        exit 1
    fi

    local policy_name_upper; policy_name_upper="$(capitalize_policy "$policy")"
    local hdf5_path="$data_root/$hdf5_name"
    local robotwin_policy_dir="$ROBOTWIN_ROOT/policy/$policy_name_upper"

    if [[ ! -d "$robotwin_policy_dir" ]]; then
        echo "Error: RoboTwin policy directory not found: $robotwin_policy_dir" >&2
        exit 1
    fi

    local -a cmd=(
        bash process_data.sh
        "$object_name"
        "$scene_name"
        "$num_episodes"
        --input_hdf5 "$hdf5_path"
        "$@"
    )

    echo "Command: ${cmd[*]}"
    echo ""
    pushd "$robotwin_policy_dir" > /dev/null
    run_logged "${cmd[@]}"
    popd > /dev/null
}

do_train() {
    local benchmark="$(normalize_benchmark "$1")"; shift
    local policy_raw="$1"; shift
    local object_name="$1"; shift
    local scene_name="$1"; shift
    local num_episodes="$1"; shift

    local policy="$(normalize_policy "$policy_raw")"
    local name; name="$(mk_exp_name "$object_name" "$scene_name" "$num_episodes")"

    if [[ "$benchmark" == "lerobot" ]]; then
        local dataset_repo_id="$name"
        local dataset_root="$REPO_ROOT/data/lerobot/automoma/${name}"
        local output_dir="$REPO_ROOT/outputs/train/lerobot/${policy}_${name}"
        local job_name="${policy}_${name}"
        local force_overwrite="${FORCE_TRAIN_OVERWRITE:-0}"
        local train_steps="100000"
        local batch_size="8"

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --dataset.repo_id=*) dataset_repo_id="${1#*=}"; shift ;;
                --dataset.repo_id) dataset_repo_id="$2"; shift 2 ;;
                --dataset.root=*) dataset_root="${1#*=}"; shift ;;
                --dataset.root) dataset_root="$2"; shift 2 ;;
                --output_dir=*) output_dir="${1#*=}"; shift ;;
                --output_dir) output_dir="$2"; shift 2 ;;
                --job_name=*) job_name="${1#*=}"; shift ;;
                --job_name) job_name="$2"; shift 2 ;;
                --steps=*) train_steps="${1#*=}"; shift ;;
                --steps) train_steps="$2"; shift 2 ;;
                --batch_size=*) batch_size="${1#*=}"; shift ;;
                --batch_size) batch_size="$2"; shift 2 ;;
                *) break ;;
            esac
        done

        setup_log train "$object_name" "$scene_name"

        if [[ -d "$output_dir" ]]; then
            echo "Warning: Output directory already exists: $output_dir"
            if [[ "$force_overwrite" == "1" ]]; then
                rm -rf "$output_dir"
            fi
        fi

        local eval_freq=$(( train_steps / 100 ))
        local save_freq=$(( train_steps / 10 ))
        (( eval_freq < 1 )) && eval_freq=1
        (( save_freq < 1 )) && save_freq=1

        local -a cmd=(
            lerobot-train
            --policy.type="$policy"
            --batch_size="$batch_size"
            --steps="$train_steps"
            --log_freq=50
            --eval_freq="$eval_freq"
            --save_freq="$save_freq"
            --job_name="$job_name"
            --dataset.repo_id="$dataset_repo_id"
            --dataset.root="$dataset_root"
            --policy.optimizer_lr=1e-4
            --policy.push_to_hub=false
            --policy.device=cuda
            --wandb.enable=true
            --output_dir="$output_dir"
            --dataset.video_backend=pyav
        )

        if [[ "$policy" == "diffusion" ]]; then
            cmd+=(--policy.horizon=16 --policy.n_action_steps=16 --policy.n_obs_steps=2)
        else
            cmd+=(--policy.chunk_size=16 --policy.n_action_steps=16)
        fi

        cmd+=("$@")
        echo "Command: ${cmd[*]}"
        echo ""
        run_logged "${cmd[@]}"
        return
    fi

    local policy_name_upper; policy_name_upper="$(capitalize_policy "$policy")"
    local seed="42"
    local gpu_id="0"
    local ckpt_setting="$scene_name"
    local output_dir="$REPO_ROOT/outputs/train/robotwin/${policy}_${name}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --seed=*) seed="${1#*=}"; shift ;;
            --seed) seed="$2"; shift 2 ;;
            --gpu_id=*) gpu_id="${1#*=}"; shift ;;
            --gpu_id) gpu_id="$2"; shift 2 ;;
            --ckpt_setting=*) ckpt_setting="${1#*=}"; shift ;;
            --ckpt_setting) ckpt_setting="$2"; shift 2 ;;
            --output_dir=*) output_dir="${1#*=}"; shift ;;
            --output_dir) output_dir="$2"; shift 2 ;;
            *) break ;;
        esac
    done

    setup_log train "$object_name" "$scene_name"

    local train_wrapper="$REPO_ROOT/scripts/robotwin_train.sh"
    local -a cmd=(
        bash "$train_wrapper"
        "$policy_name_upper"
        "$object_name"
        "$scene_name"
        "$num_episodes"
        "$ckpt_setting"
        "$seed"
        "$gpu_id"
        "$output_dir"
        "$@"
    )

    echo "Command: ${cmd[*]}"
    echo ""
    pushd "$REPO_ROOT" > /dev/null
    run_logged "${cmd[@]}"
    popd > /dev/null
}

do_eval() {
    local benchmark="$(normalize_benchmark "$1")"; shift
    local policy_raw="$1"; shift
    local object_name="$1"; shift
    local scene_name="$1"; shift
    local num_episodes="$1"; shift

    local policy="$(normalize_policy "$policy_raw")"
    local name; name="$(mk_exp_name "$object_name" "$scene_name" "$num_episodes")"

    if [[ "$benchmark" == "lerobot" ]]; then
        local policy_path="$REPO_ROOT/outputs/train/lerobot/${policy}_${name}/checkpoints/last/pretrained_model"
        local output_dir="$REPO_ROOT/outputs/eval/lerobot/${policy}_${name}"
        local traj_file="$REPO_ROOT/data/trajs/summit_franka/${object_name}/${scene_name}/test/traj_data_test.pt"
        local traj_seed="${EVAL_TRAJ_SEED:-42}"
        local env_headless="false"

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --headless) env_headless="true"; shift ;;
                --no-headless) env_headless="false"; shift ;;
                --policy.path=*) policy_path="${1#*=}"; shift ;;
                --policy.path) policy_path="$2"; shift 2 ;;
                --traj_file=*) traj_file="${1#*=}"; shift ;;
                --traj_file) traj_file="$2"; shift 2 ;;
                --traj_seed=*) traj_seed="${1#*=}"; shift ;;
                --traj_seed) traj_seed="$2"; shift 2 ;;
                --output_dir=*) output_dir="${1#*=}"; shift ;;
                --output_dir) output_dir="$2"; shift 2 ;;
                *) break ;;
            esac
        done

        setup_log eval "$object_name" "$scene_name"

        rm -f "$output_dir/per_episode_results.csv" "$output_dir/eval_info.json"
        rm -rf "$output_dir/videos"

        local openness_threshold="${OPENNESS_THRESHOLD:-0.3}"
        local proximity_threshold="${PROXIMITY_THRESHOLD:-0.12}"
        local proximity_window_steps="${PROXIMITY_WINDOW_STEPS:-8}"
        local proximity_required_steps="${PROXIMITY_REQUIRED_STEPS:-5}"
        local use_fingertips="${USE_FINGERTIP_PROXIMITY:-true}"
        local disable_fingertip_proximity="false"
        [[ "$use_fingertips" == "false" ]] && disable_fingertip_proximity="true"

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
            --output_dir="$output_dir"
            "$@"
        )

        echo "Command: ${cmd[*]}"
        echo ""
        pushd "$ISAACLAB_ARENA" > /dev/null
        run_logged "${cmd[@]}"
        popd > /dev/null

        if [[ -f "$output_dir/per_episode_results.csv" && ! -f "$output_dir/eval_info.json" ]]; then
            python - "$output_dir" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
rows = list(csv.DictReader((output_dir / "per_episode_results.csv").open()))
sum_rewards = [float(row["sum_reward"]) for row in rows]
max_rewards = [float(row["max_reward"]) for row in rows]
successes = [str(row["success"]).lower() == "true" for row in rows]
seeds = [int(row["seed"]) if row.get("seed") else None for row in rows]
video_paths = [row.get("video_path", "") for row in rows if row.get("video_path")]

eval_info = {
    "per_episode": [
        {
            "episode_ix": int(row["episode_ix"]),
            "sum_reward": sum_rewards[i],
            "max_reward": max_rewards[i],
            "success": successes[i],
            "seed": seeds[i],
        }
        for i, row in enumerate(rows)
    ],
    "aggregated": {
        "avg_sum_reward": float(sum(sum_rewards) / len(sum_rewards)) if sum_rewards else 0.0,
        "avg_max_reward": float(sum(max_rewards) / len(max_rewards)) if max_rewards else 0.0,
        "pc_success": float(sum(successes) / len(successes) * 100) if successes else 0.0,
    },
}
if video_paths:
    eval_info["video_paths"] = video_paths
(output_dir / "eval_info.json").write_text(json.dumps(eval_info, indent=2))
PY
        fi
        return
    fi

    setup_log eval "$object_name" "$scene_name"
    local policy_name_upper; policy_name_upper="$(capitalize_policy "$policy")"
    local seed="42"
    local gpu_id="0"
    local ckpt_setting="$scene_name"
    local checkpoint_root="$REPO_ROOT/outputs/train/robotwin/${policy}_${name}"
    local output_dir="$REPO_ROOT/outputs/eval/robotwin/${policy}_${name}"
    local -a passthrough=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --seed=*) seed="${1#*=}"; shift ;;
            --seed) seed="$2"; shift 2 ;;
            --gpu_id=*) gpu_id="${1#*=}"; shift ;;
            --gpu_id) gpu_id="$2"; shift 2 ;;
            --ckpt_setting=*) ckpt_setting="${1#*=}"; shift ;;
            --ckpt_setting) ckpt_setting="$2"; shift 2 ;;
            --checkpoint_root=*) checkpoint_root="${1#*=}"; shift ;;
            --checkpoint_root) checkpoint_root="$2"; shift 2 ;;
            --output_dir=*) output_dir="${1#*=}"; shift ;;
            --output_dir) output_dir="$2"; shift 2 ;;
            *) passthrough+=("$1"); shift ;;
        esac
    done

    local eval_wrapper="$REPO_ROOT/scripts/robotwin_eval.sh"
    local -a cmd=(
        bash "$eval_wrapper"
        "$policy_name_upper"
        "$object_name"
        "$scene_name"
        "$num_episodes"
        "$ckpt_setting"
        "$seed"
        "$gpu_id"
        "$checkpoint_root"
        "$output_dir"
        "${passthrough[@]}"
    )

    echo "Command: ${cmd[*]}"
    echo ""
    pushd "$REPO_ROOT" > /dev/null
    run_logged "${cmd[@]}"
    popd > /dev/null
}

[[ $# -lt 1 ]] && usage
MODE="$1"; shift

case "$MODE" in
    plan)
        require_arg "${1:-}" "object_id"
        require_arg "${2:-}" "scene_name"
        require_arg "${3:-}" "split"
        do_plan "$1" "$2" "$3" "${@:4}"
        ;;
    record)
        require_arg "${1:-}" "object_name"
        require_arg "${2:-}" "scene_name"
        require_arg "${3:-}" "num_episodes"
        do_record "$1" "$2" "$3" "${@:4}"
        ;;
    convert|convert_hdf5_to_lerobot)
        require_arg "${1:-}" "benchmark"
        require_arg "${2:-}" "object_name"
        require_arg "${3:-}" "scene_name"
        require_arg "${4:-}" "num_episodes"
        do_convert "$1" "$2" "$3" "$4" "${@:5}"
        ;;
    train)
        require_arg "${1:-}" "benchmark"
        require_arg "${2:-}" "policy"
        require_arg "${3:-}" "object_name"
        require_arg "${4:-}" "scene_name"
        require_arg "${5:-}" "num_episodes"
        do_train "$1" "$2" "$3" "$4" "$5" "${@:6}"
        ;;
    eval)
        require_arg "${1:-}" "benchmark"
        require_arg "${2:-}" "policy"
        require_arg "${3:-}" "object_name"
        require_arg "${4:-}" "scene_name"
        require_arg "${5:-}" "num_episodes"
        do_eval "$1" "$2" "$3" "$4" "$5" "${@:6}"
        ;;
    debug)
        require_arg "${1:-}" "object_name"
        require_arg "${2:-}" "scene_name"
        do_debug "$1" "$2" "${@:3}"
        ;;
    *)
        echo "Error: Unknown mode '$MODE'" >&2
        usage
        ;;
esac
