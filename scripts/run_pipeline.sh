#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Unified pipeline for the lerobot-arena project.
#
# Supports: plan, record, replay, convert, train, eval, record_dataset_eval, debug
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ISAACLAB_ARENA="$REPO_ROOT/third_party/IsaacLab-Arena"
ROBOTWIN_ROOT="$REPO_ROOT/third_party/RoboTwin"

export AUTOMOMA_OBJECT_ROOT="$REPO_ROOT/assets/object"
export AUTOMOMA_SCENE_ROOT="$REPO_ROOT/assets/scene/infinigen/kitchen_1130"
export AUTOMOMA_ROBOT_ROOT="$REPO_ROOT/assets/robot"
export AUTOMOMA_ROBOT_OBJECT_STATIC_FRICTION="${AUTOMOMA_ROBOT_OBJECT_STATIC_FRICTION:-1.0}"
export AUTOMOMA_ROBOT_OBJECT_DYNAMIC_FRICTION="${AUTOMOMA_ROBOT_OBJECT_DYNAMIC_FRICTION:-1.0}"

# Isaac Sim loads Python extension dependencies through the process dynamic
# linker. Keep the conda runtime first so packages built against the conda C++
# runtime do not accidentally bind to the older system libstdc++.
if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$CONDA_PREFIX/lib:"*) ;;
        *) export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi

echo "Using AUTOMOMA_OBJECT_ROOT=$AUTOMOMA_OBJECT_ROOT"
echo "Using AUTOMOMA_SCENE_ROOT=$AUTOMOMA_SCENE_ROOT"
echo "Using AUTOMOMA_ROBOT_ROOT=$AUTOMOMA_ROBOT_ROOT"
echo "Using AUTOMOMA_ROBOT_OBJECT_STATIC_FRICTION=$AUTOMOMA_ROBOT_OBJECT_STATIC_FRICTION"
echo "Using AUTOMOMA_ROBOT_OBJECT_DYNAMIC_FRICTION=$AUTOMOMA_ROBOT_OBJECT_DYNAMIC_FRICTION"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_pipeline.sh plan    <object_id> <scene_name> <split> [overrides...]
  bash scripts/run_pipeline.sh record  <object_name> <scene_name> <num_ep> [--record_format full|episodes] [--headless|--no-headless] [--validate_record_success] [overrides...]
  bash scripts/run_pipeline.sh replay  <object_name> <scene_name> <num_ep> [--metrics [--metrics_file PATH]] [--headless|--no-headless] [overrides...]
  bash scripts/run_pipeline.sh convert <benchmark> <object_name> <scene_name> <num_ep> [--policy=POLICY] [overrides...]
  bash scripts/run_pipeline.sh train   <benchmark> <policy> <object_name> <scene_name> <num_ep> [overrides...]
  bash scripts/run_pipeline.sh eval    <benchmark> <policy> <object_name> <scene_name> <num_ep> [--headless|--no-headless] [overrides...]
  bash scripts/run_pipeline.sh record_dataset_eval <object_name> <scene_name> <num_ep> [--headless|--no-headless] [overrides...]
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
resolve_repo_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$REPO_ROOT/$path"
    fi
}
require_eval_traj_file() {
    local traj_file="$1"
    if [[ ! -f "$traj_file" ]]; then
        echo "Error: Eval trajectory file not found: $traj_file" >&2
        echo "Eval requires the source IK/test trajectory. Run planning first or pass --traj_file to an existing traj_data_test.pt." >&2
        exit 1
    fi
}
append_robot_object_friction_args() {
    local -n out_ref=$1
    if [[ -n "${AUTOMOMA_ROBOT_OBJECT_STATIC_FRICTION:-}" ]]; then
        out_ref+=(--robot_object_static_friction "$AUTOMOMA_ROBOT_OBJECT_STATIC_FRICTION")
    fi
    if [[ -n "${AUTOMOMA_ROBOT_OBJECT_DYNAMIC_FRICTION:-}" ]]; then
        out_ref+=(--robot_object_dynamic_friction "$AUTOMOMA_ROBOT_OBJECT_DYNAMIC_FRICTION")
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

    local arg
    for arg in "$@"; do
        case "$arg" in
            --no_record|--no-record)
                echo "Warning: record --no_record is deprecated; use replay instead. Routing to replay_automoma_demos.py." >&2
                do_replay "$object_name" "$scene_name" "$num_episodes" "$@"
                return
                ;;
        esac
    done

    local name; name="$(mk_exp_name "$object_name" "$scene_name" "$num_episodes")"
    local traj_file="$REPO_ROOT/data/trajs/summit_franka/${object_name}/${scene_name}/train/traj_data_train.pt"
    local dataset_file="$REPO_ROOT/data/automoma/${name}.hdf5"
    local dataset_file_explicit="false"
    local record_format="${RECORD_HDF5_FORMAT:-full}"
    local start_episode="${RECORD_START_EPISODE:-0}"
    local episode_file_start="${RECORD_EPISODE_FILE_START:-1}"
    local headless_flag=""
    local record_interpolated="${RECORD_INTERPOLATED:-5}"
    local record_interpolation_type="${RECORD_INTERPOLATION_TYPE:-cubic}"
    local record_decimation="${RECORD_DECIMATION:-1}"
    local record_init_steps="${RECORD_INIT_STEPS:-5}"
    local auto_debug_tracking="${RECORD_AUTO_DEBUG_TRACKING:-0}"
    local add_cameras="true"

    local -a record_overrides=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --headless) headless_flag="--headless"; shift ;;
            --no-headless) headless_flag=""; shift ;;
            --traj_file=*) traj_file="${1#*=}"; shift ;;
            --traj_file) traj_file="$2"; shift 2 ;;
            --dataset_file=*) dataset_file="${1#*=}"; dataset_file_explicit="true"; shift ;;
            --dataset_file) dataset_file="$2"; dataset_file_explicit="true"; shift 2 ;;
            --record_format=*) record_format="${1#*=}"; shift ;;
            --record_format) record_format="$2"; shift 2 ;;
            --hdf5_format=*) record_format="${1#*=}"; shift ;;
            --hdf5_format) record_format="$2"; shift 2 ;;
            --split_episodes) record_format="episodes"; shift ;;
            --single_hdf5) record_format="full"; shift ;;
            --start_episode=*) start_episode="${1#*=}"; shift ;;
            --start_episode) start_episode="$2"; shift 2 ;;
            --episode_file_start=*) episode_file_start="${1#*=}"; shift ;;
            --episode_file_start) episode_file_start="$2"; shift 2 ;;
            --disable_cameras|--disable-cameras)
                add_cameras="false"
                shift
                ;;
            *) record_overrides+=("$1"); shift ;;
        esac
    done
    set -- "${record_overrides[@]}"
    case "${record_format,,}" in
        full|merged|single) record_format="full" ;;
        episodes|split|per_episode|per-episode) record_format="episodes" ;;
        *)
            echo "Error: Unsupported record format '$record_format'. Use full or episodes." >&2
            exit 1
            ;;
    esac
    if ! [[ "$num_episodes" =~ ^[0-9]+$ ]] || (( num_episodes < 1 )); then
        echo "Error: num_episodes must be a positive integer." >&2
        exit 1
    fi
    if ! [[ "$start_episode" =~ ^[0-9]+$ ]]; then
        echo "Error: --start_episode must be a non-negative integer." >&2
        exit 1
    fi
    if ! [[ "$episode_file_start" =~ ^[0-9]+$ ]] || (( episode_file_start < 1 )); then
        echo "Error: --episode_file_start must be a positive integer." >&2
        exit 1
    fi
    if [[ "$record_format" == "episodes" && "$dataset_file_explicit" == "false" ]]; then
        dataset_file="$REPO_ROOT/data/automoma/${name}"
    fi
    if [[ "$dataset_file" != /* ]]; then
        dataset_file="$REPO_ROOT/$dataset_file"
    fi

    setup_log record "$object_name" "$scene_name"

    local add_default_interpolated="true"
    local add_default_interpolation_type="true"
    local add_default_decimation="true"
    local add_default_init_steps="true"
    local has_debug_joint_tracking="false"
    local has_debug_joint_tracking_steps="false"
    local has_debug_joint_tracking_interval="false"
    local has_debug_joint_tracking_topk="false"
    local has_debug_joint_tracking_fk_link="false"
    local arg
    for arg in "$@"; do
        case "$arg" in
            --interpolated|--interpolated=*)
                add_default_interpolated="false"
                ;;
            --interpolation_type|--interpolation_type=*)
                add_default_interpolation_type="false"
                ;;
            --decimation|--decimation=*)
                add_default_decimation="false"
                ;;
            --init_steps|--init_steps=*)
                add_default_init_steps="false"
                ;;
            --debug_joint_tracking)
                has_debug_joint_tracking="true"
                ;;
            --debug_joint_tracking_steps|--debug_joint_tracking_steps=*)
                has_debug_joint_tracking_steps="true"
                ;;
            --debug_joint_tracking_interval|--debug_joint_tracking_interval=*)
                has_debug_joint_tracking_interval="true"
                ;;
            --debug_joint_tracking_topk|--debug_joint_tracking_topk=*)
                has_debug_joint_tracking_topk="true"
                ;;
            --debug_joint_tracking_fk_link|--debug_joint_tracking_fk_link=*)
                has_debug_joint_tracking_fk_link="true"
                ;;
        esac
    done

    local -a recorder_args=(
        --mobile_base_relative
        --traj_file "$traj_file"
    )
    if [[ "$add_cameras" == "true" ]]; then
        recorder_args=(--enable_cameras "${recorder_args[@]}")
    fi

    if [[ -n "$headless_flag" ]]; then
        recorder_args+=("$headless_flag")
    fi
    if [[ "$add_default_interpolated" == "true" ]]; then
        recorder_args+=(--interpolated "$record_interpolated")
    fi
    if [[ "$add_default_interpolation_type" == "true" ]]; then
        recorder_args+=(--interpolation_type "$record_interpolation_type")
    fi
    if [[ "$add_default_decimation" == "true" && -n "$record_decimation" ]]; then
        recorder_args+=(--decimation "$record_decimation")
    fi
    if [[ "$add_default_init_steps" == "true" && -n "$record_init_steps" ]]; then
        recorder_args+=(--init_steps "$record_init_steps")
    fi
    if [[ "$auto_debug_tracking" == "1" ]]; then
        if [[ "$has_debug_joint_tracking" != "true" ]]; then
            recorder_args+=(--debug_joint_tracking)
        fi
        if [[ "$has_debug_joint_tracking_steps" != "true" ]]; then
            recorder_args+=(--debug_joint_tracking_steps 0)
        fi
        if [[ "$has_debug_joint_tracking_interval" != "true" ]]; then
            recorder_args+=(--debug_joint_tracking_interval 0)
        fi
        if [[ "$has_debug_joint_tracking_topk" != "true" ]]; then
            recorder_args+=(--debug_joint_tracking_topk 12)
        fi
        if [[ "$has_debug_joint_tracking_fk_link" != "true" ]]; then
            recorder_args+=(--debug_joint_tracking_fk_link ee_link)
        fi
    fi
    append_robot_object_friction_args recorder_args

    local -a task_args=(
        "$@"
        summit_franka_open_door
        --object_name "$object_name"
        --scene_name "$scene_name"
        --object_center
    )

    local -a cmd=()
    if [[ "$record_format" == "full" ]]; then
        cmd=(
            python isaaclab_arena/scripts/record_automoma_demos.py
            "${recorder_args[@]}"
            --dataset_file "$dataset_file"
            --num_episodes "$num_episodes"
            --start_episode "$start_episode"
            "${task_args[@]}"
        )
    else
        mkdir -p "$dataset_file"
        cmd=(
            python isaaclab_arena/scripts/record_automoma_demos.py
            "${recorder_args[@]}"
            --dataset_file "$dataset_file/episode_000001.hdf5"
            --num_episodes 1
            --start_episode "$start_episode"
            "${task_args[@]}"
        )
    fi

    echo "Record format: $record_format"
    echo "Command: ${cmd[*]}"
    echo ""

    pushd "$ISAACLAB_ARENA" > /dev/null
    if [[ "$record_format" == "full" ]]; then
        run_logged "${cmd[@]}"
    else
        local ep_idx
        for ((ep_idx = 0; ep_idx < num_episodes; ep_idx++)); do
            local episode_number=$((episode_file_start + ep_idx))
            local actual_start=$((start_episode + ep_idx))
            local episode_file
            printf -v episode_file "%s/episode_%06d.hdf5" "$dataset_file" "$episode_number"
            cmd=(
                python isaaclab_arena/scripts/record_automoma_demos.py
                "${recorder_args[@]}"
                --dataset_file "$episode_file"
                --num_episodes 1
                --start_episode "$actual_start"
                "${task_args[@]}"
            )
            echo ""
            echo "[SplitRecord] episode ${episode_number}/${episode_file_start}+${num_episodes}-1: $episode_file"
            echo "Command: ${cmd[*]}"
            run_logged "${cmd[@]}"
        done
    fi
    popd > /dev/null
}

do_replay() {
    local object_name="$1"; shift
    local scene_name="$1"; shift
    local num_episodes="$1"; shift

    local traj_file="$REPO_ROOT/data/trajs/summit_franka/${object_name}/${scene_name}/train/traj_data_train.pt"
    local start_episode="${REPLAY_START_EPISODE:-${RECORD_START_EPISODE:-0}}"
    local headless_flag=""
    local replay_interpolated="${REPLAY_INTERPOLATED:-${RECORD_INTERPOLATED:-5}}"
    local replay_interpolation_type="${REPLAY_INTERPOLATION_TYPE:-${RECORD_INTERPOLATION_TYPE:-cubic}}"
    local replay_decimation="${REPLAY_DECIMATION:-${RECORD_DECIMATION:-1}}"
    local replay_init_steps="${REPLAY_INIT_STEPS:-${RECORD_INIT_STEPS:-5}}"
    local auto_debug_tracking="${REPLAY_AUTO_DEBUG_TRACKING:-${RECORD_AUTO_DEBUG_TRACKING:-0}}"
    local add_cameras="false"
    local metrics_enabled="false"
    local metrics_file=""

    local -a replay_overrides=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --headless) headless_flag="--headless"; shift ;;
            --no-headless) headless_flag=""; shift ;;
            --traj_file=*) traj_file="${1#*=}"; shift ;;
            --traj_file) traj_file="$2"; shift 2 ;;
            --start_episode=*) start_episode="${1#*=}"; shift ;;
            --start_episode) start_episode="$2"; shift 2 ;;
            --metrics)
                metrics_enabled="true"
                if [[ $# -gt 1 && "$2" != --* ]]; then
                    metrics_file="$2"
                    shift 2
                else
                    shift
                fi
                ;;
            --metrics=*) metrics_enabled="true"; metrics_file="${1#*=}"; shift ;;
            --metrics_file=*) metrics_enabled="true"; metrics_file="${1#*=}"; shift ;;
            --metrics_file) metrics_enabled="true"; metrics_file="$2"; shift 2 ;;
            --episode_indices_file=*) replay_overrides+=(--episode_indices_file "$(resolve_repo_path "${1#*=}")"); shift ;;
            --episode_indices_file) replay_overrides+=(--episode_indices_file "$(resolve_repo_path "$2")"); shift 2 ;;
            --enable_cameras) add_cameras="true"; shift ;;
            --disable_cameras|--disable-cameras) add_cameras="false"; shift ;;
            --no_record|--no-record|--validate_record_success|--keep_failed_record_demos)
                shift
                ;;
            --dataset_file=*) shift ;;
            --dataset_file) shift 2 ;;
            --record_format=*|--hdf5_format=*|--split_episodes|--single_hdf5) shift ;;
            --record_format|--hdf5_format) shift 2 ;;
            *) replay_overrides+=("$1"); shift ;;
        esac
    done
    set -- "${replay_overrides[@]}"

    if ! [[ "$num_episodes" =~ ^[0-9]+$ ]] || (( num_episodes < 1 )); then
        echo "Error: num_episodes must be a positive integer." >&2
        exit 1
    fi
    if ! [[ "$start_episode" =~ ^[0-9]+$ ]]; then
        echo "Error: --start_episode must be a non-negative integer." >&2
        exit 1
    fi
    if [[ "$traj_file" != /* ]]; then
        traj_file="$REPO_ROOT/$traj_file"
    fi
    if [[ -n "$metrics_file" && "$metrics_file" != /* ]]; then
        metrics_file="$REPO_ROOT/$metrics_file"
    fi

    setup_log replay "$object_name" "$scene_name"

    local add_default_interpolated="true"
    local add_default_interpolation_type="true"
    local add_default_decimation="true"
    local add_default_init_steps="true"
    local has_debug_joint_tracking="false"
    local has_debug_joint_tracking_steps="false"
    local has_debug_joint_tracking_interval="false"
    local has_debug_joint_tracking_topk="false"
    local has_debug_joint_tracking_fk_link="false"
    local arg
    for arg in "$@"; do
        case "$arg" in
            --interpolated|--interpolated=*) add_default_interpolated="false" ;;
            --interpolation_type|--interpolation_type=*) add_default_interpolation_type="false" ;;
            --decimation|--decimation=*) add_default_decimation="false" ;;
            --init_steps|--init_steps=*) add_default_init_steps="false" ;;
            --debug_joint_tracking) has_debug_joint_tracking="true" ;;
            --debug_joint_tracking_steps|--debug_joint_tracking_steps=*) has_debug_joint_tracking_steps="true" ;;
            --debug_joint_tracking_interval|--debug_joint_tracking_interval=*) has_debug_joint_tracking_interval="true" ;;
            --debug_joint_tracking_topk|--debug_joint_tracking_topk=*) has_debug_joint_tracking_topk="true" ;;
            --debug_joint_tracking_fk_link|--debug_joint_tracking_fk_link=*) has_debug_joint_tracking_fk_link="true" ;;
        esac
    done

    local -a replay_args=(--traj_file "$traj_file")
    if [[ "$add_cameras" == "true" ]]; then
        replay_args=(--enable_cameras "${replay_args[@]}")
    fi
    if [[ -n "$headless_flag" ]]; then
        replay_args+=("$headless_flag")
    fi
    if [[ "$metrics_enabled" == "true" ]]; then
        replay_args+=(--metrics)
        if [[ -n "$metrics_file" ]]; then
            replay_args+=(--metrics_file "$metrics_file")
        fi
    fi
    if [[ "$add_default_interpolated" == "true" ]]; then
        replay_args+=(--interpolated "$replay_interpolated")
    fi
    if [[ "$add_default_interpolation_type" == "true" ]]; then
        replay_args+=(--interpolation_type "$replay_interpolation_type")
    fi
    if [[ "$add_default_decimation" == "true" && -n "$replay_decimation" ]]; then
        replay_args+=(--decimation "$replay_decimation")
    fi
    if [[ "$add_default_init_steps" == "true" && -n "$replay_init_steps" ]]; then
        replay_args+=(--init_steps "$replay_init_steps")
    fi
    if [[ "$auto_debug_tracking" == "1" ]]; then
        [[ "$has_debug_joint_tracking" != "true" ]] && replay_args+=(--debug_joint_tracking)
        [[ "$has_debug_joint_tracking_steps" != "true" ]] && replay_args+=(--debug_joint_tracking_steps 0)
        [[ "$has_debug_joint_tracking_interval" != "true" ]] && replay_args+=(--debug_joint_tracking_interval 0)
        [[ "$has_debug_joint_tracking_topk" != "true" ]] && replay_args+=(--debug_joint_tracking_topk 12)
        [[ "$has_debug_joint_tracking_fk_link" != "true" ]] && replay_args+=(--debug_joint_tracking_fk_link ee_link)
    fi
    append_robot_object_friction_args replay_args

    local -a task_args=(
        "$@"
        summit_franka_open_door
        --object_name "$object_name"
        --scene_name "$scene_name"
        --object_center
    )
    local -a cmd=(
        python isaaclab_arena/scripts/replay_automoma_demos.py
        "${replay_args[@]}"
        --num_episodes "$num_episodes"
        --start_episode "$start_episode"
        "${task_args[@]}"
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
    local use_rgb=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --policy=*) policy="${1#*=}"; shift ;;
            --policy) policy="$2"; shift 2 ;;
            --data_root=*) data_root="$(resolve_repo_path "${1#*=}")"; shift ;;
            --data_root) data_root="$(resolve_repo_path "$2")"; shift 2 ;;
            --hdf5_name=*) hdf5_name="${1#*=}"; shift ;;
            --hdf5_name) hdf5_name="$2"; shift 2 ;;
            --repo_id=*) repo_id="${1#*=}"; shift ;;
            --repo_id) repo_id="$2"; shift 2 ;;
            --output_dir=*) output_dir="$(resolve_repo_path "${1#*=}")"; shift ;;
            --output_dir) output_dir="$(resolve_repo_path "$2")"; shift 2 ;;
            --use_rgb=*) use_rgb="${1#*=}"; shift ;;
            --use_rgb) use_rgb="$2"; shift 2 ;;
            *) break ;;
        esac
    done

    setup_log convert "$object_name" "$scene_name"

    if [[ "$benchmark" == "lerobot" ]]; then
        local convert_data_root="$data_root"
        local convert_hdf5_name="$hdf5_name"
        local conversion_tmp_dir=""
        local data_root_abs; data_root_abs="$(resolve_repo_path "$data_root")"
        local hdf5_path="$data_root_abs/$hdf5_name"
        local split_hdf5_dir=""
        if [[ -d "$hdf5_path" ]]; then
            split_hdf5_dir="$hdf5_path"
        elif [[ "$hdf5_name" == *.hdf5 && -d "$data_root_abs/${hdf5_name%.hdf5}" ]]; then
            split_hdf5_dir="$data_root_abs/${hdf5_name%.hdf5}"
        fi

        if [[ -n "$split_hdf5_dir" ]]; then
            conversion_tmp_dir="$(mktemp -d "$REPO_ROOT/data/automoma/.merge_for_convert.XXXXXX")"
            local merged_hdf5="$conversion_tmp_dir/$(basename "$split_hdf5_dir").hdf5"
            python "$REPO_ROOT/scripts/convert_automoma_hdf5_layout.py" \
                "$split_hdf5_dir" \
                "$merged_hdf5" \
                --direction merge \
                --mode copy \
                --overwrite
            convert_data_root="$conversion_tmp_dir"
            convert_hdf5_name="$(basename "$merged_hdf5")"
        fi

        local -a cmd=(
            python isaaclab_arena_gr00t/data_utils/convert_hdf5_to_lerobot_v30.py
            --yaml_file isaaclab_arena_gr00t/config/summit_franka_manip_config.yaml
            --data_root "$convert_data_root"
            --hdf5_name "$convert_hdf5_name"
            --repo_id "$repo_id"
            --output_dir "$output_dir"
            "$@"
        )

        echo "Command: ${cmd[*]}"
        echo ""
        pushd "$ISAACLAB_ARENA" > /dev/null
        set +e
        run_logged "${cmd[@]}"
        local status=$?
        set -e
        popd > /dev/null
        if [[ -n "$conversion_tmp_dir" ]]; then
            rm -rf "$conversion_tmp_dir"
        fi
        return "$status"
    fi

    if [[ -z "$policy" ]]; then
        echo "Error: robotwin convert requires --policy." >&2
        exit 1
    fi

    local policy_name_upper; policy_name_upper="$(capitalize_policy "$policy")"
    local hdf5_path="$data_root/$hdf5_name"
    local robotwin_tmp_dir=""
    local robotwin_policy_dir="$ROBOTWIN_ROOT/policy/$policy_name_upper"

    if [[ ! -d "$robotwin_policy_dir" ]]; then
        echo "Error: RoboTwin policy directory not found: $robotwin_policy_dir" >&2
        exit 1
    fi

    if [[ -d "$hdf5_path" ]]; then
        robotwin_tmp_dir="$(mktemp -d "$REPO_ROOT/data/automoma/.merge_for_convert.XXXXXX")"
        local robotwin_merged_hdf5="$robotwin_tmp_dir/$(basename "$hdf5_path").hdf5"
        python "$REPO_ROOT/scripts/convert_automoma_hdf5_layout.py" \
            "$hdf5_path" \
            "$robotwin_merged_hdf5" \
            --direction merge \
            --mode copy \
            --overwrite
        hdf5_path="$robotwin_merged_hdf5"
    elif [[ "$hdf5_name" == *.hdf5 && -d "$data_root/${hdf5_name%.hdf5}" ]]; then
        local split_hdf5_dir="$data_root/${hdf5_name%.hdf5}"
        robotwin_tmp_dir="$(mktemp -d "$REPO_ROOT/data/automoma/.merge_for_convert.XXXXXX")"
        local robotwin_merged_hdf5="$robotwin_tmp_dir/$(basename "$split_hdf5_dir").hdf5"
        python "$REPO_ROOT/scripts/convert_automoma_hdf5_layout.py" \
            "$split_hdf5_dir" \
            "$robotwin_merged_hdf5" \
            --direction merge \
            --mode copy \
            --overwrite
        hdf5_path="$robotwin_merged_hdf5"
    fi

    local -a cmd=(
        bash process_data.sh
        "$object_name"
        "$scene_name"
        "$num_episodes"
        --input_hdf5 "$hdf5_path"
    )
    if [[ -n "$use_rgb" ]]; then
        cmd+=(--use_rgb "$use_rgb")
    fi
    cmd+=("$@")

    echo "Command: ${cmd[*]}"
    echo ""
    pushd "$robotwin_policy_dir" > /dev/null
    set +e
    run_logged "${cmd[@]}"
    local status=$?
    set -e
    popd > /dev/null
    if [[ -n "$robotwin_tmp_dir" ]]; then
        rm -rf "$robotwin_tmp_dir"
    fi
    return "$status"
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
    local use_rgb=""

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
            --use_rgb=*) use_rgb="${1#*=}"; shift ;;
            --use_rgb) use_rgb="$2"; shift 2 ;;
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
    )
    if [[ -n "$use_rgb" ]]; then
        cmd+=("policy.use_pc_color=${use_rgb}")
    fi
    cmd+=("$@")

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
        local eval_episode_length="${EVAL_EPISODE_LENGTH:-300}"
        local eval_interpolated="${EVAL_INTERPOLATED:-1}"
        local eval_interpolation_type="${EVAL_INTERPOLATION_TYPE:-linear}"
        local eval_decimation="${EVAL_DECIMATION:-1}"
        local eval_init_steps="${EVAL_INIT_STEPS:-5}"
        local eval_mobile_base_relative="${EVAL_MOBILE_BASE_RELATIVE:-true}"

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --headless) env_headless="true"; shift ;;
                --no-headless) env_headless="false"; shift ;;
                --policy.path=*) policy_path="${1#*=}"; shift ;;
                --policy.path) policy_path="$2"; shift 2 ;;
                --traj_file=*) traj_file="$(resolve_repo_path "${1#*=}")"; shift ;;
                --traj_file) traj_file="$(resolve_repo_path "$2")"; shift 2 ;;
                --traj_seed=*) traj_seed="${1#*=}"; shift ;;
                --traj_seed) traj_seed="$2"; shift 2 ;;
                --output_dir=*) output_dir="$(resolve_repo_path "${1#*=}")"; shift ;;
                --output_dir) output_dir="$(resolve_repo_path "$2")"; shift 2 ;;
                --interpolated=*) eval_interpolated="${1#*=}"; shift ;;
                --interpolated) eval_interpolated="$2"; shift 2 ;;
                --interpolation_type=*) eval_interpolation_type="${1#*=}"; shift ;;
                --interpolation_type) eval_interpolation_type="$2"; shift 2 ;;
                --mobile_base_relative=*) eval_mobile_base_relative="${1#*=}"; shift ;;
                --mobile_base_relative) eval_mobile_base_relative="$2"; shift 2 ;;
                *) break ;;
            esac
        done

        require_eval_traj_file "$traj_file"
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

        local -a cmd=(
            python "$REPO_ROOT/scripts/lerobot_eval_aligned.py"
            --policy.path="$policy_path"
            --policy.device=cuda
            --object_name="$object_name"
            --scene_name="$scene_name"
            --traj_file="$traj_file"
            --traj_seed="$traj_seed"
            --headless="$env_headless"
            --env.episode_length="$eval_episode_length"
            --decimation="$eval_decimation"
            --init_steps="$eval_init_steps"
            --interpolated="$eval_interpolated"
            --interpolation_type="$eval_interpolation_type"
            --mobile_base_relative="$eval_mobile_base_relative"
            --openness_threshold="$openness_threshold"
            --proximity_threshold="$proximity_threshold"
            --proximity_window_steps="$proximity_window_steps"
            --proximity_required_steps="$proximity_required_steps"
            --disable_fingertip_proximity="$disable_fingertip_proximity"
            --debug_visualize_handle="$debug_visualize_handle"
            --debug_record_handle_diagnostics="$debug_record_handle_diagnostics"
            --debug_marker_scale="$debug_marker_scale"
            --eval.n_episodes=50
            --output_dir="$output_dir"
        )
        append_robot_object_friction_args cmd
        cmd+=("$@")

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
successes = [str(row["success"]).lower() == "true" for row in rows]
video_paths = [row.get("video_path", "") for row in rows if row.get("video_path")]

per_episode = [
    {
        "episode_ix": int(row["episode_ix"]),
        "final_door_openness": float(row["final_door_openness"]) if row.get("final_door_openness") else None,
        "final_handle_distance": float(row["final_handle_distance"]) if row.get("final_handle_distance") else None,
        "success": str(row["success"]).lower() == "true",
        "seed": int(row["seed"]) if row.get("seed") else None,
    }
    for row in rows
]

eval_info = {
    "per_episode": per_episode,
    "aggregated": {
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
    local traj_file="$REPO_ROOT/data/trajs/summit_franka/${object_name}/${scene_name}/test/traj_data_test.pt"
    local use_rgb=""
    local legacy_cvpr26="false"
    local eval_mobile_base_relative="${EVAL_MOBILE_BASE_RELATIVE:-true}"
    local seed_explicit="false"
    local checkpoint_root_explicit="false"
    local output_dir_explicit="false"
    local -a passthrough=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --seed=*) seed="${1#*=}"; seed_explicit="true"; shift ;;
            --seed) seed="$2"; seed_explicit="true"; shift 2 ;;
            --gpu_id=*) gpu_id="${1#*=}"; shift ;;
            --gpu_id) gpu_id="$2"; shift 2 ;;
            --ckpt_setting=*) ckpt_setting="${1#*=}"; shift ;;
            --ckpt_setting) ckpt_setting="$2"; shift 2 ;;
            --checkpoint_root=*) checkpoint_root="${1#*=}"; checkpoint_root_explicit="true"; shift ;;
            --checkpoint_root) checkpoint_root="$2"; checkpoint_root_explicit="true"; shift 2 ;;
            --output_dir=*) output_dir="${1#*=}"; output_dir_explicit="true"; shift ;;
            --output_dir) output_dir="$2"; output_dir_explicit="true"; shift 2 ;;
            --traj_file=*)
                traj_file="$(resolve_repo_path "${1#*=}")"
                shift
                ;;
            --traj_file)
                traj_file="$(resolve_repo_path "$2")"
                shift 2
                ;;
            --use_rgb=*) use_rgb="${1#*=}"; shift ;;
            --use_rgb) use_rgb="$2"; shift 2 ;;
            --mobile_base_relative=*) eval_mobile_base_relative="${1#*=}"; shift ;;
            --mobile_base_relative) eval_mobile_base_relative="$2"; shift 2 ;;
            --legacy_cvpr26)
                legacy_cvpr26="true"
                shift
                ;;
            --legacy_cvpr26=*)
                local legacy_value="${1#*=}"
                case "${legacy_value,,}" in
                    1|true|yes|y|on) legacy_cvpr26="true" ;;
                    *) legacy_cvpr26="false" ;;
                esac
                shift
                ;;
            *) passthrough+=("$1"); shift ;;
        esac
    done

    if [[ "$legacy_cvpr26" == "true" ]]; then
        if [[ "$seed_explicit" == "false" ]]; then
            seed="0"
        fi
        if [[ "$checkpoint_root_explicit" == "false" ]]; then
            checkpoint_root="$REPO_ROOT/outputs/train/debug_eval/robotwin/${policy}"
        fi
        if [[ "$output_dir_explicit" == "false" ]]; then
            output_dir="$REPO_ROOT/outputs/eval/debug_eval/robotwin/${policy}_${name}"
        fi
    fi

    require_eval_traj_file "$traj_file"
    local -a friction_passthrough=()
    append_robot_object_friction_args friction_passthrough
    passthrough=(--traj_file "$traj_file" --mobile_base_relative "$eval_mobile_base_relative" "${friction_passthrough[@]}" "${passthrough[@]}")
    local eval_wrapper="$REPO_ROOT/scripts/robotwin_eval.sh"
    local task_name="$object_name"
    local task_config="$scene_name"
    local -a cmd=()
    if [[ "$legacy_cvpr26" == "true" ]]; then
        if [[ "$policy_name_upper" != "DP3" ]]; then
            echo "Error: --legacy_cvpr26 is only supported for RoboTwin DP3 eval." >&2
            exit 1
        fi
        cmd=(
            python "$REPO_ROOT/scripts/cvpr26/robotwin_dp3_eval.py"
            --task_name "$task_name"
            --task_config "$task_config"
            --expert_data_num "$num_episodes"
            --ckpt_setting "$ckpt_setting"
            --seed "$seed"
            --gpu_id "$gpu_id"
            --checkpoint_root "$checkpoint_root"
            --output_dir "$output_dir"
            --legacy_cvpr26
        )
    else
        cmd=(
            bash "$eval_wrapper"
            "$policy_name_upper"
            "$task_name"
            "$task_config"
            "$num_episodes"
            "$ckpt_setting"
            "$seed"
            "$gpu_id"
            "$checkpoint_root"
            "$output_dir"
        )
    fi
    if [[ -n "$use_rgb" ]]; then
        cmd+=(--use_rgb "$use_rgb")
    fi
    cmd+=("${passthrough[@]}")

    echo "Command: ${cmd[*]}"
    echo ""
    pushd "$REPO_ROOT" > /dev/null
    run_logged "${cmd[@]}"
    popd > /dev/null
}

do_record_dataset_eval() {
    local object_name="$1"; shift
    local scene_name="$1"; shift
    local num_episodes="$1"; shift

    local name; name="$(mk_exp_name "$object_name" "$scene_name" "$num_episodes")"
    local dataset_file="$REPO_ROOT/data/automoma/${name}.hdf5"
    local output_dir="$REPO_ROOT/outputs/eval/record_dataset/${name}"
    local traj_file="$REPO_ROOT/data/trajs/summit_franka/${object_name}/${scene_name}/test/traj_data_test.pt"
    local traj_seed="${EVAL_TRAJ_SEED:-42}"
    local env_headless="false"
    local eval_episode_length="${EVAL_EPISODE_LENGTH:-}"
    local eval_interpolated="${EVAL_INTERPOLATED:-1}"
    local eval_interpolation_type="${EVAL_INTERPOLATION_TYPE:-linear}"
    local eval_decimation="${EVAL_DECIMATION:-}"
    local eval_init_steps="${EVAL_INIT_STEPS:-1}"
    local action_key="${RECORD_DATASET_EVAL_ACTION_KEY:-actions}"
    local max_episodes_rendered="${MAX_EPISODES_RENDERED:-10}"
    local -a passthrough=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --headless) env_headless="true"; shift ;;
            --no-headless) env_headless="false"; shift ;;
            --dataset_file=*) dataset_file="$(resolve_repo_path "${1#*=}")"; shift ;;
            --dataset_file) dataset_file="$(resolve_repo_path "$2")"; shift 2 ;;
            --traj_file=*) traj_file="$(resolve_repo_path "${1#*=}")"; shift ;;
            --traj_file) traj_file="$(resolve_repo_path "$2")"; shift 2 ;;
            --traj_seed=*) traj_seed="${1#*=}"; shift ;;
            --traj_seed) traj_seed="$2"; shift 2 ;;
            --output_dir=*) output_dir="$(resolve_repo_path "${1#*=}")"; shift ;;
            --output_dir) output_dir="$(resolve_repo_path "$2")"; shift 2 ;;
            --action_key=*) action_key="${1#*=}"; shift ;;
            --action_key) action_key="$2"; shift 2 ;;
            --max_episodes_rendered=*) max_episodes_rendered="${1#*=}"; shift ;;
            --max_episodes_rendered) max_episodes_rendered="$2"; shift 2 ;;
            --init_steps=*) eval_init_steps="${1#*=}"; shift ;;
            --init_steps) eval_init_steps="$2"; shift 2 ;;
            --interpolated=*) eval_interpolated="${1#*=}"; shift ;;
            --interpolated) eval_interpolated="$2"; shift 2 ;;
            --interpolation_type=*) eval_interpolation_type="${1#*=}"; shift ;;
            --interpolation_type) eval_interpolation_type="$2"; shift 2 ;;
            --decimation=*) eval_decimation="${1#*=}"; shift ;;
            --decimation) eval_decimation="$2"; shift 2 ;;
            --env.episode_length=*) eval_episode_length="${1#*=}"; shift ;;
            --env.episode_length) eval_episode_length="$2"; shift 2 ;;
            *) passthrough+=("$1"); shift ;;
        esac
    done

    if [[ ! -f "$dataset_file" ]]; then
        echo "Error: Record dataset HDF5 not found: $dataset_file" >&2
        exit 1
    fi

    setup_log record_dataset_eval "$object_name" "$scene_name"

    rm -f "$output_dir/per_episode_results.csv" "$output_dir/eval_info.json" "$output_dir/action_trace_joint_states.csv"
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

    local -a cmd=(
        python "$REPO_ROOT/scripts/record_dataset_eval.py"
        --dataset_file="$dataset_file"
        --output_dir="$output_dir"
        --object_name="$object_name"
        --scene_name="$scene_name"
        --traj_seed="$traj_seed"
        --headless="$env_headless"
        --init_steps="$eval_init_steps"
        --interpolated="$eval_interpolated"
        --interpolation_type="$eval_interpolation_type"
        --action_key="$action_key"
        --openness_threshold="$openness_threshold"
        --proximity_threshold="$proximity_threshold"
        --proximity_window_steps="$proximity_window_steps"
        --proximity_required_steps="$proximity_required_steps"
        --disable_fingertip_proximity="$disable_fingertip_proximity"
        --debug_visualize_handle="$debug_visualize_handle"
        --debug_record_handle_diagnostics="$debug_record_handle_diagnostics"
        --debug_marker_scale="$debug_marker_scale"
        --max_episodes_rendered="$max_episodes_rendered"
        --eval.n_episodes="$num_episodes"
    )
    if [[ -n "$eval_decimation" ]]; then
        cmd+=(--decimation="$eval_decimation")
    fi
    if [[ -n "$eval_episode_length" ]]; then
        cmd+=(--env.episode_length="$eval_episode_length")
    fi
    append_robot_object_friction_args cmd
    if [[ -f "$traj_file" ]]; then
        cmd+=(--traj_file="$traj_file")
    else
        echo "Warning: Eval trajectory file not found; relying on HDF5 initial state only: $traj_file"
    fi
    cmd+=("${passthrough[@]}")

    echo "Command: ${cmd[*]}"
    echo ""
    pushd "$ISAACLAB_ARENA" > /dev/null
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
    replay)
        require_arg "${1:-}" "object_name"
        require_arg "${2:-}" "scene_name"
        require_arg "${3:-}" "num_episodes"
        do_replay "$1" "$2" "$3" "${@:4}"
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
    record_dataset_eval)
        require_arg "${1:-}" "object_name"
        require_arg "${2:-}" "scene_name"
        require_arg "${3:-}" "num_episodes"
        do_record_dataset_eval "$1" "$2" "$3" "${@:4}"
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
