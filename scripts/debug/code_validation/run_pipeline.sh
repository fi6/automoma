#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUN_PIPELINE="$REPO_ROOT/scripts/run_pipeline.sh"
TRAJ_UTIL="$SCRIPT_DIR/manage_validation_trajs.py"
SUBSET_UTIL="$SCRIPT_DIR/build_validation_subsets.py"
SUMMARY_UTIL="$SCRIPT_DIR/summarize_validation_results.py"

SCENE_NAME="${SCENE_NAME:-scene_0_seed_0}"
OBJECT_ID="${OBJECT_ID:-7221}"
OBJECT_NAME="${OBJECT_NAME:-microwave_${OBJECT_ID}}"
TRAIN_MIN_PLANNED="${TRAIN_MIN_PLANNED:-7000}"
TEST_MIN_PLANNED="${TEST_MIN_PLANNED:-50}"
PLAN_MAX_ROUNDS="${PLAN_MAX_ROUNDS:-10}"
PLAN_EXTRA_ARGS="${PLAN_EXTRA_ARGS:-}"
RECORD_EPISODES="${RECORD_EPISODES:-6400}"
RECORD_MODE="${RECORD_MODE:-set_state}"
RECORD_EXTRA_ARGS="${RECORD_EXTRA_ARGS:---headless}"
SUBSET_SIZES="${SUBSET_SIZES:-100 200 400 800 1600 3200 6400}"
POLICIES="${POLICIES:-diffusion}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
BASE_TRAIN_STEPS="${BASE_TRAIN_STEPS:-20000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_TRAJ_SEED="${EVAL_TRAJ_SEED:-42}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:---env.headless=true}"
FORCE_TRAIN_OVERWRITE="${FORCE_TRAIN_OVERWRITE:-1}"

VALIDATION_TAG="${VALIDATION_TAG:-code_validation}"
VALIDATION_NAME="${VALIDATION_TAG}-${OBJECT_NAME}-${SCENE_NAME}-${RECORD_EPISODES}-${RECORD_MODE}"
HDF5_ROOT="$REPO_ROOT/data/automoma/${VALIDATION_TAG}"
LEROBOT_ROOT="$REPO_ROOT/data/lerobot/${VALIDATION_TAG}"
TRAIN_ROOT="$REPO_ROOT/outputs/train/${VALIDATION_TAG}"
EVAL_ROOT="$REPO_ROOT/outputs/eval/${VALIDATION_TAG}"
SUMMARY_ROOT="$REPO_ROOT/outputs/${VALIDATION_TAG}"
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$HDF5_ROOT" "$LEROBOT_ROOT" "$TRAIN_ROOT" "$EVAL_ROOT" "$SUMMARY_ROOT"

TRAIN_TRAJ_FILE="$REPO_ROOT/data/trajs/summit_franka/${OBJECT_NAME}/${SCENE_NAME}/train/traj_data_train.pt"
TEST_TRAJ_FILE="$REPO_ROOT/data/trajs/summit_franka/${OBJECT_NAME}/${SCENE_NAME}/test/traj_data_test.pt"
HDF5_NAME="${VALIDATION_NAME}.hdf5"
FULL_DATASET_REPO_ID="${VALIDATION_NAME}"
FULL_DATASET_ROOT="$LEROBOT_ROOT/$FULL_DATASET_REPO_ID"
SUBSET_OUTPUT_ROOT="$LEROBOT_ROOT/subsets"
SUMMARY_CSV="$SUMMARY_ROOT/${VALIDATION_NAME}_summary.csv"
SUMMARY_JSON="$SUMMARY_ROOT/${VALIDATION_NAME}_summary.json"
SUBSET_MANIFEST="$SUBSET_OUTPUT_ROOT/subset_manifest.json"

run_cmd() {
    echo ""
    echo ">>> $*"
    eval "$@"
}

warn_skip() {
    echo "Warning: $1"
}

remove_if_forced() {
    local path="$1"
    if [[ "$FORCE" == "1" && -e "$path" ]]; then
        rm -rf "$path"
    fi
}

plan_until_target() {
    local split="$1"
    local target="$2"
    local traj_file="$3"
    local round=1

    while true; do
        if [[ -f "$traj_file" ]]; then
            local current
            current=$(python "$TRAJ_UTIL" count "$traj_file" | python -c 'import sys,json; print(json.load(sys.stdin)["successful"])')
            if [[ "$current" -ge "$target" ]]; then
                warn_skip "skip plan(${split}): found ${traj_file} with ${current} successful trajectories >= ${target}"
                return 0
            fi
            if [[ "$round" -gt "$PLAN_MAX_ROUNDS" ]]; then
                echo "Error: ${split} planning did not reach target after ${PLAN_MAX_ROUNDS} rounds" >&2
                return 1
            fi
            warn_skip "plan(${split}) will append into existing per-grasp and merged trajectory files in $(dirname "$traj_file") during this round"
            run_cmd "bash '$RUN_PIPELINE' plan '$OBJECT_ID' '$SCENE_NAME' '$split' $PLAN_EXTRA_ARGS"
            round=$((round + 1))
        else
            run_cmd "bash '$RUN_PIPELINE' plan '$OBJECT_ID' '$SCENE_NAME' '$split' $PLAN_EXTRA_ARGS"
        fi
    done
}

record_args=()
if [[ "$RECORD_MODE" == "set_state" ]]; then
    record_args+=(--set_state)
fi

ensure_recorded() {
    local hdf5_path="$HDF5_ROOT/$HDF5_NAME"
    remove_if_forced "$hdf5_path"
    if [[ -f "$hdf5_path" ]]; then
        warn_skip "skip record: found $hdf5_path (use --force to rerun)"
        return 0
    fi
    run_cmd "bash '$RUN_PIPELINE' record '$OBJECT_NAME' '$SCENE_NAME' '$RECORD_EPISODES' --dataset_file='$hdf5_path' ${record_args[*]} $RECORD_EXTRA_ARGS"
}

ensure_converted() {
    remove_if_forced "$FULL_DATASET_ROOT"
    if [[ -d "$FULL_DATASET_ROOT" ]]; then
        warn_skip "skip convert: found $FULL_DATASET_ROOT (use --force to rerun)"
        return 0
    fi
    run_cmd "bash '$RUN_PIPELINE' convert '$OBJECT_NAME' '$SCENE_NAME' '$RECORD_EPISODES' --data_root='$HDF5_ROOT' --hdf5_name='$HDF5_NAME' --repo_id='$FULL_DATASET_REPO_ID' --output_dir='$FULL_DATASET_ROOT'"
}

ensure_subsets() {
    remove_if_forced "$SUBSET_OUTPUT_ROOT"
    if [[ -f "$SUBSET_MANIFEST" ]]; then
        warn_skip "skip subsets: found $SUBSET_MANIFEST (use --force to rerun)"
        return 0
    fi
    run_cmd "PYTHONPATH='$REPO_ROOT/third_party/lerobot/src' python '$SUBSET_UTIL' --repo-id '$FULL_DATASET_REPO_ID' --root '$FULL_DATASET_ROOT' --output-root '$SUBSET_OUTPUT_ROOT' --subset-sizes $SUBSET_SIZES --seed '$EVAL_TRAJ_SEED' --prefix '$VALIDATION_NAME'"
}

train_all() {
    for policy in $POLICIES; do
        for size in $SUBSET_SIZES; do
            local dataset_repo_id="${VALIDATION_NAME}-${size}"
            local dataset_root="$SUBSET_OUTPUT_ROOT/$dataset_repo_id"
            local output_dir="$TRAIN_ROOT/$policy/$size"
            local job_name="${VALIDATION_NAME}-${policy}-${size}"
            local train_steps=$(( BASE_TRAIN_STEPS * size / 100 ))
            remove_if_forced "$output_dir"
            if [[ -d "$output_dir/checkpoints/last/pretrained_model" ]]; then
                warn_skip "skip train: found $output_dir/checkpoints/last/pretrained_model (use --force to rerun)"
                continue
            fi
            run_cmd "FORCE_TRAIN_OVERWRITE='$FORCE_TRAIN_OVERWRITE' bash '$RUN_PIPELINE' train '$policy' '$OBJECT_NAME' '$SCENE_NAME' '$size' --steps='$train_steps' --dataset.repo_id='$dataset_repo_id' --dataset.root='$dataset_root' --output_dir='$output_dir' --job_name='$job_name' $TRAIN_EXTRA_ARGS"
        done
    done
}

eval_all() {
    for policy in $POLICIES; do
        for size in $SUBSET_SIZES; do
            local policy_path="$TRAIN_ROOT/$policy/$size/checkpoints/last/pretrained_model"
            local train_output_dir="$EVAL_ROOT/$policy/$size/train_init"
            local test_output_dir="$EVAL_ROOT/$policy/$size/test_init"
            remove_if_forced "$train_output_dir"
            remove_if_forced "$test_output_dir"
            if [[ -f "$train_output_dir/eval_info.json" ]]; then
                warn_skip "skip eval train_init: found $train_output_dir/eval_info.json (use --force to rerun)"
            else
                run_cmd "bash '$RUN_PIPELINE' eval '$policy' '$OBJECT_NAME' '$SCENE_NAME' '$size' --policy.path='$policy_path' --traj_file='$TRAIN_TRAJ_FILE' --traj_seed='$EVAL_TRAJ_SEED' --output_dir='$train_output_dir' --eval.n_episodes='$EVAL_EPISODES' $EVAL_EXTRA_ARGS"
            fi
            if [[ -f "$test_output_dir/eval_info.json" ]]; then
                warn_skip "skip eval test_init: found $test_output_dir/eval_info.json (use --force to rerun)"
            else
                run_cmd "bash '$RUN_PIPELINE' eval '$policy' '$OBJECT_NAME' '$SCENE_NAME' '$size' --policy.path='$policy_path' --traj_file='$TEST_TRAJ_FILE' --traj_seed='$EVAL_TRAJ_SEED' --output_dir='$test_output_dir' --eval.n_episodes='$EVAL_EPISODES' $EVAL_EXTRA_ARGS"
            fi
        done
    done
}

summarize() {
    remove_if_forced "$SUMMARY_CSV"
    remove_if_forced "$SUMMARY_JSON"
    if [[ -f "$SUMMARY_CSV" && -f "$SUMMARY_JSON" ]]; then
        warn_skip "skip summarize: found $SUMMARY_CSV and $SUMMARY_JSON (use --force to rerun)"
        return 0
    fi
    run_cmd "python '$SUMMARY_UTIL' --eval-root '$EVAL_ROOT' --output-csv '$SUMMARY_CSV' --output-json '$SUMMARY_JSON'"
}

if [[ "$FORCE" == "1" ]]; then
    remove_if_forced "$TRAIN_TRAJ_FILE"
    remove_if_forced "$TEST_TRAJ_FILE"
fi

plan_until_target train "$TRAIN_MIN_PLANNED" "$TRAIN_TRAJ_FILE"
plan_until_target test "$TEST_MIN_PLANNED" "$TEST_TRAJ_FILE"
ensure_recorded
ensure_converted
ensure_subsets
train_all
eval_all
summarize

echo ""
echo "Validation complete"
echo "  summary_csv=$SUMMARY_CSV"
echo "  summary_json=$SUMMARY_JSON"
