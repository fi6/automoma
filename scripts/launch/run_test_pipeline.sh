#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SCENE_NAME="${SCENE_NAME:-scene_1_seed_1}"
OBJECT_ID="${OBJECT_ID:-7221}"
OBJECT_NAME="${OBJECT_NAME:-microwave_${OBJECT_ID}}"
POLICY="${POLICY:-act}"
TRAIN_EPISODES="${TRAIN_EPISODES:-1000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
GRASP_COUNT="${GRASP_COUNT:-20}"
PLAN_EXTRA_ARGS="${PLAN_EXTRA_ARGS:-}"
RECORD_EXTRA_ARGS="${RECORD_EXTRA_ARGS:---headless --set_state}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:---env.headless=true}"
FORCE="${FORCE:-0}"

make_grasp_override() {
    local count="$1"
    local -a ids=()
    local i
    for ((i = 0; i < count; i++)); do
        ids+=("$i")
    done
    local joined
    joined="$(IFS=,; echo "${ids[*]}")"
    printf 'objects.%s.grasp_ids=[%s]' "$OBJECT_ID" "$joined"
}

run_cmd() {
    echo ""
    echo ">>> $*"
    eval "$@"
}

remove_if_forced() {
    local path="$1"
    if [[ "$FORCE" == "1" && -e "$path" ]]; then
        rm -rf "$path"
    fi
}

run_stage() {
    local label="$1"
    local path="$2"
    local cmd="$3"

    remove_if_forced "$path"

    if [[ -e "$path" ]]; then
        echo ""
        echo ">>> Skip ${label}: found $path"
        return
    fi

    run_cmd "$cmd"
}

TRAIN_TRAJ_FILE="$REPO_ROOT/data/trajs/summit_franka/${OBJECT_NAME}/${SCENE_NAME}/traj_data_train.pt"
TEST_TRAJ_FILE="$REPO_ROOT/data/trajs/summit_franka/${OBJECT_NAME}/${SCENE_NAME}/traj_data_test.pt"
TRAIN_HDF5="$REPO_ROOT/data/automoma/summit_franka_open-${OBJECT_NAME}-${SCENE_NAME}-${TRAIN_EPISODES}.hdf5"
TRAIN_DATASET_DIR="$REPO_ROOT/data/lerobot/automoma/summit_franka_open-${OBJECT_NAME}-${SCENE_NAME}-${TRAIN_EPISODES}"
TRAIN_OUTPUT_DIR="$REPO_ROOT/outputs/train/${POLICY}_summit_franka_open-${OBJECT_NAME}-${SCENE_NAME}-${TRAIN_EPISODES}"
EVAL_OUTPUT_DIR="$REPO_ROOT/outputs/eval/${POLICY}_summit_franka_open-${OBJECT_NAME}-${SCENE_NAME}-${TRAIN_EPISODES}"

GRASP_OVERRIDE="$(make_grasp_override "$GRASP_COUNT")"

run_stage \
    "plan(train)" \
    "$TRAIN_TRAJ_FILE" \
    "cd '$REPO_ROOT' && python scripts/plan.py scene_name=${SCENE_NAME} object_id=${OBJECT_ID} mode=train \"$GRASP_OVERRIDE\" ${PLAN_EXTRA_ARGS}"

run_stage \
    "record" \
    "$TRAIN_HDF5" \
    "cd '$REPO_ROOT' && bash scripts/run_pipeline.sh record ${OBJECT_NAME} ${SCENE_NAME} ${TRAIN_EPISODES} ${RECORD_EXTRA_ARGS}"

run_stage \
    "convert" \
    "$TRAIN_DATASET_DIR" \
    "cd '$REPO_ROOT' && bash scripts/run_pipeline.sh convert ${OBJECT_NAME} ${SCENE_NAME} ${TRAIN_EPISODES}"

run_stage \
    "train" \
    "$TRAIN_OUTPUT_DIR" \
    "cd '$REPO_ROOT' && bash scripts/run_pipeline.sh train ${POLICY} ${OBJECT_NAME} ${SCENE_NAME} ${TRAIN_EPISODES} ${TRAIN_EXTRA_ARGS}"

run_stage \
    "plan(test)" \
    "$TEST_TRAJ_FILE" \
    "cd '$REPO_ROOT' && python scripts/plan.py scene_name=${SCENE_NAME} object_id=${OBJECT_ID} mode=test \"$GRASP_OVERRIDE\" ${PLAN_EXTRA_ARGS}"

run_stage \
    "eval" \
    "$EVAL_OUTPUT_DIR" \
    "cd '$REPO_ROOT' && bash scripts/run_pipeline.sh eval ${POLICY} ${OBJECT_NAME} ${SCENE_NAME} ${TRAIN_EPISODES} --output_dir=${EVAL_OUTPUT_DIR} --eval.n_episodes=${EVAL_EPISODES} ${EVAL_EXTRA_ARGS}"

echo ""
echo "Pipeline complete:"
echo "  scene=${SCENE_NAME}"
echo "  object=${OBJECT_NAME}"
echo "  policy=${POLICY}"
echo "  grasp_count=${GRASP_COUNT}"
echo "  train_episodes=${TRAIN_EPISODES}"
echo "  eval_episodes=${EVAL_EPISODES}"
echo "  eval_output_dir=${EVAL_OUTPUT_DIR}"
