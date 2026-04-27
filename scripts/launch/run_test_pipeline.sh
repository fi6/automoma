#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SCENE_NAME="${SCENE_NAME:-scene_54_seed_54}"
OBJECT_ID="${OBJECT_ID:-7221}"
OBJECT_NAME="${OBJECT_NAME:-microwave_${OBJECT_ID}}"
POLICY="${POLICY:-act}"
BENCHMARK="${BENCHMARK:-lerobot}"
TRAIN_EPISODES="${TRAIN_EPISODES:-10}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
GRASP_COUNT="${GRASP_COUNT:-20}"
PLAN_EXTRA_ARGS="${PLAN_EXTRA_ARGS:-}"
RECORD_EXTRA_ARGS="${RECORD_EXTRA_ARGS:---headless --set_state}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:---env.headless=true}"
FORCE="${FORCE:-0}"

# OOD evaluation: comma-separated list of trained scenes and eval scenes
# e.g., TRAIN_SCENES=scene_0_seed_0,scene_7_seed_7 EVAL_SCENES=scene_40_seed_40,scene_41_seed_41
TRAIN_SCENES="${TRAIN_SCENES:-${SCENE_NAME}}"
EVAL_SCENES="${EVAL_SCENES:-}"

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

# Convert comma-separated string to array
split_to_array() {
    local IFS=','
    read -ra "$1" <<< "$2"
}

# -----------------------------------------------------------------------------
# Normal pipeline (train on SCENE_NAME, eval on same SCENE_NAME)
# -----------------------------------------------------------------------------
run_normal_pipeline() {
    local scene="$1"

    TRAIN_TRAJ_FILE="$REPO_ROOT/data/trajs/summit_franka/${OBJECT_NAME}/${scene}/train/traj_data_train.pt"
    TEST_TRAJ_FILE="$REPO_ROOT/data/trajs/summit_franka/${OBJECT_NAME}/${scene}/test/traj_data_test.pt"
    TRAIN_HDF5="$REPO_ROOT/data/automoma/summit_franka_open-${OBJECT_NAME}-${scene}-${TRAIN_EPISODES}.hdf5"
    TRAIN_DATASET_DIR="$REPO_ROOT/data/lerobot/automoma/summit_franka_open-${OBJECT_NAME}-${scene}-${TRAIN_EPISODES}"
    TRAIN_OUTPUT_DIR="$REPO_ROOT/outputs/train/${BENCHMARK}/${POLICY}_summit_franka_open-${OBJECT_NAME}-${scene}-${TRAIN_EPISODES}"
    EVAL_OUTPUT_DIR="$REPO_ROOT/outputs/eval/${BENCHMARK}/${POLICY}_summit_franka_open-${OBJECT_NAME}-${scene}-${TRAIN_EPISODES}"

    GRASP_OVERRIDE="$(make_grasp_override "$GRASP_COUNT")"

    run_stage \
        "plan(train)" \
        "$TRAIN_TRAJ_FILE" \
        "cd '$REPO_ROOT' && python scripts/plan.py scene_name=${scene} object_id=${OBJECT_ID} mode=train \"$GRASP_OVERRIDE\" ${PLAN_EXTRA_ARGS}"

    run_stage \
        "record" \
        "$TRAIN_HDF5" \
        "cd '$REPO_ROOT' && bash scripts/run_pipeline.sh record ${OBJECT_NAME} ${scene} ${TRAIN_EPISODES} ${RECORD_EXTRA_ARGS}"

    run_stage \
        "convert" \
        "$TRAIN_DATASET_DIR" \
        "cd '$REPO_ROOT' && bash scripts/run_pipeline.sh convert ${BENCHMARK} ${OBJECT_NAME} ${scene} ${TRAIN_EPISODES}"

    run_stage \
        "train" \
        "$TRAIN_OUTPUT_DIR" \
        "cd '$REPO_ROOT' && bash scripts/run_pipeline.sh train ${BENCHMARK} ${POLICY} ${OBJECT_NAME} ${scene} ${TRAIN_EPISODES} ${TRAIN_EXTRA_ARGS}"

    run_stage \
        "plan(test)" \
        "$TEST_TRAJ_FILE" \
        "cd '$REPO_ROOT' && python scripts/plan.py scene_name=${scene} object_id=${OBJECT_ID} mode=test \"$GRASP_OVERRIDE\" ${PLAN_EXTRA_ARGS}"

    run_stage \
        "eval" \
        "$EVAL_OUTPUT_DIR" \
        "cd '$REPO_ROOT' && bash scripts/run_pipeline.sh eval ${BENCHMARK} ${POLICY} ${OBJECT_NAME} ${scene} ${TRAIN_EPISODES} --output_dir=${EVAL_OUTPUT_DIR} --eval.n_episodes=${EVAL_EPISODES} ${EVAL_EXTRA_ARGS}"
}

# -----------------------------------------------------------------------------
# OOD pipeline (train on TRAIN_SCENES, eval on EVAL_SCENES)
# -----------------------------------------------------------------------------
run_ood_pipeline() {
    local trained_scene="$1"
    local eval_scene="$2"

    # Use test traj from eval_scene
    TEST_TRAJ_FILE="$REPO_ROOT/data/trajs/summit_franka/${OBJECT_NAME}/${eval_scene}/test/traj_data_test.pt"

    # Model from trained_scene
    TRAIN_OUTPUT_DIR="$REPO_ROOT/outputs/train/${BENCHMARK}/${POLICY}_summit_franka_open-${OBJECT_NAME}-${trained_scene}-${TRAIN_EPISODES}"
    POLICY_PATH="$TRAIN_OUTPUT_DIR/checkpoints/last/pretrained_model"

    # Eval output dir indicates which model was used and on which scene
    EVAL_OUTPUT_DIR="$REPO_ROOT/outputs/eval/${BENCHMARK}/${POLICY}_summit_franka_open-${OBJECT_NAME}-${trained_scene}-${TRAIN_EPISODES}/ood_on_${eval_scene}"

    GRASP_OVERRIDE="$(make_grasp_override "$GRASP_COUNT")"

    echo ""
    echo "=== OOD Eval: model trained on ${trained_scene}, evaluating on ${eval_scene} ==="

    run_stage \
        "plan(test) for ood" \
        "$TEST_TRAJ_FILE" \
        "cd '$REPO_ROOT' && python scripts/plan.py scene_name=${eval_scene} object_id=${OBJECT_ID} mode=test \"$GRASP_OVERRIDE\" ${PLAN_EXTRA_ARGS}"

    run_stage \
        "ood_eval(${trained_scene}->${eval_scene})" \
        "$EVAL_OUTPUT_DIR" \
        "cd '$REPO_ROOT' && bash scripts/run_pipeline.sh eval ${BENCHMARK} ${POLICY} ${OBJECT_NAME} ${eval_scene} ${TRAIN_EPISODES} --output_dir=${EVAL_OUTPUT_DIR} --eval.n_episodes=${EVAL_EPISODES} --policy.path=${POLICY_PATH} ${EVAL_EXTRA_ARGS}"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if [[ -n "$EVAL_SCENES" ]]; then
    # OOD mode: train on TRAIN_SCENES, eval on EVAL_SCENES
    split_to_array TRAIN_SCENES_ARRAY "$TRAIN_SCENES"
    split_to_array EVAL_SCENES_ARRAY "$EVAL_SCENES"

    echo "OOD Evaluation mode:"
    echo "  trained_scenes=${TRAIN_SCENES}"
    echo "  eval_scenes=${EVAL_SCENES}"

    for eval_scene in "${EVAL_SCENES_ARRAY[@]}"; do
        for trained_scene in "${TRAIN_SCENES_ARRAY[@]}"; do
            run_ood_pipeline "$trained_scene" "$eval_scene"
        done
    done
else
    # Normal mode
    run_normal_pipeline "$SCENE_NAME"
fi

echo ""
echo "Pipeline complete:"
echo "  scene=${SCENE_NAME}"
echo "  object=${OBJECT_NAME}"
echo "  policy=${POLICY}"
echo "  benchmark=${BENCHMARK}"
echo "  grasp_count=${GRASP_COUNT}"
echo "  train_episodes=${TRAIN_EPISODES}"
echo "  eval_episodes=${EVAL_EPISODES}"
if [[ -n "$EVAL_SCENES" ]]; then
    echo "  trained_scenes=${TRAIN_SCENES}"
    echo "  eval_scenes=${EVAL_SCENES}"
fi
