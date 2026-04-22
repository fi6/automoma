#!/usr/bin/env bash
set -euo pipefail

POLICY=${1}
TASK_NAME=${2}
TASK_CONFIG=${3}
EXPERT_DATA_NUM=${4}
CKPT_SETTING=${5}
SEED=${6}
GPU_ID=${7}
CHECKPOINT_ROOT=${8}
OUTPUT_DIR=${9}
shift 9

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOTWIN_ROOT="$REPO_ROOT/third_party/RoboTwin"
POLICY_DIR="$ROBOTWIN_ROOT/policy/$POLICY"

if [[ ! -d "$POLICY_DIR" ]]; then
    echo "Error: RoboTwin policy directory not found: $POLICY_DIR" >&2
    exit 1
fi

case "$POLICY" in
    DP3)
        python "$REPO_ROOT/scripts/robotwin_dp3_eval.py" \
            --task_name "$TASK_NAME" \
            --task_config "$TASK_CONFIG" \
            --expert_data_num "$EXPERT_DATA_NUM" \
            --ckpt_setting "$CKPT_SETTING" \
            --seed "$SEED" \
            --gpu_id "$GPU_ID" \
            --checkpoint_root "$CHECKPOINT_ROOT" \
            --output_dir "$OUTPUT_DIR" \
            "$@"
        ;;
    *)
        echo "Error: Unsupported RoboTwin policy for eval wrapper: $POLICY" >&2
        exit 1
        ;;
esac
