#!/usr/bin/env bash
set -euo pipefail

POLICY=${1}
TASK_NAME=${2}
TASK_CONFIG=${3}
EXPERT_DATA_NUM=${4}
CKPT_SETTING=${5}
SEED=${6}
GPU_ID=${7}
OUTPUT_DIR=${8}
shift 8

ROBOTWIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../third_party/RoboTwin" && pwd)"
POLICY_DIR="$ROBOTWIN_ROOT/policy/$POLICY"

if [[ ! -d "$POLICY_DIR" ]]; then
    echo "Error: RoboTwin policy directory not found: $POLICY_DIR" >&2
    exit 1
fi

case "$POLICY" in
    DP3)
        cd "$POLICY_DIR"
        bash train.sh "$TASK_NAME" "$TASK_CONFIG" "$EXPERT_DATA_NUM" "$CKPT_SETTING" "$SEED" "$GPU_ID" "$OUTPUT_DIR" "$@"
        ;;
    *)
        echo "Error: Unsupported RoboTwin policy for train wrapper: $POLICY" >&2
        exit 1
        ;;
esac
