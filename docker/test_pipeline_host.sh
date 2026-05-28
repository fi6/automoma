#!/usr/bin/env bash
# Host-side real test helper: run Docker plan/record/convert, then use the external
# conda automoma environment to launch a minimal LeRobot train job on the produced dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${AUTOMOMA_DOCKER_IMAGE:-automoma:main-isaacsim5.1.0}"
GPU_SPEC="${AUTOMOMA_DOCKER_GPUS:-0}"
CONDA_ENV="${EXTERNAL_AUTOMOMA_ENV:-automoma}"
OBJECT_ID="${OBJECT_ID:-7221}"
OBJECT_NAME="${OBJECT_NAME:-microwave_7221}"
SCENE_NAME="${SCENE_NAME:-scene_0_seed_0}"
NUM_EPISODES="${NUM_EPISODES:-1}"
TEST_ROOT="${TEST_ROOT:-data/docker_smoke}"
POLICY="${POLICY:-act}"
TRAIN_STEPS="${TRAIN_STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RUN_EVAL="${RUN_EVAL:-0}"

usage() {
    cat <<EOF_USAGE
Usage: bash docker/test_pipeline_host.sh [options]

Options:
  --image IMAGE        default: ${IMAGE_TAG}
  --gpu GPU           default: ${GPU_SPEC}
  --conda-env ENV     external training env, default: ${CONDA_ENV}
  --object-id ID      default: ${OBJECT_ID}
  --object-name NAME  default: ${OBJECT_NAME}
  --scene NAME        default: ${SCENE_NAME}
  --episodes N        default: ${NUM_EPISODES}
  --test-root PATH    default: ${TEST_ROOT}
  --policy POLICY     default: ${POLICY}
  --steps N           default: ${TRAIN_STEPS}
  --batch-size N      default: ${BATCH_SIZE}
  --run-eval          after external train, run a one-episode Docker eval load test
  -h, --help          Show help
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) IMAGE_TAG="$2"; shift 2 ;;
        --image=*) IMAGE_TAG="${1#*=}"; shift ;;
        --gpu|--gpus) GPU_SPEC="$2"; shift 2 ;;
        --gpu=*|--gpus=*) GPU_SPEC="${1#*=}"; shift ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --conda-env=*) CONDA_ENV="${1#*=}"; shift ;;
        --object-id) OBJECT_ID="$2"; shift 2 ;;
        --object-id=*) OBJECT_ID="${1#*=}"; shift ;;
        --object-name) OBJECT_NAME="$2"; shift 2 ;;
        --object-name=*) OBJECT_NAME="${1#*=}"; shift ;;
        --scene) SCENE_NAME="$2"; shift 2 ;;
        --scene=*) SCENE_NAME="${1#*=}"; shift ;;
        --episodes) NUM_EPISODES="$2"; shift 2 ;;
        --episodes=*) NUM_EPISODES="${1#*=}"; shift ;;
        --test-root) TEST_ROOT="$2"; shift 2 ;;
        --test-root=*) TEST_ROOT="${1#*=}"; shift ;;
        --policy) POLICY="$2"; shift 2 ;;
        --policy=*) POLICY="${1#*=}"; shift ;;
        --steps) TRAIN_STEPS="$2"; shift 2 ;;
        --steps=*) TRAIN_STEPS="${1#*=}"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --batch-size=*) BATCH_SIZE="${1#*=}"; shift ;;
        --run-eval) RUN_EVAL=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

EXP_NAME="summit_franka_open-${OBJECT_NAME}-${SCENE_NAME}-${NUM_EPISODES}"
LEROBOT_ROOT="${TEST_ROOT}/lerobot/automoma/${EXP_NAME}"
TRAIN_OUTPUT="outputs/train/docker_smoke/${POLICY}_${EXP_NAME}"

cd "${REPO_ROOT}"

echo "[Docker] plan + record + convert"
AUTOMOMA_DOCKER_IMAGE="${IMAGE_TAG}" bash docker/run_docker.sh \
    --gpu "${GPU_SPEC}" \
    --name "automoma-smoke-${USER}" \
    --no-tty \
    -- bash docker/smoke_plan_record_convert.sh \
        --object-id "${OBJECT_ID}" \
        --object-name "${OBJECT_NAME}" \
        --scene "${SCENE_NAME}" \
        --episodes "${NUM_EPISODES}" \
        --test-root "${TEST_ROOT}"

test -d "${LEROBOT_ROOT}/meta"

echo "[Host:${CONDA_ENV}] minimal train on ${LEROBOT_ROOT}"
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "Error: conda was not found; cannot run external-env training." >&2
    exit 1
fi
conda activate "${CONDA_ENV}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export FORCE_TRAIN_OVERWRITE=1

bash scripts/run_pipeline.sh train lerobot "${POLICY}" "${OBJECT_NAME}" "${SCENE_NAME}" "${NUM_EPISODES}" \
    --dataset.repo_id "automoma/${EXP_NAME}" \
    --dataset.root "${LEROBOT_ROOT}" \
    --output_dir "${TRAIN_OUTPUT}" \
    --job_name "docker_smoke_${POLICY}_${EXP_NAME}" \
    --steps "${TRAIN_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --wandb.enable=false

if [[ "${RUN_EVAL}" == "1" ]]; then
    POLICY_PATH="/workspace/automoma/${TRAIN_OUTPUT}/checkpoints/last/pretrained_model"
    TRAJ_FILE="/workspace/automoma/${TEST_ROOT}/trajs/summit_franka/${OBJECT_NAME}/${SCENE_NAME}/train/traj_data_train.pt"
    echo "[Docker] one-episode eval load test with ${POLICY_PATH}"
    AUTOMOMA_DOCKER_IMAGE="${IMAGE_TAG}" bash docker/run_docker.sh \
        --gpu "${GPU_SPEC}" \
        --name "automoma-eval-smoke-${USER}" \
        --no-tty \
        -- bash scripts/run_pipeline.sh eval lerobot "${POLICY}" "${OBJECT_NAME}" "${SCENE_NAME}" "${NUM_EPISODES}" \
            --headless \
            --policy.path "${POLICY_PATH}" \
            --traj_file "${TRAJ_FILE}" \
            --output_dir "outputs/eval/docker_smoke/${POLICY}_${EXP_NAME}" \
            --eval.n_episodes=1
fi

cat <<EOF_DONE

Host-side Docker validation finished.
LeRobot dataset path:
  ${LEROBOT_ROOT}

Visualize dataset:
  conda activate ${CONDA_ENV}
  lerobot-dataset-viz --repo-id automoma/${EXP_NAME} --root ${LEROBOT_ROOT} --episode-index 0 --video-backend pyav

Train output:
  ${TRAIN_OUTPUT}
EOF_DONE
