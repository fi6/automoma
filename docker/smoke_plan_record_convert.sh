#!/usr/bin/env bash
# Run inside the AutoMoMa container. It performs a real planner run, records one or
# more IsaacLab-Arena demos, converts the HDF5 to LeRobot format, and prints the
# dataset path that can be visualized/trained from the host.
set -euo pipefail

OBJECT_ID="${OBJECT_ID:-7221}"
OBJECT_NAME="${OBJECT_NAME:-microwave_7221}"
SCENE_NAME="${SCENE_NAME:-scene_0_seed_0}"
SPLIT="${SPLIT:-train}"
NUM_EPISODES="${NUM_EPISODES:-1}"
BENCHMARK="${BENCHMARK:-lerobot}"
TEST_ROOT="${TEST_ROOT:-data/docker_smoke}"
HEADLESS_FLAG="${HEADLESS_FLAG:---headless}"

usage() {
    cat <<EOF_USAGE
Usage: bash docker/smoke_plan_record_convert.sh [options]

Options:
  --object-id ID          default: ${OBJECT_ID}
  --object-name NAME      default: ${OBJECT_NAME}
  --scene NAME            default: ${SCENE_NAME}
  --episodes N            default: ${NUM_EPISODES}
  --test-root PATH        default: ${TEST_ROOT}
  --no-headless           Run recorder with GUI display
  -h, --help              Show help

Extra planner overrides can be supplied through PLAN_OVERRIDES, for example:
  PLAN_OVERRIDES='planner.ik_seeds=4096 objects.7221.grasp_ids=[0,1,2]'
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --no-headless) HEADLESS_FLAG="--no-headless"; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ ! -f scripts/run_pipeline.sh ]]; then
    echo "Error: run this script from the AutoMoMa repo root inside the container." >&2
    exit 1
fi

REPO_ROOT="$(pwd -P)"
if [[ "${TEST_ROOT}" = /* ]]; then
    TEST_ROOT_ABS="${TEST_ROOT}"
    TEST_ROOT_DISPLAY="${TEST_ROOT}"
else
    TEST_ROOT_ABS="${REPO_ROOT}/${TEST_ROOT}"
    TEST_ROOT_DISPLAY="${TEST_ROOT}"
fi

EXP_NAME="summit_franka_open-${OBJECT_NAME}-${SCENE_NAME}-${NUM_EPISODES}"
TRAJ_ROOT="${TEST_ROOT_ABS}/trajs"
TRAJ_FILE="${TRAJ_ROOT}/summit_franka/${OBJECT_NAME}/${SCENE_NAME}/${SPLIT}/traj_data_${SPLIT}.pt"
HDF5_ROOT="${TEST_ROOT_ABS}/automoma"
HDF5_FILE="${HDF5_ROOT}/${EXP_NAME}.hdf5"
LEROBOT_ROOT="${TEST_ROOT_ABS}/lerobot/automoma/${EXP_NAME}"
LEROBOT_ROOT_DISPLAY="${TEST_ROOT_DISPLAY}/lerobot/automoma/${EXP_NAME}"

mkdir -p "${TEST_ROOT_ABS}" "${HDF5_ROOT}"

# Defaults keep this a real planner run while limiting runtime for Docker validation.
DEFAULT_PLAN_OVERRIDES=(
    "output_dir=${TRAJ_ROOT}"
    "resume=false"
    "planner.output.max_successful_trajectories=1"
    "planner.ik_seeds=${PLAN_IK_SEEDS:-2048}"
    "planner.traj.batch_size=${PLAN_TRAJ_BATCH_SIZE:-4}"
    "planner.traj.num_trajopt_seeds=${PLAN_TRAJOPT_SEEDS:-4}"
    "planner.traj.num_graph_seeds=${PLAN_GRAPH_SEEDS:-4}"
    "planner.clustering.kmeans_clusters=${PLAN_KMEANS_CLUSTERS:-32}"
    "planner.clustering.ap_fallback_clusters=${PLAN_AP_FALLBACK_CLUSTERS:-8}"
    "planner.clustering.ap_clusters_upperbound=${PLAN_AP_UPPERBOUND:-16}"
    "planner.clustering.ap_clusters_lowerbound=${PLAN_AP_LOWERBOUND:-2}"
    "objects.${OBJECT_ID}.grasp_ids=${PLAN_GRASP_IDS:-[0,1]}"
)

EXTRA_PLAN_OVERRIDES=()
if [[ -n "${PLAN_OVERRIDES:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_PLAN_OVERRIDES=(${PLAN_OVERRIDES})
fi

echo "[1/4] Docker env check"
python docker/check_env.py

echo "[2/4] Planning -> ${TRAJ_FILE}"
bash scripts/run_pipeline.sh plan "${OBJECT_ID}" "${SCENE_NAME}" "${SPLIT}" \
    "${DEFAULT_PLAN_OVERRIDES[@]}" \
    "${EXTRA_PLAN_OVERRIDES[@]}"

test -f "${TRAJ_FILE}"

echo "[3/4] Recording -> ${HDF5_FILE}"
rm -f "${HDF5_FILE}"
bash scripts/run_pipeline.sh record "${OBJECT_NAME}" "${SCENE_NAME}" "${NUM_EPISODES}" \
    "${HEADLESS_FLAG}" \
    --traj_file "${TRAJ_FILE}" \
    --dataset_file "${HDF5_FILE}" \
    --interpolated "${RECORD_INTERPOLATED:-1}" \
    --interpolation_type "${RECORD_INTERPOLATION_TYPE:-linear}" \
    --init_steps "${RECORD_INIT_STEPS:-5}"

test -f "${HDF5_FILE}"

echo "[4/4] Convert HDF5 -> LeRobot dataset ${LEROBOT_ROOT}"
rm -rf "${LEROBOT_ROOT}"
bash scripts/run_pipeline.sh convert "${BENCHMARK}" "${OBJECT_NAME}" "${SCENE_NAME}" "${NUM_EPISODES}" \
    --data_root "${HDF5_ROOT}" \
    --hdf5_name "${EXP_NAME}.hdf5" \
    --repo_id "automoma/${EXP_NAME}" \
    --output_dir "${LEROBOT_ROOT}"

test -d "${LEROBOT_ROOT}/meta"

cat <<EOF_DONE

Docker plan/record/convert smoke test finished.
Trajectory: ${TRAJ_FILE}
HDF5:       ${HDF5_FILE}
LeRobot:   ${LEROBOT_ROOT}

Visualize from the repo root (host or container):
  lerobot-dataset-viz --repo-id automoma/${EXP_NAME} --root ${LEROBOT_ROOT_DISPLAY} --episode-index 0 --video-backend pyav
EOF_DONE
