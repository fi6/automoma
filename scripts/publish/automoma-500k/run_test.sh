#!/usr/bin/env bash
# Quick AutoMoMa smoke test for one scene:
#   plan 10 successful trajectories for each configured object, then record and
#   convert each object/scene pair into a LeRobot dataset.
#
# Example:
#   bash scripts/publish/automoma-500k/run_test.sh scene_0_seed_0
#   bash scripts/publish/automoma-500k/run_test.sh --dry-run scene_0_seed_0
#   bash scripts/publish/automoma-500k/run_test.sh scene_0_seed_0 -- planner.traj.batch_size=10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DEFAULT_PYTHON="/home/xinhai/miniconda3/envs/lerobot-arena/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON:-python}"
fi

CONFIG="${REPO_ROOT}/configs/plan.yaml"
SCENE_DIR=""
SCENE_NAME=""
OUTPUT_ROOT="${REPO_ROOT}/data/trajs"
ROUND_ROOT="${REPO_ROOT}/data/trajs/_automoma_test_rounds"
STATISTICS_DIR="${REPO_ROOT}/outputs/statistics/automoma-test"
TARGET_SUCCESS=10
NUM_EPISODES=""
MAX_ROUNDS_PER_OBJECT=10
CLEANUP_ROUNDS=1
HEADLESS=1
SKIP_PLAN=0
SKIP_RECORD=0
SKIP_CONVERT=0
DRY_RUN=0
OBJECT_IDS=()
PLAN_OVERRIDES=()

usage() {
  cat <<EOF
Usage: $0 <scene_name> [options] [-- plan.py overrides...]
       $0 --scene <scene_name> [options] [-- plan.py overrides...]

Options:
  --scene NAME             Scene to test.
  --python PATH            Python executable (default: automoma conda env if present).
  --config PATH            Planning config (default: configs/plan.yaml).
  --scene-dir PATH         Scene directory. Defaults to scene_dir in config.
  --output-root PATH       Canonical trajectory root (default: data/trajs).
  --round-root PATH        Isolated planning round root (default: data/trajs/_automoma_test_rounds).
  --statistics-dir PATH    Manifest/statistics root (default: outputs/statistics/automoma-test).
  --target-success N       Successful trajectories per object before record/convert (default: 10).
  --num-episodes N         Episodes to record/convert (default: target-success).
  --max-rounds N           Max planning rounds per object (default: 10).
  --objects "IDS"          Object ids to test, comma or space separated. Defaults to all objects in config.
  --headless               Run record headless (default).
  --no-headless            Run record with UI.
  --skip-plan              Do not run planning.
  --skip-record            Do not run recording.
  --skip-convert           Do not run LeRobot conversion.
  --no-cleanup-rounds      Keep isolated planning round directories.
  --dry-run                Print commands without executing them.
  -h, --help               Show this help.

Extra overrides after -- are forwarded to scripts/plan.py through plan_automoma_500k.py.
EOF
}

split_object_ids() {
  local value="${1//,/ }"
  local -a ids=()
  read -r -a ids <<< "${value}"
  OBJECT_IDS+=("${ids[@]}")
}

while (($#)); do
  case "$1" in
    --scene)
      SCENE_NAME="$2"; shift 2 ;;
    --scene=*)
      SCENE_NAME="${1#*=}"; shift ;;
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --python=*)
      PYTHON_BIN="${1#*=}"; shift ;;
    --config)
      CONFIG="$2"; shift 2 ;;
    --config=*)
      CONFIG="${1#*=}"; shift ;;
    --scene-dir)
      SCENE_DIR="$2"; shift 2 ;;
    --scene-dir=*)
      SCENE_DIR="${1#*=}"; shift ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --output-root=*)
      OUTPUT_ROOT="${1#*=}"; shift ;;
    --round-root)
      ROUND_ROOT="$2"; shift 2 ;;
    --round-root=*)
      ROUND_ROOT="${1#*=}"; shift ;;
    --statistics-dir)
      STATISTICS_DIR="$2"; shift 2 ;;
    --statistics-dir=*)
      STATISTICS_DIR="${1#*=}"; shift ;;
    --target-success)
      TARGET_SUCCESS="$2"; shift 2 ;;
    --target-success=*)
      TARGET_SUCCESS="${1#*=}"; shift ;;
    --num-episodes)
      NUM_EPISODES="$2"; shift 2 ;;
    --num-episodes=*)
      NUM_EPISODES="${1#*=}"; shift ;;
    --max-rounds)
      MAX_ROUNDS_PER_OBJECT="$2"; shift 2 ;;
    --max-rounds=*)
      MAX_ROUNDS_PER_OBJECT="${1#*=}"; shift ;;
    --objects)
      split_object_ids "$2"; shift 2 ;;
    --objects=*)
      split_object_ids "${1#*=}"; shift ;;
    --headless)
      HEADLESS=1; shift ;;
    --no-headless)
      HEADLESS=0; shift ;;
    --skip-plan)
      SKIP_PLAN=1; shift ;;
    --skip-record)
      SKIP_RECORD=1; shift ;;
    --skip-convert)
      SKIP_CONVERT=1; shift ;;
    --no-cleanup-rounds)
      CLEANUP_ROUNDS=0; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift
      PLAN_OVERRIDES=("$@")
      break ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
    *)
      if [[ -z "${SCENE_NAME}" ]]; then
        SCENE_NAME="$1"
        shift
      else
        echo "Unexpected positional argument: $1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -z "${SCENE_NAME}" ]]; then
  echo "Error: scene_name is required." >&2
  usage >&2
  exit 2
fi
if ! [[ "${TARGET_SUCCESS}" =~ ^[0-9]+$ ]] || ((TARGET_SUCCESS < 1)); then
  echo "--target-success must be a positive integer, got: ${TARGET_SUCCESS}" >&2
  exit 2
fi
if [[ -z "${NUM_EPISODES}" ]]; then
  NUM_EPISODES="${TARGET_SUCCESS}"
fi
if ! [[ "${NUM_EPISODES}" =~ ^[0-9]+$ ]] || ((NUM_EPISODES < 1)); then
  echo "--num-episodes must be a positive integer, got: ${NUM_EPISODES}" >&2
  exit 2
fi
if ! [[ "${MAX_ROUNDS_PER_OBJECT}" =~ ^[0-9]+$ ]] || ((MAX_ROUNDS_PER_OBJECT < 1)); then
  echo "--max-rounds must be a positive integer, got: ${MAX_ROUNDS_PER_OBJECT}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

mapfile -t OBJECT_LINES < <("${PYTHON_BIN}" - "${CONFIG}" "${OBJECT_IDS[@]}" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

config = Path(sys.argv[1])
requested = [str(item) for item in sys.argv[2:]]
cfg = OmegaConf.load(config)
objects = OmegaConf.to_container(cfg.objects, resolve=True)
object_ids = requested or list(objects.keys())

for object_id in object_ids:
    if object_id not in objects:
        raise SystemExit(f"Object id {object_id!r} not found in {config}")
    asset_type = objects[object_id]["asset_type"].lower()
    print(f"{object_id}:{asset_type}_{object_id}")
PY
)

if ((${#OBJECT_LINES[@]} == 0)); then
  echo "No objects selected." >&2
  exit 1
fi

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if ((DRY_RUN)); then
    return 0
  fi
  "$@"
}

echo "AutoMoMa quick dataset test"
echo "  repo: ${REPO_ROOT}"
echo "  python: ${PYTHON_BIN}"
echo "  scene: ${SCENE_NAME}"
echo "  target_success/object: ${TARGET_SUCCESS}"
echo "  record/convert episodes: ${NUM_EPISODES}"
echo "  objects:"
for line in "${OBJECT_LINES[@]}"; do
  echo "    ${line#*:} (${line%%:*})"
done

if ((SKIP_PLAN == 0)); then
  plan_cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/plan_automoma_500k.py"
    --config "${CONFIG}"
    --output-root "${OUTPUT_ROOT}"
    --statistics-dir "${STATISTICS_DIR}"
    --split train
    --target-per-object "${TARGET_SUCCESS}"
    --per-scene-target "${TARGET_SUCCESS}"
    --max-rounds-per-object "${MAX_ROUNDS_PER_OBJECT}"
    --round-root "${ROUND_ROOT}"
    --objects
  )
  for line in "${OBJECT_LINES[@]}"; do
    plan_cmd+=("${line%%:*}")
  done
  plan_cmd+=(--scenes "${SCENE_NAME}" --skip-statistics)
  if [[ -n "${SCENE_DIR}" ]]; then
    plan_cmd+=(--scene-dir "${SCENE_DIR}")
  fi
  if ((CLEANUP_ROUNDS)); then
    plan_cmd+=(--cleanup-rounds)
  fi
  if ((${#PLAN_OVERRIDES[@]})); then
    plan_cmd+=(--)
    plan_cmd+=("${PLAN_OVERRIDES[@]}")
  fi
  run_cmd "${plan_cmd[@]}"
fi

for line in "${OBJECT_LINES[@]}"; do
  object_name="${line#*:}"

  if ((SKIP_RECORD == 0)); then
    record_cmd=(
      bash "${REPO_ROOT}/scripts/run_pipeline.sh"
      record "${object_name}" "${SCENE_NAME}" "${NUM_EPISODES}"
    )
    if ((HEADLESS)); then
      record_cmd+=(--headless)
    else
      record_cmd+=(--no-headless)
    fi
    run_cmd "${record_cmd[@]}"
  fi

  if ((SKIP_CONVERT == 0)); then
    convert_cmd=(
      bash "${REPO_ROOT}/scripts/run_pipeline.sh"
      convert lerobot "${object_name}" "${SCENE_NAME}" "${NUM_EPISODES}"
    )
    run_cmd "${convert_cmd[@]}"
  fi
done

echo "Done."
