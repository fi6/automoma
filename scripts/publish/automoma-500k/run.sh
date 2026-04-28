#!/usr/bin/env bash
# Launch AutoMoMa-500k planning workers across one or more GPUs.
#
# Example:
#   bash scripts/public/automoma-500k/run.sh --GPUS 2
#   bash scripts/public/automoma-500k/run.sh --GPUS 2 -- planner.traj.batch_size=10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DEFAULT_PYTHON="/home/xinhai/miniconda3/envs/automoma/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON:-python}"
fi

GPUS=1
START_GPU=0
CONFIG="${REPO_ROOT}/configs/plan.yaml"
SCENE_DIR=""
OUTPUT_ROOT="${REPO_ROOT}/data/trajs"
TARGET_PER_OBJECT=100000
SPLIT="train"
LOG_DIR="${REPO_ROOT}/logs/automoma-500k"
STATISTICS_DIR="${REPO_ROOT}/outputs/statistics/automoma-500k"
ROUND_ROOT_PARENT="${REPO_ROOT}/data/trajs"
CLEANUP_ROUNDS=1
MONITOR=1
DRY_RUN=0
FORCE=0
EXTRA_OVERRIDES=()

usage() {
  cat <<EOF
Usage: $0 [options] [-- plan.py overrides...]

Options:
  --GPUS N                 Number of GPUs/workers to run in parallel (default: 1).
  --START-GPU N            First physical GPU id to use (default: 0).
  --python PATH            Python executable (default: automoma conda env if present).
  --config PATH            Planning config (default: configs/plan.yaml).
  --scene-dir PATH         Scene directory. Defaults to scene_dir in config.
  --output-root PATH       Canonical trajectory root (default: data/trajs).
  --target-per-object N    Global target per object across all workers (default: 100000).
  --split NAME             Dataset split/mode (default: train).
  --log-dir PATH           Worker and monitor log directory (default: logs/automoma-500k).
  --statistics-dir PATH    Statistics/manifest root (default: outputs/statistics/automoma-500k).
  --round-root-parent PATH Parent for per-GPU isolated round dirs (default: data/trajs).
  --no-cleanup-rounds      Keep isolated round directories after successful merge.
  --no-monitor             Do not start the detached monitor process.
  --dry-run                Pass --dry-run to plan_automoma_500k.py.
  --force                  Start even if gpu*.pid files point at live processes.
  -h, --help               Show this help.

Extra overrides after -- are forwarded to scripts/plan.py through plan_automoma_500k.py.
EOF
}

while (($#)); do
  case "$1" in
    --GPUS|--gpus)
      GPUS="$2"; shift 2 ;;
    --GPUS=*|--gpus=*)
      GPUS="${1#*=}"; shift ;;
    --START-GPU|--start-gpu)
      START_GPU="$2"; shift 2 ;;
    --START-GPU=*|--start-gpu=*)
      START_GPU="${1#*=}"; shift ;;
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
    --target-per-object)
      TARGET_PER_OBJECT="$2"; shift 2 ;;
    --target-per-object=*)
      TARGET_PER_OBJECT="${1#*=}"; shift ;;
    --split)
      SPLIT="$2"; shift 2 ;;
    --split=*)
      SPLIT="${1#*=}"; shift ;;
    --log-dir)
      LOG_DIR="$2"; shift 2 ;;
    --log-dir=*)
      LOG_DIR="${1#*=}"; shift ;;
    --statistics-dir)
      STATISTICS_DIR="$2"; shift 2 ;;
    --statistics-dir=*)
      STATISTICS_DIR="${1#*=}"; shift ;;
    --round-root-parent)
      ROUND_ROOT_PARENT="$2"; shift 2 ;;
    --round-root-parent=*)
      ROUND_ROOT_PARENT="${1#*=}"; shift ;;
    --no-cleanup-rounds)
      CLEANUP_ROUNDS=0; shift ;;
    --no-monitor)
      MONITOR=0; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --force)
      FORCE=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift
      EXTRA_OVERRIDES=("$@")
      break ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if ! [[ "${GPUS}" =~ ^[0-9]+$ ]] || ((GPUS < 1)); then
  echo "--GPUS must be a positive integer, got: ${GPUS}" >&2
  exit 2
fi
if ! [[ "${START_GPU}" =~ ^[0-9]+$ ]]; then
  echo "--START-GPU must be a non-negative integer, got: ${START_GPU}" >&2
  exit 2
fi
if ! [[ "${TARGET_PER_OBJECT}" =~ ^[0-9]+$ ]] || ((TARGET_PER_OBJECT < 1)); then
  echo "--target-per-object must be a positive integer, got: ${TARGET_PER_OBJECT}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${STATISTICS_DIR}" "${ROUND_ROOT_PARENT}"

if ((FORCE == 0)); then
  for ((worker = 0; worker < GPUS; worker++)); do
    pid_file="${LOG_DIR}/gpu${worker}.pid"
    if [[ -s "${pid_file}" ]]; then
      old_pid="$(cat "${pid_file}")"
      if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
        echo "Refusing to start: ${pid_file} points at live process ${old_pid}." >&2
        echo "Stop it first, remove the pid file, or pass --force." >&2
        exit 1
      fi
    fi
  done
fi

SCENE_DIR_RESOLVED="$("${PYTHON_BIN}" - "${REPO_ROOT}" "${CONFIG}" "${SCENE_DIR}" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

repo = Path(sys.argv[1])
config = Path(sys.argv[2])
scene_dir_arg = sys.argv[3]
cfg = OmegaConf.load(config)
scene_dir = Path(scene_dir_arg) if scene_dir_arg else Path(cfg.get("scene_dir", "assets/scene/infinigen/kitchen_1130"))
if not scene_dir.is_absolute():
    scene_dir = repo / scene_dir
print(scene_dir)
PY
)"

mapfile -t ALL_SCENES < <("${PYTHON_BIN}" - "${SCENE_DIR_RESOLVED}" <<'PY'
import sys
from pathlib import Path

scene_dir = Path(sys.argv[1])
if not scene_dir.exists():
    raise SystemExit(f"No scene directory found: {scene_dir}")
for scene in sorted(p.name for p in scene_dir.iterdir() if p.is_dir()):
    print(scene)
PY
)

if ((${#ALL_SCENES[@]} == 0)); then
  echo "No scenes found under ${SCENE_DIR_RESOLVED}" >&2
  exit 1
fi

WORKER_TARGET=$(((TARGET_PER_OBJECT + GPUS - 1) / GPUS))

write_worker_command() {
  local worker="$1"
  local gpu_id="$2"
  local cmd_file="${LOG_DIR}/gpu${worker}.cmd.sh"
  local round_root="${ROUND_ROOT_PARENT}/_automoma_500k_rounds_gpu${worker}"
  local worker_statistics_dir="${STATISTICS_DIR}/gpu${worker}"
  local -a scenes=()

  for ((idx = worker; idx < ${#ALL_SCENES[@]}; idx += GPUS)); do
    scenes+=("${ALL_SCENES[idx]}")
  done

  if ((${#scenes[@]} == 0)); then
    echo "Worker ${worker} has no scenes assigned; reduce --GPUS." >&2
    exit 1
  fi

  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf "cd %q\n" "${REPO_ROOT}"
    printf "export PATH=%q:\$PATH\n" "$(dirname "${PYTHON_BIN}")"
    printf "export CUDA_HOME=%q\n" "${CUDA_HOME:-/usr/local/cuda}"
    printf "export TORCH_CUDA_ARCH_LIST=%q\n" "${TORCH_CUDA_ARCH_LIST:-8.9}"
    printf "export CUDA_VISIBLE_DEVICES=%q\n" "${gpu_id}"
    printf "exec %q %q" "${PYTHON_BIN}" "${SCRIPT_DIR}/plan_automoma_500k.py"
    printf " --config %q" "${CONFIG}"
    printf " --scene-dir %q" "${SCENE_DIR_RESOLVED}"
    printf " --output-root %q" "${OUTPUT_ROOT}"
    printf " --statistics-dir %q" "${worker_statistics_dir}"
    printf " --split %q" "${SPLIT}"
    printf " --target-per-object %q" "${WORKER_TARGET}"
    printf " --round-root %q" "${round_root}"
    printf " --scenes"
    for scene in "${scenes[@]}"; do
      printf " %q" "${scene}"
    done
    printf " --skip-statistics"
    if ((CLEANUP_ROUNDS)); then
      printf " --cleanup-rounds"
    fi
    if ((DRY_RUN)); then
      printf " --dry-run"
    fi
    if ((${#EXTRA_OVERRIDES[@]})); then
      printf " --"
      for override in "${EXTRA_OVERRIDES[@]}"; do
        printf " %q" "${override}"
      done
    fi
    printf "\n"
  } > "${cmd_file}"
  chmod +x "${cmd_file}"

  printf "%s\n" "${scenes[*]}" > "${LOG_DIR}/gpu${worker}.scenes.txt"
  echo "${cmd_file}"
}

echo "AutoMoMa-500k launcher"
echo "  repo: ${REPO_ROOT}"
echo "  python: ${PYTHON_BIN}"
echo "  config: ${CONFIG}"
echo "  scene_dir: ${SCENE_DIR_RESOLVED}"
echo "  scenes: ${#ALL_SCENES[@]}"
echo "  gpus/workers: ${GPUS}"
echo "  global target/object: ${TARGET_PER_OBJECT}"
echo "  worker target/object: ${WORKER_TARGET}"
echo "  log_dir: ${LOG_DIR}"

for ((worker = 0; worker < GPUS; worker++)); do
  gpu_id=$((START_GPU + worker))
  cmd_file="$(write_worker_command "${worker}" "${gpu_id}")"
  log_file="${LOG_DIR}/gpu${worker}.log"
  : > "${log_file}"
  setsid /usr/bin/bash "${cmd_file}" > "${log_file}" 2>&1 &
  pid="$!"
  echo "${pid}" > "${LOG_DIR}/gpu${worker}.pid"
  echo "Started worker ${worker} on physical GPU ${gpu_id}: pid=${pid}"
  echo "  log: ${log_file}"
  echo "  cmd: ${cmd_file}"
  echo "  scenes: ${LOG_DIR}/gpu${worker}.scenes.txt"
done

if ((MONITOR)); then
  monitor_cmd="${LOG_DIR}/monitor.cmd.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf "cd %q\n" "${REPO_ROOT}"
    printf "export PATH=%q:\$PATH\n" "$(dirname "${PYTHON_BIN}")"
    printf "export CUDA_HOME=%q\n" "${CUDA_HOME:-/usr/local/cuda}"
    printf "export TORCH_CUDA_ARCH_LIST=%q\n" "${TORCH_CUDA_ARCH_LIST:-8.9}"
    printf "LOG_DIR=%q\n" "${LOG_DIR}"
    printf "GPUS=%q\n" "${GPUS}"
    printf "PYTHON_BIN=%q\n" "${PYTHON_BIN}"
    printf "CONFIG=%q\n" "${CONFIG}"
    printf "OUTPUT_ROOT=%q\n" "${OUTPUT_ROOT}"
    printf "STATISTICS_DIR=%q\n" "${STATISTICS_DIR}"
    printf "SCENE_DIR=%q\n" "${SCENE_DIR_RESOLVED}"
    printf "SPLIT=%q\n" "${SPLIT}"
    printf "TARGET_PER_OBJECT=%q\n" "${TARGET_PER_OBJECT}"
    cat <<'EOS'
while true; do
  echo "===== $(date -Is) ====="
  live=0
  pids=()
  for ((worker = 0; worker < GPUS; worker++)); do
    pid_file="${LOG_DIR}/gpu${worker}.pid"
    if [[ -s "${pid_file}" ]]; then
      pid="$(cat "${pid_file}")"
      echo "worker ${worker}: pid=${pid}"
      pids+=("${pid}")
      if kill -0 "${pid}" 2>/dev/null; then
        live=1
      fi
    else
      echo "worker ${worker}: no pid file"
    fi
  done
  if ((${#pids[@]})); then
    ps -p "$(IFS=,; echo "${pids[*]}")" -o pid,ppid,sid,stat,etime,%cpu,%mem,cmd 2>/dev/null || true
  fi
  pgrep -af "scripts/plan.py.*_automoma_500k_rounds_gpu" || true
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  echo "canonical_files=$(find "${OUTPUT_ROOT}/summit_franka" -path "*traj_data_${SPLIT}.pt" -type f 2>/dev/null | wc -l)"
  echo "round_files=$(find data/trajs/_automoma_500k_rounds_gpu* -path "*traj_data.pt" -type f 2>/dev/null | wc -l)"
  echo "disk=$(df -h . | tail -n 1)"
  if ((live == 0)); then
    echo "workers exited; running final statistics"
    "${PYTHON_BIN}" scripts/public/automoma-500k/trajectory_statistics.py \
      --config "${CONFIG}" \
      --root "${OUTPUT_ROOT}" \
      --output-dir "${STATISTICS_DIR}" \
      --scene-dir "${SCENE_DIR}" \
      --split "${SPLIT}" \
      --target-per-object "${TARGET_PER_OBJECT}" || true
    echo "monitor finished at $(date -Is)"
    break
  fi
  sleep 300
done
EOS
  } > "${monitor_cmd}"
  chmod +x "${monitor_cmd}"

  : > "${LOG_DIR}/monitor.log"
  setsid /usr/bin/bash "${monitor_cmd}" > "${LOG_DIR}/monitor.log" 2>&1 &
  echo "$!" > "${LOG_DIR}/monitor.pid"
  echo "Started monitor: pid=$(cat "${LOG_DIR}/monitor.pid")"
  echo "  log: ${LOG_DIR}/monitor.log"
  echo "  cmd: ${monitor_cmd}"
fi

