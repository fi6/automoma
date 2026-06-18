#!/usr/bin/env bash
# Record AutoMoMa-30K trajectory subsets into chunked HDF5 files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

OBJECT_NAME="microwave_7221"
TRAJ_ROOT=""
WORK_ROOT=""
READY_ROOT=""
ARCHIVE_RECORD_ROOT=""
LOG_ROOT=""
EPISODES_PER_SCENE=1000
CHUNK_SIZE=25
GPU_ID=0
HEADLESS_FLAG="--headless"
MAX_ATTEMPTS=3
RETRY_DELAY_SEC=30
EXTRA_RECORD_ARGS=()
SCENES=()

usage() {
  cat <<EOF
Usage: $0 [options] --scenes SCENE [SCENE...]

Options:
  --repo-root PATH             Repo root (default: current repo).
  --object-name NAME           Object name (default: microwave_7221).
  --traj-root PATH             Root containing selected_1000 traj files.
                               Expected: <traj-root>/<scene>/train/traj_data_train.pt
  --work-root PATH             Local temp work root for recording files.
  --ready-root PATH            Local/remote directory watched by archive_hdf5_chunks.py.
  --archive-record-root PATH   Optional final archive record root; existing chunks are skipped.
  --log-root PATH              Worker log root (default: <repo>/logs/automoma-30k-record).
  --episodes-per-scene N       Episodes per scene (default: 1000).
  --chunk-size N               Episodes per HDF5 chunk (default: 25).
  --gpu-id N                   CUDA_VISIBLE_DEVICES value (default: 0).
  --max-attempts N             Attempts per chunk before failing (default: 3).
  --retry-delay-sec N          Delay between chunk attempts (default: 30).
  --no-headless                Run with UI.
  --extra-record-arg ARG       Additional run_pipeline record arg; repeatable.
  --scenes SCENE...            Scene names to record.
EOF
}

while (($#)); do
  case "$1" in
    --repo-root)
      REPO_ROOT="$2"; shift 2 ;;
    --repo-root=*)
      REPO_ROOT="${1#*=}"; shift ;;
    --object-name)
      OBJECT_NAME="$2"; shift 2 ;;
    --object-name=*)
      OBJECT_NAME="${1#*=}"; shift ;;
    --traj-root)
      TRAJ_ROOT="$2"; shift 2 ;;
    --traj-root=*)
      TRAJ_ROOT="${1#*=}"; shift ;;
    --work-root)
      WORK_ROOT="$2"; shift 2 ;;
    --work-root=*)
      WORK_ROOT="${1#*=}"; shift ;;
    --ready-root)
      READY_ROOT="$2"; shift 2 ;;
    --ready-root=*)
      READY_ROOT="${1#*=}"; shift ;;
    --archive-record-root)
      ARCHIVE_RECORD_ROOT="$2"; shift 2 ;;
    --archive-record-root=*)
      ARCHIVE_RECORD_ROOT="${1#*=}"; shift ;;
    --log-root)
      LOG_ROOT="$2"; shift 2 ;;
    --log-root=*)
      LOG_ROOT="${1#*=}"; shift ;;
    --episodes-per-scene)
      EPISODES_PER_SCENE="$2"; shift 2 ;;
    --episodes-per-scene=*)
      EPISODES_PER_SCENE="${1#*=}"; shift ;;
    --chunk-size)
      CHUNK_SIZE="$2"; shift 2 ;;
    --chunk-size=*)
      CHUNK_SIZE="${1#*=}"; shift ;;
    --gpu-id)
      GPU_ID="$2"; shift 2 ;;
    --gpu-id=*)
      GPU_ID="${1#*=}"; shift ;;
    --max-attempts)
      MAX_ATTEMPTS="$2"; shift 2 ;;
    --max-attempts=*)
      MAX_ATTEMPTS="${1#*=}"; shift ;;
    --retry-delay-sec)
      RETRY_DELAY_SEC="$2"; shift 2 ;;
    --retry-delay-sec=*)
      RETRY_DELAY_SEC="${1#*=}"; shift ;;
    --no-headless)
      HEADLESS_FLAG="--no-headless"; shift ;;
    --extra-record-arg)
      EXTRA_RECORD_ARGS+=("$2"); shift 2 ;;
    --extra-record-arg=*)
      EXTRA_RECORD_ARGS+=("${1#*=}"); shift ;;
    --scenes)
      shift
      SCENES=("$@")
      break ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ -z "$TRAJ_ROOT" ]]; then
  TRAJ_ROOT="$REPO_ROOT/data/trajs_30k/selected_1000/summit_franka/$OBJECT_NAME"
fi
if [[ -z "$WORK_ROOT" ]]; then
  WORK_ROOT="$REPO_ROOT/data/automoma/automoma-30k-work"
fi
if [[ -z "$READY_ROOT" ]]; then
  READY_ROOT="$WORK_ROOT/ready"
fi
if [[ -z "$LOG_ROOT" ]]; then
  LOG_ROOT="$REPO_ROOT/logs/automoma-30k-record"
fi
if ((${#SCENES[@]} == 0)); then
  echo "No scenes provided. Use --scenes SCENE [SCENE...]." >&2
  exit 2
fi
if ! [[ "$EPISODES_PER_SCENE" =~ ^[0-9]+$ ]] || ((EPISODES_PER_SCENE < 1)); then
  echo "--episodes-per-scene must be positive: $EPISODES_PER_SCENE" >&2
  exit 2
fi
if ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]] || ((CHUNK_SIZE < 1)); then
  echo "--chunk-size must be positive: $CHUNK_SIZE" >&2
  exit 2
fi
if ! [[ "$MAX_ATTEMPTS" =~ ^[0-9]+$ ]] || ((MAX_ATTEMPTS < 1)); then
  echo "--max-attempts must be positive: $MAX_ATTEMPTS" >&2
  exit 2
fi
if ! [[ "$RETRY_DELAY_SEC" =~ ^[0-9]+$ ]]; then
  echo "--retry-delay-sec must be a non-negative integer: $RETRY_DELAY_SEC" >&2
  exit 2
fi

mkdir -p "$WORK_ROOT" "$READY_ROOT" "$LOG_ROOT"

echo "AutoMoMa-30K record worker"
echo "  repo: $REPO_ROOT"
echo "  object: $OBJECT_NAME"
echo "  traj_root: $TRAJ_ROOT"
echo "  work_root: $WORK_ROOT"
echo "  ready_root: $READY_ROOT"
echo "  archive_record_root: ${ARCHIVE_RECORD_ROOT:-<none>}"
echo "  log_root: $LOG_ROOT"
echo "  gpu: $GPU_ID"
echo "  scenes: ${SCENES[*]}"
echo "  episodes/scene: $EPISODES_PER_SCENE"
echo "  chunk_size: $CHUNK_SIZE"
echo "  max_attempts: $MAX_ATTEMPTS"
echo "  retry_delay_sec: $RETRY_DELAY_SEC"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

for scene in "${SCENES[@]}"; do
  traj_file="$TRAJ_ROOT/$scene/train/traj_data_train.pt"
  if [[ ! -f "$traj_file" ]]; then
    echo "Missing selected trajectory file: $traj_file" >&2
    exit 1
  fi

  scene_recording_dir="$WORK_ROOT/_recording/$OBJECT_NAME/$scene"
  scene_ready_dir="$READY_ROOT/$OBJECT_NAME/$scene"
  scene_log_dir="$LOG_ROOT/$OBJECT_NAME/$scene"
  mkdir -p "$scene_recording_dir" "$scene_ready_dir" "$scene_log_dir"

  for ((start = 0; start < EPISODES_PER_SCENE; start += CHUNK_SIZE)); do
    count="$CHUNK_SIZE"
    if ((start + count > EPISODES_PER_SCENE)); then
      count=$((EPISODES_PER_SCENE - start))
    fi
    end=$((start + count - 1))
    printf -v chunk_name "chunk_%06d_%06d.hdf5" "$((start + 1))" "$((end + 1))"
    recording_file="$scene_recording_dir/$chunk_name"
    ready_file="$scene_ready_dir/$chunk_name"
    done_file="$scene_ready_dir/$chunk_name.done"
    log_file="$scene_log_dir/${chunk_name%.hdf5}.log"

    if [[ -f "$done_file" ]]; then
      echo "Skip completed chunk marker: $done_file"
      continue
    fi
    if [[ -f "$ready_file" ]]; then
      echo "Skip existing ready chunk: $ready_file"
      continue
    fi
    if [[ -n "$ARCHIVE_RECORD_ROOT" && -f "$ARCHIVE_RECORD_ROOT/$OBJECT_NAME/$scene/$chunk_name" ]]; then
      echo "Skip archived chunk: $ARCHIVE_RECORD_ROOT/$OBJECT_NAME/$scene/$chunk_name"
      continue
    fi

    attempt=1
    status=1
    : > "$log_file"
    while ((attempt <= MAX_ATTEMPTS)); do
      rm -f "$recording_file"
      echo
      echo "[$(date -Is)] Recording $OBJECT_NAME $scene $chunk_name start=$start count=$count attempt=$attempt/$MAX_ATTEMPTS"
      set +e
      (
        set -x
        bash scripts/run_pipeline.sh record "$OBJECT_NAME" "$scene" "$count" \
          "$HEADLESS_FLAG" \
          --traj_file "$traj_file" \
          --dataset_file "$recording_file" \
          --start_episode "$start" \
          --set_state \
          --interpolated 1 \
          --interpolation_type none \
          "${EXTRA_RECORD_ARGS[@]}"
      ) 2>&1 | tee -a "$log_file"
      status="${PIPESTATUS[0]}"
      set -e
      if ((status == 0)) && [[ -f "$recording_file" ]]; then
        break
      fi
      if ((status == 0)); then
        echo "Record command succeeded but did not produce $recording_file" >&2
      else
        echo "Record failed for $scene $chunk_name with status $status; see $log_file" >&2
      fi
      if ((attempt >= MAX_ATTEMPTS)); then
        break
      fi
      echo "Retrying $scene $chunk_name after ${RETRY_DELAY_SEC}s"
      sleep "$RETRY_DELAY_SEC"
      attempt=$((attempt + 1))
    done
    if ((status != 0)); then
      echo "Record failed for $scene $chunk_name after $MAX_ATTEMPTS attempts; see $log_file" >&2
      exit "$status"
    fi
    if [[ ! -f "$recording_file" ]]; then
      echo "Record command succeeded but did not produce $recording_file after $MAX_ATTEMPTS attempts" >&2
      exit 1
    fi
    mv "$recording_file" "$ready_file"
    touch "$done_file"
    echo "[$(date -Is)] Ready: $ready_file"
  done
done

echo "[$(date -Is)] Worker complete."
