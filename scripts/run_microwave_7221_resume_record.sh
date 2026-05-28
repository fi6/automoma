#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xinhai/miniconda3/envs/automoma/bin/python}"

OBJECT_NAME="${OBJECT_NAME:-microwave_7221}"
SCENE_NAME="${SCENE_NAME:-scene_0_seed_0}"
START_EPISODE="${START_EPISODE:-6602}"
NUM_EPISODES="${NUM_EPISODES:-12000}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-$REPO_ROOT/logs/formal_record_split/${OBJECT_NAME}/${SCENE_NAME}/${RUN_ID}}"
LOCAL_EPISODE_DIR="${LOCAL_EPISODE_DIR:-$LOCAL_RUN_DIR/episodes}"
REMOTE_HOST="${REMOTE_HOST:-thalia}"
REMOTE_BASE="${REMOTE_BASE:-/media/xinhai/GIANT/Research/AutoMoMa/archive/repo-temp/2026-05-26}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-$REMOTE_BASE/${OBJECT_NAME}_${SCENE_NAME}_resume_${RUN_ID}}"
REMOTE_EPISODE_DIR="$REMOTE_RUN_DIR/episodes"

LEGACY_LOCAL="${LEGACY_LOCAL:-$REPO_ROOT/logs/formal_record/${OBJECT_NAME}/${SCENE_NAME}/20260523_093931}"
LEGACY_REMOTE="${LEGACY_REMOTE:-$REMOTE_BASE/${OBJECT_NAME}_${SCENE_NAME}_legacy_20260523_093931}"

mkdir -p "$LOCAL_RUN_DIR" "$LOCAL_EPISODE_DIR"
LOG_FILE="$LOCAL_RUN_DIR/orchestrator.log"
TRANSFER_LEDGER="$LOCAL_RUN_DIR/transfer_ledger.jsonl"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_FILE"
}

verify_legacy_remote() {
    local hdf5_path="$LEGACY_REMOTE/shards/shard_000_start_000000_attempt_07200.hdf5"
    ssh "$REMOTE_HOST" "HDF5_USE_FILE_LOCKING=FALSE $PYTHON_BIN - '$hdf5_path'" <<'PY'
import h5py
import sys
from pathlib import Path
p = Path(sys.argv[1])
with h5py.File(p, 'r') as f:
    data = f['data']
    keys = sorted(data.keys(), key=lambda k: int(k.rsplit('_', 1)[1]))
    assert len(keys) == 6602, len(keys)
print('legacy_hdf5_ok demos=6602')
PY
}

cleanup_legacy_local() {
    log "Stopping legacy paused processes if present"
    for pid in 1816273 1816248 1817234; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -CONT "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 5
    for pid in 1816273 1816248 1817234; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done

    if [[ -d "$LEGACY_LOCAL" ]]; then
        log "Deleting local legacy run: $LEGACY_LOCAL"
        rm -rf "$LEGACY_LOCAL"
    fi
}

start_transfer_watcher() {
    log "Starting transfer watcher"
    nohup "$PYTHON_BIN" "$REPO_ROOT/scripts/watch_transfer_hdf5_episodes.py" \
        --local_dir "$LOCAL_EPISODE_DIR" \
        --host "$REMOTE_HOST" \
        --remote_dir "$REMOTE_EPISODE_DIR" \
        --ledger "$TRANSFER_LEDGER" \
        --interval_sec 300 \
        --min_age_sec 60 \
        > "$LOCAL_RUN_DIR/transfer_watcher.stdout.log" 2>&1 &
    echo "$!" > "$LOCAL_RUN_DIR/transfer_watcher.pid"
}

run_record() {
    log "Starting per-episode record from start_episode=$START_EPISODE num_episodes=$NUM_EPISODES on CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
    (
        cd "$REPO_ROOT"
        export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
        export RECORD_INTERPOLATED=5
        export RECORD_INTERPOLATION_TYPE=cubic
        export RECORD_DECIMATION=1
        export RECORD_INIT_STEPS=5
        export AUTOMOMA_ROBOT_OBJECT_STATIC_FRICTION=1.0
        export AUTOMOMA_ROBOT_OBJECT_DYNAMIC_FRICTION=1.0
        bash scripts/run_pipeline.sh record "$OBJECT_NAME" "$SCENE_NAME" "$NUM_EPISODES" \
            --record_format episodes \
            --dataset_file "$LOCAL_EPISODE_DIR" \
            --start_episode "$START_EPISODE" \
            --headless \
            --validate_record_success \
            --keep_failed_record_demos
    ) >> "$LOCAL_RUN_DIR/record.stdout.log" 2>&1
}

log "Archiving legacy run to $REMOTE_HOST:$LEGACY_REMOTE"
ssh "$REMOTE_HOST" "mkdir -p '$LEGACY_REMOTE'"
rsync -aH --partial --append-verify -z --compress-choice=zstd --compress-level=1 \
    --info=progress2 "$LEGACY_LOCAL/" "$REMOTE_HOST:$LEGACY_REMOTE/" >> "$LOG_FILE" 2>&1

log "Verifying legacy remote HDF5"
verify_legacy_remote >> "$LOG_FILE" 2>&1

cleanup_legacy_local
df -h / >> "$LOG_FILE" 2>&1 || true

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_EPISODE_DIR'"
start_transfer_watcher
run_record
