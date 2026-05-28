#!/usr/bin/env bash
set -euo pipefail

RSYNC_PID="${1:?rsync pid required}"
LOCAL_RUN_DIR="${2:?local run dir required}"
REMOTE_HOST="${3:?remote host required}"
REMOTE_RUN_DIR="${4:?remote run dir required}"
PYTHON_BIN="${PYTHON_BIN:-/home/xinhai/miniconda3/envs/automoma/bin/python}"
LOG_FILE="${LOG_FILE:-$LOCAL_RUN_DIR/../legacy_archive_finalize.log}"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_FILE"
}

log "Waiting for legacy archive rsync pid=$RSYNC_PID"
while kill -0 "$RSYNC_PID" 2>/dev/null; do
    sleep 300
done

remote_hdf5="$REMOTE_RUN_DIR/shards/shard_000_start_000000_attempt_07200.hdf5"
while true; do
    log "Validating remote legacy HDF5: $REMOTE_HOST:$remote_hdf5"
    if ssh "$REMOTE_HOST" "HDF5_USE_FILE_LOCKING=FALSE $PYTHON_BIN - '$remote_hdf5'" <<'PY'
import h5py
import sys
from pathlib import Path

path = Path(sys.argv[1])
with h5py.File(path, "r") as root:
    data = root["data"]
    keys = sorted(data.keys(), key=lambda k: int(k.rsplit("_", 1)[1]))
    if len(keys) != 6602:
        raise SystemExit(f"expected 6602 demos, found {len(keys)}")
    total = sum(int(data[key].attrs.get("num_samples", 0)) for key in keys)
print(f"legacy_remote_ok demos={len(keys)} total_samples={total} size_bytes={path.stat().st_size}")
PY
    then
        break
    fi
    log "Remote validation did not pass yet; retrying in 300s"
    sleep 300
done

log "Remote validation passed; deleting local legacy run: $LOCAL_RUN_DIR"
rm -rf "$LOCAL_RUN_DIR"
df -h / | tee -a "$LOG_FILE"
log "Legacy archive finalize complete"
