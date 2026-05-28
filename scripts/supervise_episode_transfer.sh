#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-/home/xinhai/miniconda3/envs/automoma/bin/python}"
WATCHER_SCRIPT="${WATCHER_SCRIPT:-/home/xinhai/projects/automoma/scripts/watch_transfer_hdf5_episodes.py}"
INTERVAL_SEC="${INTERVAL_SEC:-60}"

LOCAL_DIR="${1:?local episode dir required}"
REMOTE_HOST="${2:?remote host required}"
REMOTE_DIR="${3:?remote episode dir required}"
LEDGER="${4:?ledger path required}"

while true; do
    "$PYTHON_BIN" "$WATCHER_SCRIPT" \
        --local_dir "$LOCAL_DIR" \
        --host "$REMOTE_HOST" \
        --remote_dir "$REMOTE_DIR" \
        --ledger "$LEDGER" \
        --interval_sec "$INTERVAL_SEC" \
        --min_age_sec 60 \
        --once
    sleep "$INTERVAL_SEC"
done
