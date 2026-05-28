#!/usr/bin/env bash
set -uo pipefail

RUN_DIR="${1:?run dir required}"
RECORD_LOG_DIR="${2:?record log dir required}"
ISAAC_KIT_LOG_DIR="${3:?isaac kit log dir required}"
INTERVAL_SEC="${INTERVAL_SEC:-300}"
MAX_LOG_BYTES="${MAX_LOG_BYTES:-20971520}"
KEEP_TAIL_LINES="${KEEP_TAIL_LINES:-2000}"

shrink_file() {
    local path="$1"
    [[ -f "$path" ]] || return 0
    local size
    size=$(stat -c '%s' "$path" 2>/dev/null || echo 0)
    if (( size > MAX_LOG_BYTES )); then
        local tmp
        tmp="$(mktemp)"
        tail -n "$KEEP_TAIL_LINES" "$path" > "$tmp" 2>/dev/null || true
        cat "$tmp" > "$path"
        rm -f "$tmp"
    fi
}

while true; do
    shrink_file "$RUN_DIR/record.stdout.log"
    find "$RECORD_LOG_DIR" -maxdepth 1 -type f -name '*.log' -size +"$MAX_LOG_BYTES"c -print0 2>/dev/null |
        while IFS= read -r -d '' path; do
            shrink_file "$path"
        done

    find "$ISAAC_KIT_LOG_DIR" -type f -name 'kit_*.log' -mmin +20 -delete 2>/dev/null || true
    find /tmp -maxdepth 1 -type d -name 'tmp*' -mmin +60 -exec rm -rf {} + 2>/dev/null || true
    df -h / /dev/shm > "$RUN_DIR/disk_status.txt" 2>/dev/null || true
    sleep "$INTERVAL_SEC"
done
