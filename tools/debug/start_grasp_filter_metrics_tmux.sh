#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-grasp_filter_metrics}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/debug/grasp_filter/$RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-1}"
SCENES_PER_OBJECT="${SCENES_PER_OBJECT:-5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUTPUT_DIR"

CMD="cd '$REPO_ROOT' && '$PYTHON_BIN' tools/debug/run_grasp_filter_metrics.py --output-dir '$OUTPUT_DIR' --gpu '$GPU' --scenes-per-object '$SCENES_PER_OBJECT' --keep-going $EXTRA_ARGS 2>&1 | tee -a '$OUTPUT_DIR/run.log'"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION" >&2
    echo "Attach with: tmux attach -t $SESSION" >&2
    exit 1
fi

tmux new-session -d -s "$SESSION" "$CMD"
echo "$SESSION" > "$OUTPUT_DIR/tmux_session.txt"
echo "$CMD" > "$OUTPUT_DIR/command.sh"
chmod +x "$OUTPUT_DIR/command.sh"
echo "Started tmux session: $SESSION"
echo "Output dir: $OUTPUT_DIR"
echo "Attach: tmux attach -t $SESSION"
