#!/usr/bin/env bash
# Render reproducible source layers and composites for the poster trajectory figure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/paper/poster/actual_trajectory_layers}"
if [[ "$OUTPUT_ROOT" = /* ]]; then
  OUTPUT_ABS="$OUTPUT_ROOT"
else
  OUTPUT_ABS="$REPO_ROOT/$OUTPUT_ROOT"
fi

bash "$SCRIPT_DIR/run_actual_ghost_render.sh" \
  --output_root "$OUTPUT_ROOT" \
  --render_views overview close \
  --export_layer_sources \
  --image_width "${IMAGE_WIDTH:-1600}" \
  --image_height "${IMAGE_HEIGHT:-1050}" \
  --mobile_traj_count "${MOBILE_EPISODES:-3}" \
  --mobile_keyframes "${MOBILE_KEYFRAMES:-8}" \
  --mobile_workspace_ghosts "${MOBILE_WORKSPACE_GHOSTS:-18}" \
  --fixed_episode_count "${FIXED_EPISODES:-3}" \
  --fixed_keyframes "${FIXED_KEYFRAMES:-8}" \
  --fixed_arm_ghosts "${FIXED_RANDOM_GHOSTS:-0}" \
  "$@"

"${AUTOMOMA_PYTHON:-python}" "$SCRIPT_DIR/compose_layered_trajectory_sources.py" \
  --input_root "$OUTPUT_ABS/fixed_base" \
  --views overview close \
  --alpha "${FIXED_COMPOSE_ALPHA:-0.28}" \
  --workspace_alpha 0.0 \
  --max_frames "${FIXED_COMPOSE_MAX_FRAMES:-0}" \
  --output_dir "$OUTPUT_ABS/fixed_base/composites"

"${AUTOMOMA_PYTHON:-python}" "$SCRIPT_DIR/compose_layered_trajectory_sources.py" \
  --input_root "$OUTPUT_ABS/mobile_base" \
  --views overview close \
  --alpha "${MOBILE_COMPOSE_ALPHA:-0.24}" \
  --workspace_alpha "${MOBILE_WORKSPACE_COMPOSE_ALPHA:-0.035}" \
  --max_frames "${MOBILE_COMPOSE_MAX_FRAMES:-0}" \
  --output_dir "$OUTPUT_ABS/mobile_base/composites"

echo "[layered] outputs written under $OUTPUT_ABS"
