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
  --raw_alpha "${FIXED_COMPOSE_RAW_ALPHA:-0.18}" \
  --alpha "${FIXED_COMPOSE_ALPHA:-0.28}" \
  --workspace_alpha 0.0 \
  --max_frames "${FIXED_COMPOSE_MAX_FRAMES:-0}" \
  --output_dir "$OUTPUT_ABS/fixed_base/composites"

"${AUTOMOMA_PYTHON:-python}" "$SCRIPT_DIR/compose_layered_trajectory_sources.py" \
  --input_root "$OUTPUT_ABS/mobile_base" \
  --views overview close \
  --raw_alpha "${MOBILE_COMPOSE_RAW_ALPHA:-0.16}" \
  --alpha "${MOBILE_COMPOSE_ALPHA:-0.24}" \
  --workspace_alpha "${MOBILE_WORKSPACE_COMPOSE_ALPHA:-0.035}" \
  --max_frames "${MOBILE_COMPOSE_MAX_FRAMES:-0}" \
  --output_dir "$OUTPUT_ABS/mobile_base/composites"

for panel in fixed_base mobile_base; do
  for view in overview close; do
    cp "$OUTPUT_ABS/$panel/composites/${view}_all.png" \
      "$OUTPUT_ABS/$panel/${panel}_${view}_all.png"
    cp "$OUTPUT_ABS/$panel/composites/${view}_all_ghost.png" \
      "$OUTPUT_ABS/$panel/${panel}_${view}_all_ghost.png"
  done
  rm -f "$OUTPUT_ABS/$panel/${panel}_topdown_ghost.png"
done
rm -f "$OUTPUT_ABS/fixed_vs_mobile_actual_ghost.png"

echo "[layered] outputs written under $OUTPUT_ABS"
