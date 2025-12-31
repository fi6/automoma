#!/bin/bash

# --- CONFIGURATION ---
# Set what to run: ("PLAN" "RECORD") or just ("PLAN") or just ("RECORD")
RUNNING=("PLAN")

OBJECTS=("7221" "11622" "103634" "46197" "101773")

# RUNNING=("RECORD")

# OBJECTS=("7221")

SCENES=(
    "scene_20_seed_20"
)

TIMEOUT_DURATION="5m"
EXP_NAME="multi_object_open"
MAX_EPISODES=5

# --- FUNCTIONS ---

run_plan() {
    local scene=$1
    local obj=$2
    echo "[PLAN] Processing Scene: $scene, Object: $obj"
    timeout $TIMEOUT_DURATION python scripts/pipeline/1_generate_plans.py \
        --exp "$EXP_NAME" \
        --scene "$scene" \
        --object "$obj"
    
    local status=$?
    if [ $status -eq 124 ]; then echo ">>> [PLAN] TIMEOUT"; elif [ $status -ne 0 ]; then echo ">>> [PLAN] FAILED"; fi
    return $status
}

run_record() {
    local scene=$1
    local obj=$2
    echo "[RECORD] Rendering Scene: $scene, Object: $obj"
    python scripts/pipeline/2_render_dataset.py \
        --exp "$EXP_NAME" \
        --scene "$scene" \
        --object "$obj" \
        --headless \
        --max-episodes "$MAX_EPISODES"
    
    local status=$?
    if [ $status -ne 0 ]; then echo ">>> [RECORD] FAILED"; fi
    return $status
}

vis_lerobot_dataset() {
    local scene=$1
    local obj=$2
    local root_path="data/${EXP_NAME}/lerobot/${EXP_NAME}_${obj}_${scene}"
    
    echo "[VISUALIZE] Opening: $root_path"
    lerobot-dataset-viz \
        --repo-id "$EXP_NAME" \
        --root "$root_path" \
        --episode-index 0 \
        --video-backend pyav
}

# --- MAIN EXECUTION ---

# Check if a specific mode is requested
contains_element() {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

# Loop through all scenes and objects
for scene in "${SCENES[@]}"; do
    for obj in "${OBJECTS[@]}"; do
        echo "===================================================="
        echo "TARGET: $scene | OBJECT: $obj"
        
        # Run Planning if in RUNNING array
        if contains_element "PLAN" "${RUNNING[@]}"; then
            run_plan "$scene" "$obj"
        fi

        # Run Recording if in RUNNING array AND (Plan succeeded OR we didn't run Plan)
        if contains_element "RECORD" "${RUNNING[@]}"; then
            run_record "$scene" "$obj"
        fi
    done
done

echo "===================================================="
echo "Pipeline finished."