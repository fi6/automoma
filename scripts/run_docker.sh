#!/bin/bash

# Usage: ./scripts/run_docker.sh <object_id>

OBJECT_ID=$1
if [ -z "$OBJECT_ID" ]; then
    echo "Error: OBJECT_ID is required."
    exit 1
fi

# --- CONFIGURATION ---
# RUNNING=("PLAN" "RECORD")
RUNNING=("PLAN")

OBJECTS=("7221" "11622" "103634" "46197" "101773")

# Check if OBJECT_ID is in OBJECTS
found=0
for obj in "${OBJECTS[@]}"; do
    if [ "$obj" == "$OBJECT_ID" ]; then
        found=1
        break
    fi
done

if [ $found -eq 0 ]; then
    echo "Error: Object $OBJECT_ID not found in allowed OBJECTS list."
    echo "Allowed objects: ${OBJECTS[*]}"
    exit 1
fi

SCENES=(
    "scene_0_seed_0" "scene_1_seed_1" "scene_3_seed_3" "scene_7_seed_7" "scene_8_seed_8" 
    "scene_9_seed_9" "scene_10_seed_10" "scene_12_seed_12" "scene_13_seed_13" "scene_16_seed_16" 
    "scene_17_seed_17" "scene_18_seed_18" "scene_20_seed_20" "scene_21_seed_21" "scene_23_seed_23" 
    "scene_24_seed_24" "scene_25_seed_25" "scene_27_seed_27" "scene_28_seed_28" "scene_29_seed_29" 
    "scene_30_seed_30" "scene_31_seed_31" "scene_32_seed_32" "scene_33_seed_33" "scene_34_seed_34" 
    "scene_35_seed_35" "scene_36_seed_36" "scene_38_seed_38" "scene_39_seed_39" "scene_40_seed_40" 
    "scene_41_seed_41" "scene_42_seed_42" "scene_43_seed_43" "scene_44_seed_44" "scene_45_seed_45" 
    "scene_46_seed_46" "scene_47_seed_47" "scene_48_seed_48" "scene_52_seed_52" "scene_53_seed_53"
)

TIMEOUT_DURATION="100m"
EXP_NAME="multi_object_open"
MAX_EPISODES=1000
LOG_ROOT="/pkgs/automoma-docker/logs"

# --- FUNCTIONS ---

contains_element() {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

run_plan() {
    local scene=$1
    local obj=$2
    local log_dir="$LOG_ROOT/$scene/$obj"
    mkdir -p "$log_dir"
    local log_file="$log_dir/plan.log"
    
    echo "[PLAN] Processing Scene: $scene, Object: $obj. Log: $log_file"
    /isaac-sim/python.sh scripts/pipeline/1_generate_plans.py \
        --exp "$EXP_NAME" \
        --scene "$scene" \
        --object "$obj" > "$log_file" 2>&1
    
    local status=$?
    if [ $status -eq 124 ]; then echo ">>> [PLAN] TIMEOUT"; elif [ $status -ne 0 ]; then echo ">>> [PLAN] FAILED"; fi
    return $status
}

run_record() {
    local scene=$1
    local obj=$2
    local log_dir="$LOG_ROOT/$scene/$obj"
    mkdir -p "$log_dir"
    local log_file="$log_dir/record.log"

    echo "[RECORD] Rendering Scene: $scene, Object: $obj. Log: $log_file"

    # data/automoma-docker-1/multi_object_open/lerobot/multi_object_open_7221_scene_0_seed_0 if found, skip
    if [ -d "/pkgs/automoma-docker/data/$EXP_NAME/lerobot/${EXP_NAME}_${obj}_${scene}" ]; then
        echo ">>> [RECORD] Data already exists in docker $docker_id. Skipping rendering."
        return 0
    fi
    /isaac-sim/python.sh scripts/pipeline/2_render_dataset.py \
        --exp "$EXP_NAME" \
        --scene "$scene" \
        --object "$obj" \
        --headless \
        --max-episodes "$MAX_EPISODES" > "$log_file" 2>&1
    
    local status=$?
    if [ $status -ne 0 ]; then echo ">>> [RECORD] FAILED"; fi
    return $status
}

# --- MAIN EXECUTION ---

echo "===================================================="
echo "TARGET OBJECT: $OBJECT_ID"

# 1. Run all plans
if contains_element "PLAN" "${RUNNING[@]}"; then
    echo "Starting PLAN phase for object $OBJECT_ID..."
    for scene in "${SCENES[@]}"; do
        run_plan "$scene" "$OBJECT_ID"
    done
fi

# 2. Run all records
if contains_element "RECORD" "${RUNNING[@]}"; then
    echo "Starting RECORD phase for object $OBJECT_ID..."
    for scene in "${SCENES[@]}"; do
        run_record "$scene" "$OBJECT_ID"
    done
fi

echo "===================================================="
echo "Pipeline finished for object $OBJECT_ID."
