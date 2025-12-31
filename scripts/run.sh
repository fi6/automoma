#!/bin/bash

# --- CONFIGURATION ---
# Set what to run: ("PLAN" "RECORD") or just ("PLAN") or just ("RECORD")
RUNNING=("PLAN" "RECORD")

OBJECTS=("7221" "11622" "103634" "46197" "101773")

SCENES=(
    "scene_0_seed_0" "scene_1_seed_1" "scene_3_seed_3" "scene_7_seed_7" "scene_8_seed_8" 
    "scene_9_seed_9" "scene_10_seed_10" "scene_12_seed_12" "scene_13_seed_13" "scene_16_seed_16" 
    "scene_17_seed_17" "scene_18_seed_18" "scene_20_seed_20" "scene_21_seed_21" "scene_23_seed_23" 
    "scene_24_seed_24" "scene_25_seed_25" "scene_27_seed_27" "scene_28_seed_28" "scene_29_seed_29" 
    "scene_30_seed_30" "scene_31_seed_31" "scene_32_seed_32" "scene_33_seed_33" "scene_34_seed_34" 
    "scene_35_seed_35" "scene_36_seed_36" "scene_38_seed_38" "scene_39_seed_39" "scene_40_seed_40" 
    # "scene_41_seed_41" "scene_42_seed_42" "scene_43_seed_43" "scene_44_seed_44" "scene_45_seed_45" 
    # "scene_46_seed_46" "scene_47_seed_47" "scene_48_seed_48" "scene_52_seed_52" "scene_53_seed_53"
)

TIMEOUT_DURATION="100m"
EXP_NAME="multi_object_open"
MAX_EPISODES=1000

# --- FUNCTIONS ---

run_plan() {
    local scene=$1
    local obj=$2
    echo "[PLAN] Processing Scene: $scene, Object: $obj"
    python scripts/pipeline/1_generate_plans.py \
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
            plan_success=$?
        else
            plan_success=0 # Skip check if not running plan
        fi

        # Run Recording if in RUNNING array AND (Plan succeeded OR we didn't run Plan)
        if contains_element "RECORD" "${RUNNING[@]}"; then
            if [ $plan_success -eq 0 ]; then
                run_record "$scene" "$obj"
            else
                echo ">>> Skipping RECORD because PLAN failed/timed out."
            fi
        fi
    done
done

echo "===================================================="
echo "Pipeline finished."