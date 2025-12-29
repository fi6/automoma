# #!/bin/bash

# # Define the list of object IDs from your OBJECT_CONFIG_MAP
# OBJECT_IDS=("7221" "11622" "103634" "46197" "101773")
# # OBJECT_IDS=("46197")


# # Set the timeout duration (6 minutes)
# TIMEOUT_DURATION="5m"

# echo "Starting pipeline for ${#OBJECT_IDS[@]} objects..."

# for obj_id in "${OBJECT_IDS[@]}"; do
#     echo "----------------------------------------------------"
#     echo "Running object_id: $obj_id"
    
#     # Run the python script with the timeout command
#     # --scene scene_3_seed_3 and --exp multi_object_open as requested
#     timeout $TIMEOUT_DURATION python scripts/pipeline/1_generate_plans.py \
#         --exp multi_object_open \
#         --scene scene_3_seed_3 \
#         --object "$obj_id"
    
#     # Capture the exit status of the timeout/python command
#     exit_status=$?
    
#     if [ $exit_status -eq 124 ]; then
#         echo "Status: TIMEOUT (Exceeded $TIMEOUT_DURATION)"
#     elif [ $exit_status -eq 0 ]; then
#         echo "Status: SUCCESS"
#     else
#         echo "Status: FAILED (Exit Code: $exit_status)"
#     fi
# done

# echo "----------------------------------------------------"
# echo "All tasks completed."


#!/bin/bash

# Define the list of object IDs from the OBJECT_CONFIG_MAP
OBJECT_IDS=("7221" "11622" "103634" "46197" "10944" "101773")

# Set the timeout duration for the generation script
TIMEOUT_DURATION="6m"

echo "Starting sequential pipeline for ${#OBJECT_IDS[@]} objects..."

for obj_id in "${OBJECT_IDS[@]}"; do
    echo "===================================================="
    echo "PROCESSING OBJECT: $obj_id"
    echo "===================================================="
    
    # Task 1: Generate Plans (with 6-minute timeout)
    # echo "[1/2] Generating plans for $obj_id..."
    # timeout $TIMEOUT_DURATION python scripts/pipeline/1_generate_plans.py \
    #     --exp multi_object_open \
    #     --scene scene_3_seed_3 \
    #     --object "$obj_id"
    
    # gen_status=$?
    
    # if [ $gen_status -eq 124 ]; then
    #     echo ">>> Status: Generation TIMED OUT after $TIMEOUT_DURATION. Skipping render."
    #     continue 
    # elif [ $gen_status -ne 0 ]; then
    #     echo ">>> Status: Generation FAILED (Exit Code: $gen_status). Skipping render."
    #     continue
    # fi

    # Task 2: Render Dataset (Runs only if generation succeeded)
    echo "[2/2] Rendering dataset for $obj_id..."
    python scripts/pipeline/2_render_dataset.py \
        --exp multi_object_open \
        --scene scene_3_seed_3 \
        --object "$obj_id" \
        --headless \
        --max-episodes 10
    
    render_status=$?
    
    if [ $render_status -eq 0 ]; then
        echo ">>> Status: Object $obj_id completed successfully."
    else
        echo ">>> Status: Rendering FAILED for $obj_id (Exit Code: $render_status)."
    fi
done

echo "===================================================="
echo "Pipeline execution finished."



lerobot-dataset-viz \
    --repo-id multi_object_open \
    --root data/multi_object_open/lerobot/multi_object_open_11622_scene_3_seed_3 \
    --episode-index 0 \
    --video-backend pyav