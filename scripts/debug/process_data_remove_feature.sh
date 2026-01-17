#!/bin/bash

# Directory containing the datasets (based on your pwd and ls output)
BASE_DIR="/home/xinhai/projects/automoma/data/multi_object_open/lerobot"

# Features to remove
# Note: We escape the outer brackets and quotes for safety in the shell command
FEATURES="['observation.depth.ego_topdown', 'observation.depth.ego_wrist', 'observation.depth.fix_local', 'observation.eef']"

# Initialize counters
total_count=0
processed_count=0

# Get initial disk usage
# usage of du -sh: -s for summary, -h for human readable
echo "Calculating initial disk usage..."
INITIAL_SIZE_HUMAN=$(du -sh "$BASE_DIR" | cut -f1)
INITIAL_SIZE_BYTES=$(du -sb "$BASE_DIR" 2>/dev/null | cut -f1) # Try bytes for precise calc if supported

echo "---------------------------------------------------"
echo "Starting Bulk Processing"
echo "Target Directory: $BASE_DIR"
echo "Initial Disk Usage: $INITIAL_SIZE_HUMAN"
echo "---------------------------------------------------"

# Loop through all directories matching the pattern
for dataset_path in "$BASE_DIR"/multi_object_open_*; do
    if [ -d "$dataset_path" ]; then
        dirname=$(basename "$dataset_path")

        # Skip directory if it ends with _old (backup folders)
        if [[ "$dirname" == *"_old" ]]; then
            echo "Skipping backup directory: $dirname"
            continue
        fi

        ((total_count++))
        echo "[$total_count] Processing: $dirname"

        # Run the python script
        # We pass the absolute path as repo_id
        python -m lerobot.scripts.lerobot_edit_dataset \
            --repo_id "$dataset_path" \
            --operation.type remove_feature \
            --operation.backup false \
            --operation.ignore_invalid true \
            --operation.feature_names "$FEATURES"

        # Check if the command succeeded
        if [ $? -eq 0 ]; then
            echo "✓ Success: $dirname"
            ((processed_count++))
        else
            echo "✗ Failed: $dirname"
        fi

        # Remove the huggingface cache
        echo "Removing Huggingface cache..."
        rm -rf ~/.cache/huggingface/datasets/*

        echo "---------------------------------------------------"
    fi
done

# Get final disk usage
echo "Calculating final disk usage..."
FINAL_SIZE_HUMAN=$(du -sh "$BASE_DIR" | cut -f1)
FINAL_SIZE_BYTES=$(du -sb "$BASE_DIR" 2>/dev/null | cut -f1)

# Calculate reduction if byte counts are available
REDUCTION_MSG=""
if [ ! -z "$INITIAL_SIZE_BYTES" ] && [ ! -z "$FINAL_SIZE_BYTES" ]; then
    DIFF=$((INITIAL_SIZE_BYTES - FINAL_SIZE_BYTES))
    DIFF_HUMAN=$(numfmt --to=iec $DIFF)
    REDUCTION_MSG="(Reduced by $DIFF_HUMAN)"
fi

echo "==================================================="
echo "SUMMARY"
echo "==================================================="
echo "Total Datasets Found:      $total_count"
echo "Successfully Processed:    $processed_count"
echo "Initial Disk Usage:        $INITIAL_SIZE_HUMAN"
echo "Final Disk Usage:          $FINAL_SIZE_HUMAN $REDUCTION_MSG"
echo "==================================================="
