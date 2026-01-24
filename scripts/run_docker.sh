#!/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <OBJECT_ID>"
    echo "Allowed IDs: 7221, 11622, 103634, 46197, 101773"
    exit 1
fi

OBJECT_ID=$1

# Validation for the five specific IDs
case "$OBJECT_ID" in
    "7221" | "11622" | "103634" | "46197" | "101773")
        echo "Valid Object ID detected: $OBJECT_ID. Starting pipeline..."
        
        # Step 1: Data Planning
        /isaac-sim/python.sh scripts/pipeline_plan.py \
            --scene_dir assets/scene/infinigen/kitchen_1130 \
            --plan_dir output/collect_0123/traj \
            --robot_name summit_franka \
            --object_id "$OBJECT_ID"

        # Step 2: Data Collection
        /isaac-sim/python.sh scripts/pipeline_collect.py \
            --scene_dir assets/scene/infinigen/kitchen_1130 \
            --plan_dir output/collect_0123/traj \
            --robot_name summit_franka \
            --num_episodes 1000 \
            --object_id "$OBJECT_ID"
            
        echo "Pipeline execution for $OBJECT_ID finished."
        ;;
    *)
        # Error handling for invalid IDs
        echo "Error: '$OBJECT_ID' is not a valid ID."
        echo "Please use one of the following: 7221, 11622, 103634, 46197, 101773"
        exit 1
        ;;
esac