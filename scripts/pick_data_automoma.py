



#!/usr/bin/env python3
"""
Pick Data Automoma Script

This script collects statistics and picks camera data from automoma pipeline collection results.
It follows the structure from pipeline_collect.py where data is organized as:
output/collect/traj/{robot_name}/{scene_name}/{object_id}/camera_data/episode*.hdf5

Usage:
    # Collect statistics of available camera data
    python pick_data_automoma.py --mode collect --output_dir output/collect/traj

    # Pick data according to configuration (after manually editing the pick config)
    python pick_data_automoma.py --mode pick --output_dir output/collect/traj

    # Use symbolic links instead of copying files to save space
    python pick_data_automoma.py --mode pick --output_dir output/collect/traj --link

    # Use sequential picking instead of random picking
    python pick_data_automoma.py --mode pick --output_dir output/collect/traj --sequential
"""

import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def collect_data_statistics(output_dir, config_path):
    """
    Collect statistics of hdf5 files in output directory following automoma structure
    Structure: output_dir/{robot_name}/{scene_name}/{object_id}/camera_data/episode*.hdf5
    """
    statistics = {
        "task_name": "automoma_camera_collection",
        "statics": defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    }
    
    # Traverse output directory structure
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
        return
    
    # Find all robot directories
    for robot_name in os.listdir(output_dir):
        robot_path = os.path.join(output_dir, robot_name)
        if not os.path.isdir(robot_path):
            continue
            
        # Skip statistics files
        if robot_name.endswith('.json'):
            continue
            
        print(f"Processing robot: {robot_name}")
        
        # Find all scene directories under robot
        for scene_name in os.listdir(robot_path):
            scene_path = os.path.join(robot_path, scene_name)
            if not os.path.isdir(scene_path):
                continue
                
            # Skip non-scene directories (like statistics files)
            if not scene_name.startswith('scene_'):
                continue
                
            print(f"  Processing scene: {scene_name}")
            
            # Find all object directories under scene
            for object_id in os.listdir(scene_path):
                object_path = os.path.join(scene_path, object_id)
                if not os.path.isdir(object_path):
                    continue
                    
                # Skip non-object directories (like grasp folders, json files)
                if object_id.startswith('grasp_') or object_id.endswith('.json'):
                    continue
                    
                # Check for camera_data directory
                camera_data_path = os.path.join(object_path, "camera_data")
                if not os.path.exists(camera_data_path):
                    # Record 0 episodes if camera_data doesn't exist
                    statistics["statics"][robot_name][scene_name][object_id] = 0
                    continue
                    
                # Count hdf5 files
                hdf5_count = 0
                for file in os.listdir(camera_data_path):
                    if file.endswith('.hdf5'):
                        hdf5_count += 1
                
                statistics["statics"][robot_name][scene_name][object_id] = hdf5_count
                print(f"    Object {object_id}: {hdf5_count} episodes")
    
    # Convert defaultdict to regular dict with sorted keys
    def convert_to_dict_sorted(d):
        if isinstance(d, defaultdict):
            # Sort keys numerically if they are numeric strings or scene names
            if all(isinstance(k, str) and (k.isdigit() or k.startswith('scene_')) for k in d.keys()):
                if all(k.startswith('scene_') for k in d.keys()):
                    # Sort scene names by numeric part
                    sorted_items = sorted(d.items(), key=lambda x: int(x[0].split('_')[1]))
                else:
                    sorted_items = sorted(d.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
                return {k: convert_to_dict_sorted(v) for k, v in sorted_items}
            else:
                return {k: convert_to_dict_sorted(v) for k, v in sorted(d.items())}
        return d
    
    statistics["statics"] = convert_to_dict_sorted(statistics["statics"])
    
    # Save statistics file
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"Statistics saved to {config_path}")
    
    # Print summary
    total_episodes = 0
    for robot_data in statistics["statics"].values():
        for scene_data in robot_data.values():
            for episode_count in scene_data.values():
                total_episodes += episode_count
    
    print(f"Total episodes found: {total_episodes}")
    return statistics


def pick_data(output_dir, pick_config_path, target_base_dir, config_dir, random_pick=True, copy_files=True):
    """
    Pick data according to pick configuration and copy/link to target directory
    """
    # Read pick configuration
    with open(pick_config_path, 'r') as f:
        pick_config = json.load(f)
    
    task_name = pick_config.get("task_name", "automoma_camera_collection")
    statics = pick_config.get("statics", {})
    
    # Initialize data structures for train and test json
    train_data = {
        "task_name": task_name,
        "statics": defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    }
    
    test_data = {
        "task_name": task_name,
        "statics": defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    }
    
    # Process each robot_name
    for robot_name, robot_data in statics.items():
        print(f"Processing robot: {robot_name}")
        
        # Create target directory
        target_dir = os.path.join(target_base_dir, f"automoma_manip_{robot_name}", task_name)
        os.makedirs(target_dir, exist_ok=True)
        target_data_dir = os.path.join(target_dir, "data")
        os.makedirs(target_data_dir, exist_ok=True)

        # Create config output directory
        config_output_dir = os.path.join(config_dir, f"automoma_manip_{robot_name}", task_name)
        os.makedirs(config_output_dir, exist_ok=True)
        
        # Counter for tracking files processed
        total_files_processed = 0
        
        # Initialize tracking dictionary for this robot
        tracking_log = {
            "robot_name": robot_name,
            "task_name": task_name,
            "timestamp": datetime.now().isoformat(),
            "random_pick": random_pick,
            "copy_files": copy_files,
            "scenes_processed": {}
        }
        
        # Process each scene_name
        for scene_name, scene_data in robot_data.items():
            print(f"  Processing scene: {scene_name}")
            
            # Initialize scene tracking
            tracking_log["scenes_processed"][scene_name] = {
                "objects": {}
            }
            
            # Process each object_id
            for object_id, pick_count in scene_data.items():
                print(f"    Processing object: {object_id} (pick {pick_count} episodes)")
                
                # Source data path
                source_path = os.path.join(
                    output_dir, robot_name, scene_name, object_id, "camera_data"
                )
                
                if not os.path.exists(source_path):
                    print(f"    Source path not found: {source_path}")
                    # Create empty entries even if source doesn't exist
                    train_data["statics"][robot_name][scene_name][object_id] = []
                    test_data["statics"][robot_name][scene_name][object_id] = []
                    tracking_log["scenes_processed"][scene_name]["objects"][object_id] = {
                        "status": "source_path_not_found",
                        "total_available": 0,
                        "picked": [],
                        "test": []
                    }
                    continue
                
                # Get hdf5 file list and sort them
                hdf5_files = [f for f in os.listdir(source_path) if f.endswith('.hdf5')]
                hdf5_files.sort()  # Sort alphabetically to ensure consistent order
                
                # Handle case where pick_count is 0
                if pick_count <= 0:
                    files_to_pick = []
                    remaining_files = hdf5_files
                else:
                    # Select files based on random_pick flag
                    if random_pick and len(hdf5_files) >= pick_count:
                        # Randomly select files
                        files_to_pick = random.sample(hdf5_files, pick_count)
                        # Remaining files are those not selected
                        remaining_files = [f for f in hdf5_files if f not in files_to_pick]
                    else:
                        # Take only first pick_count files for training (sequential)
                        files_to_pick = hdf5_files[:pick_count]
                        remaining_files = hdf5_files[pick_count:]
                
                print(f"    Picking {len(files_to_pick)} files from {len(hdf5_files)} available")
                
                # Copy/link and rename files (only training files)
                picked_indices = []
                for file_name in files_to_pick:
                    source_file = os.path.join(source_path, file_name)
                    
                    # Extract the original index from the file name
                    if file_name.startswith("episode") and file_name.endswith(".hdf5"):
                        try:
                            original_index = int(file_name[7:-5])  # Extract number between "episode" and ".hdf5"
                            picked_indices.append(original_index)
                        except ValueError:
                            # If we can't parse the index, skip this file
                            print(f"    Warning: Could not parse index from {file_name}")
                            continue
                    
                    # Generate new sequential file name based on existing files in target directory
                    existing_files = [f for f in os.listdir(target_data_dir) if f.endswith('.hdf5')]
                    new_file_name = f"episode{len(existing_files):06d}.hdf5"
                    target_file = os.path.join(target_data_dir, new_file_name)
                    
                    # Copy or link files based on copy_files flag
                    if copy_files:
                        shutil.copy2(source_file, target_file)
                        print(f"    Copied {file_name} -> {new_file_name}")
                    else:
                        # Create symbolic link with absolute paths
                        if os.path.exists(target_file) or os.path.islink(target_file):
                            os.remove(target_file)
                        os.symlink(os.path.abspath(source_file), target_file)
                        print(f"    Linked {file_name} -> {new_file_name}")
                    
                    total_files_processed += 1
                
                # Store picked file indices in train_data
                train_data["statics"][robot_name][scene_name][object_id] = picked_indices
                
                # For test data, record the indices of the remaining original files
                test_indices = []
                for file_name in remaining_files:
                    if file_name.startswith("episode") and file_name.endswith(".hdf5"):
                        try:
                            index = int(file_name[7:-5])  # Extract number between "episode" and ".hdf5"
                            test_indices.append(index)
                        except ValueError:
                            # If we can't parse the index, skip this file
                            pass
                
                # Store test file indices in test_data
                test_data["statics"][robot_name][scene_name][object_id] = test_indices
                
                # Store tracking information for this object
                tracking_log["scenes_processed"][scene_name]["objects"][object_id] = {
                    "status": "processed",
                    "total_available": len(hdf5_files),
                    "picked_count": len(picked_indices),
                    "test_count": len(test_indices),
                    "picked_indices": picked_indices,
                    "test_indices": test_indices,
                    "method": "random" if random_pick else "sequential"
                }
        
        print(f"Total files processed for {robot_name}: {total_files_processed}")
        
        # Save detailed tracking log
        tracking_log_path = os.path.join(config_output_dir, "pick_tracking_log.json")
        with open(tracking_log_path, 'w') as f:
            json.dump(tracking_log, f, indent=2)
        print(f"Saved detailed tracking log to {tracking_log_path}")
        
        # Copy pick configuration file to target directory
        target_config_path = os.path.join(target_dir, "data_pick_statistic.json")
        shutil.copy2(pick_config_path, target_config_path)
        print(f"Copied pick config to {target_config_path}")
        
        # Convert defaultdict to regular dict for saving
        def convert_to_dict_sorted(d):
            if isinstance(d, defaultdict):
                # Sort keys appropriately
                if all(isinstance(k, str) and k.startswith('scene_') for k in d.keys()):
                    # Sort scene names by numeric part
                    sorted_items = sorted(d.items(), key=lambda x: int(x[0].split('_')[1]))
                    return {k: convert_to_dict_sorted(v) for k, v in sorted_items}
                elif all(isinstance(k, str) and k.isdigit() for k in d.keys()):
                    sorted_items = sorted(d.items(), key=lambda x: int(x[0]))
                    return {k: convert_to_dict_sorted(v) for k, v in sorted_items}
                else:
                    return {k: convert_to_dict_sorted(v) for k, v in sorted(d.items())}
            elif isinstance(d, list):
                return d
            return d
        
        # Save train.json and test.json to config directory
        train_data["statics"] = convert_to_dict_sorted(train_data["statics"])
        train_json_path = os.path.join(config_output_dir, "train.json")
        with open(train_json_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        print(f"Train data config saved to {train_json_path}")
        
        # Save test.json
        test_data["statics"] = convert_to_dict_sorted(test_data["statics"])
        test_json_path = os.path.join(config_output_dir, "test.json")
        with open(test_json_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"Test data config saved to {test_json_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Pick data for automoma baseline following pipeline_collect structure")
    parser.add_argument("--mode", choices=["collect", "pick"], required=True, 
                       help="Mode: collect statistics or pick data")
    parser.add_argument("--output_dir", default="output/collect/traj", 
                       help="Output directory path (where robot directories are located)")
    parser.add_argument("--config_dir", default="scripts/config",
                       help="Configuration directory path")
    parser.add_argument("--sequential", action="store_true",
                       help="Use sequential picking instead of random picking (default is random)")
    parser.add_argument("--link", action="store_true",
                       help="Create symbolic links instead of copying files to save space")
    
    args = parser.parse_args()
    
    if args.mode == "collect":
        # Statistics collection mode
        config_path = os.path.join(args.config_dir, "automoma_data_statistic.json")
        statistics = collect_data_statistics(args.output_dir, config_path)
        
        # Copy data_statistic.json to data_pick_statistic.json for easy modification
        pick_config_path = os.path.join(args.config_dir, "automoma_data_pick_statistic.json")
        shutil.copy2(config_path, pick_config_path)
        print(f"Data statistic copied to {pick_config_path} for modification")
        print("\nNext steps:")
        print(f"1. Edit {pick_config_path} to specify how many episodes to pick for each robot/scene/object")
        print(f"2. Run: python {__file__} --mode pick --output_dir {args.output_dir}")
            
    elif args.mode == "pick":
        # Data picking mode
        pick_config_path = os.path.join(args.config_dir, "automoma_data_pick_statistic.json")
        target_base_dir = "baseline/RoboTwin/data"
        
        if not os.path.exists(pick_config_path):
            print(f"Pick config not found: {pick_config_path}")
            print("Please run with --mode collect first, then modify the pick config")
            return
        
        # Use random picking by default, unless --sequential is specified
        random_pick = not args.sequential
        # Use copying by default, unless --link is specified
        copy_files = not args.link
        
        print(f"Starting data picking...")
        print(f"Random pick: {random_pick}")
        print(f"Copy files: {copy_files}")
        print()
        
        pick_data(args.output_dir, pick_config_path, target_base_dir, args.config_dir, random_pick, copy_files)
        
        print("\nData picking completed!")
        print(f"Files have been {'copied' if copy_files else 'linked'} to baseline/RoboTwin/data/")


if __name__ == "__main__":
    main()