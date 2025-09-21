import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict

def collect_data_statistics(output_dir, config_path):
    """
    Collect statistics of hdf5 files in output directory and save as json file
    """
    statistics = {
        "task_name": None,
        "statics": defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))))
    }
    
    # Traverse output directory structure
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
        return
    
    for scene_id in os.listdir(output_dir):
        scene_path = os.path.join(output_dir, scene_id)
        if not os.path.isdir(scene_path):
            continue
            
        for robot_name in os.listdir(scene_path):
            robot_path = os.path.join(scene_path, robot_name)
            if not os.path.isdir(robot_path):
                continue
                
            for object_id in os.listdir(robot_path):
                object_path = os.path.join(robot_path, object_id)
                if not os.path.isdir(object_path):
                    continue
                    
                for angle_id in os.listdir(object_path):
                    angle_path = os.path.join(object_path, angle_id)
                    if not os.path.isdir(angle_path):
                        continue
                        
                    collect_data_path = os.path.join(angle_path, "collect_data")
                    if not os.path.exists(collect_data_path):
                        continue
                        
                    # Process all pose_id directories (even if empty)
                    for pose_id in os.listdir(collect_data_path):
                        pose_path = os.path.join(collect_data_path, pose_id)
                        if not os.path.isdir(pose_path):
                            continue
                            
                        # Count hdf5 files (could be 0)
                        hdf5_count = 0
                        for file in os.listdir(pose_path):
                            if file.endswith('.hdf5'):
                                hdf5_count += 1
                        
                        # Always record the pose_id, even if count is 0
                        statistics["statics"][robot_name][object_id][scene_id][angle_id][pose_id] = hdf5_count
    
    # Convert defaultdict to regular dict with sorted keys
    def convert_to_dict_sorted(d):
        if isinstance(d, defaultdict):
            # Sort keys numerically if they are numeric strings
            if all(isinstance(k, str) and k.isdigit() for k in d.keys()):
                sorted_items = sorted(d.items(), key=lambda x: int(x[0]))
                return {k: convert_to_dict_sorted(v) for k, v in sorted_items}
            else:
                return {k: convert_to_dict_sorted(v) for k, v in d.items()}
        return d
    
    statistics["statics"] = convert_to_dict_sorted(statistics["statics"])
    
    # Save statistics file
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"Statistics saved to {config_path}")
    return statistics

def pick_data(output_dir, pick_config_path, target_base_dir, config_dir, random_pick=True, copy_files=True):
    """
    Pick data according to pick configuration and copy to target directory
    """
    # Read pick configuration
    with open(pick_config_path, 'r') as f:
        pick_config = json.load(f)
    
    task_name = pick_config.get("task_name")
    if not task_name:
        print("Task name not specified in pick config")
        return
    
    statics = pick_config.get("statics", {})
    
    # Load the full statistics for test data generation
    full_stats_path = os.path.join(config_dir, "data_statistic.json")
    if os.path.exists(full_stats_path):
        with open(full_stats_path, 'r') as f:
            full_stats = json.load(f)
    else:
        full_stats = None
    
    # Initialize data structures for train and test json
    train_data = {
        "task_name": task_name,
        "statics": defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
    }
    
    test_data = {
        "task_name": task_name,
        "statics": defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
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
        
        # Counter for renaming
        global_file_counter = 0
        
        # Process each object_id
        for object_id, object_data in robot_data.items():
            for scene_id, scene_data in object_data.items():
                for angle_id, angle_data in scene_data.items():
                    # Sort pose_id numerically to ensure consistent order (0000, 0001, 0002, ...)
                    sorted_pose_items = sorted(angle_data.items(), key=lambda x: int(x[0]))
                    
                    for pose_id, pick_count in sorted_pose_items:
                        # Source data path
                        source_path = os.path.join(
                            output_dir, scene_id, robot_name, object_id, 
                            angle_id, "collect_data", pose_id
                        )
                        
                        if not os.path.exists(source_path):
                            print(f"Source path not found: {source_path}")
                            # Even if source path doesn't exist, we still need to create empty entries
                            train_data["statics"][robot_name][object_id][scene_id][angle_id][pose_id] = []
                            test_data["statics"][robot_name][object_id][scene_id][angle_id][pose_id] = []
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
                                # Take only first pick_count files for training (fallback to sequential)
                                files_to_pick = hdf5_files[:pick_count]
                                remaining_files = hdf5_files[pick_count:]
                        
                        print(f"Picking {len(files_to_pick)} files from {source_path}")
                        
                        # Copy and rename files (only training files)
                        picked_indices = []
                        for file_name in files_to_pick:
                            source_file = os.path.join(source_path, file_name)
                            # Extract the original index from the file name
                            if file_name.startswith("episode") and file_name.endswith(".hdf5"):
                                try:
                                    original_index = int(file_name[7:-5])  # Extract the number between "episode" and ".hdf5"
                                    picked_indices.append(original_index)
                                except ValueError:
                                    # If we can't parse the index, skip this file
                                    continue
                            
                            # Generate new sequential file name based on existing files in target directory
                            new_file_name = f"episode{len(os.listdir(target_data_dir)):06d}.hdf5"
                            target_file = os.path.join(target_data_dir, new_file_name)
                            
                            # Copy or link files based on copy_files flag
                            if copy_files:
                                shutil.copy2(source_file, target_file)
                                print(f"Copied {file_name} -> {new_file_name}")
                            else:
                                # Create symbolic link with absolute paths
                                if os.path.exists(target_file) or os.path.islink(target_file):
                                    os.remove(target_file)
                                os.symlink(os.path.abspath(source_file), target_file)
                                print(f"Linked {file_name} -> {new_file_name}")
                        
                        # Store picked file indices in train_data
                        train_data["statics"][robot_name][object_id][scene_id][angle_id][pose_id] = picked_indices
                        
                        # For test data, we record the indices of the remaining original files
                        # Extract indices from the file names (assuming format like "episode000123.hdf5")
                        test_indices = []
                        for file_name in remaining_files:
                            # Extract the numeric part from file name like "episode000123.hdf5"
                            if file_name.startswith("episode") and file_name.endswith(".hdf5"):
                                try:
                                    index = int(file_name[7:-5])  # Extract the number between "episode" and ".hdf5"
                                    test_indices.append(index)
                                except ValueError:
                                    # If we can't parse the index, skip this file
                                    pass
                        
                        # Store test file indices in test_data
                        test_data["statics"][robot_name][object_id][scene_id][angle_id][pose_id] = test_indices
        
        # Copy pick configuration file to target directory
        target_config_path = os.path.join(target_dir, "data_pick_statistic.json")
        shutil.copy2(pick_config_path, target_config_path)
        print(f"Copied pick config to {target_config_path}")
        
        # Convert defaultdict to regular dict for saving
        def convert_to_dict_sorted(d):
            if isinstance(d, defaultdict):
                # Sort keys numerically if they are numeric strings
                if all(isinstance(k, str) and k.isdigit() for k in d.keys()):
                    sorted_items = sorted(d.items(), key=lambda x: int(x[0]))
                    return {k: convert_to_dict_sorted(v) for k, v in sorted_items}
                else:
                    return {k: convert_to_dict_sorted(v) for k, v in d.items()}
            elif isinstance(d, list):
                return d
            return d
        
        # Save train.json and test.json to config directory
        train_data["statics"] = convert_to_dict_sorted(train_data["statics"])
        train_json_path = os.path.join(config_output_dir, "train.json")
        with open(train_json_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        print(f"Train data saved to {train_json_path}")
        
        # Save test.json
        test_data["statics"] = convert_to_dict_sorted(test_data["statics"])
        test_json_path = os.path.join(config_output_dir, "test.json")
        with open(test_json_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"Test data saved to {test_json_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Pick data for RoboTwin baseline")
    parser.add_argument("--mode", choices=["collect", "pick"], required=True, 
                       help="Mode: collect statistics or pick data")
    parser.add_argument("--output_dir", default="output", 
                       help="Output directory path")
<<<<<<< HEAD
    parser.add_argument("--config_dir", default="scripts/config",
=======
    parser.add_argument("--config_dir", default="scripts/baseline/RoboTwin/config/data",
>>>>>>> 1b49b31bfe85f70c576a7b2fb1a7f761467d3f33
                       help="Configuration directory path")
    parser.add_argument("--sequential", action="store_true",
                       help="Use sequential picking instead of random picking (default is random)")
    parser.add_argument("--link", action="store_true",
                       help="Create symbolic links instead of copying files to save space")
    
    args = parser.parse_args()
    
    if args.mode == "collect":
        # Statistics collection mode
        config_path = os.path.join(args.config_dir, "data_statistic.json")
        statistics = collect_data_statistics(args.output_dir, config_path)
        
        # Copy data_statistic.json to data_pick_statistic.json for easy modification
        pick_config_path = os.path.join(args.config_dir, "data_pick_statistic.json")
        shutil.copy2(config_path, pick_config_path)
        print(f"Data statistic copied to {pick_config_path} for modification")
            
    elif args.mode == "pick":
        # Data picking mode
        pick_config_path = os.path.join(args.config_dir, "data_pick_statistic.json")
        target_base_dir = "baseline/RoboTwin/data"
        
        if not os.path.exists(pick_config_path):
            print(f"Pick config not found: {pick_config_path}")
            print("Please run with --mode collect first, then modify the pick config")
            return
            
        # Use random picking by default, unless --sequential is specified
        random_pick = not args.sequential
        # Use copying by default, unless --link is specified
        copy_files = not args.link
        pick_data(args.output_dir, pick_config_path, target_base_dir, args.config_dir, random_pick, copy_files)

if __name__ == "__main__":
    main()