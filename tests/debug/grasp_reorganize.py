import numpy as np
import os
import shutil
import json
from pathlib import Path


def clean_existing_grasp_dirs():
    """
    Remove existing grasp directories in object folders to start fresh.
    """
    assets_path = Path("/assets")
    object_path = assets_path / "object"
    
    removed_count = 0
    for category_dir in object_path.iterdir():
        if category_dir.is_dir() and category_dir.name not in ["glb_conversion_stats.json", "stats.json"]:
            for object_dir in category_dir.iterdir():
                if object_dir.is_dir():
                    grasp_dir = object_dir / "grasp"
                    if grasp_dir.exists():
                        shutil.rmtree(grasp_dir)
                        removed_count += 1
                        print(f"Removed {category_dir.name}/{object_dir.name}/grasp")
    
    print(f"Cleaned {removed_count} existing grasp directories")


def function1_copy_grasp_to_object():
    """
    Copy grasp folder to corresponding object id folder, name as grasp.
    Only copies instance "0" from each object.
    """
    assets_path = Path("/assets")
    grasp_path = assets_path / "grasp"
    object_path = assets_path / "object"
    
    # Load object data to get category mapping
    object_data_path = grasp_path / "object_data.json"
    with open(object_data_path, 'r') as f:
        object_data = json.load(f)
    
    # Get all object folders in grasp directory
    grasp_folders = [d for d in grasp_path.iterdir() if d.is_dir() and d.name.isdigit()]
    
    for grasp_folder in grasp_folders:
        object_id = grasp_folder.name
        
        # Get category from object_data.json
        if object_id in object_data:
            category = object_data[object_id]["category"]
            
            # Target directory: assets/object/{category}/{object_id}/grasp
            target_dir = object_path / category / object_id / "grasp"
            
            # Create target directory if it doesn't exist
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Only copy instance "0" from the grasp folder
            instance_0_path = grasp_folder / "0"
            if instance_0_path.exists():
                shutil.copytree(instance_0_path, target_dir, dirs_exist_ok=True)
                print(f"Copied {object_id}/0 to {target_dir}")
            else:
                print(f"Warning: {instance_0_path} does not exist")
        else:
            print(f"Warning: Object ID {object_id} not found in object_data.json")


def function2_apply_scaling_and_convert():
    """
    Apply scaling to grasp files and convert from npz to npy format with 7D pose.
    From: object/{category}/{id}/grasp/raw/pos/{file}.npz
    To: object/{category}/{id}/grasp/{file}.npy
    """
    assets_path = Path("/assets")
    object_path = assets_path / "object"
    grasp_path = assets_path / "grasp"
    
    # Load object data to get scaling values
    object_data_path = grasp_path / "object_data.json"
    with open(object_data_path, 'r') as f:
        object_data = json.load(f)
    
    # Iterate through all categories
    for category_dir in object_path.iterdir():
        if category_dir.is_dir() and category_dir.name not in ["glb_conversion_stats.json", "stats.json"]:
            # Iterate through all object IDs in this category
            for object_dir in category_dir.iterdir():
                if object_dir.is_dir():
                    object_id = object_dir.name
                    grasp_dir = object_dir / "grasp"
                    
                    if grasp_dir.exists() and object_id in object_data:
                        print(f"Processing {category_dir.name}/{object_id}")
                        
                        # Get scaling value from object_data.json (instance "0")
                        scaling = object_data[object_id]["instances"]["0"]["scaling"]
                        print(f"  Using scaling: {scaling}")
                        
                        # Process files from raw/pos/ directory
                        raw_pos_dir = grasp_dir / "raw" / "pos"
                        if raw_pos_dir.exists():
                            npz_files = sorted(raw_pos_dir.glob("*.npz"))
                            
                            for i, npz_file in enumerate(npz_files):
                                try:
                                    # Load grasp data
                                    grasp_data = np.load(npz_file, allow_pickle=True)
                                    grasp_info = grasp_data["data"].item()
                                    
                                    # Extract and scale position
                                    position = grasp_info["pos_wf"] / scaling
                                    
                                    # Extract quaternion in w,x,y,z format
                                    after_grasp_quat = grasp_info["after_grasp_quat_wf"]
                                    quaternion = np.array([
                                        after_grasp_quat[3],  # w
                                        after_grasp_quat[0],  # x
                                        after_grasp_quat[1],  # y
                                        after_grasp_quat[2]   # z
                                    ])
                                    
                                    # Combine into 7D pose [x, y, z, qw, qx, qy, qz]
                                    pose_7d = np.concatenate([position, quaternion])
                                    
                                    # Save as .npy file directly under grasp/
                                    npy_filename = f"{i:04d}.npy"
                                    npy_path = grasp_dir / npy_filename
                                    np.save(npy_path, pose_7d)
                                    
                                except Exception as e:
                                    print(f"  Error processing {npz_file}: {e}")
                            
                            print(f"  Converted {len(npz_files)} files to 7D pose format")
                        else:
                            print(f"  Warning: {raw_pos_dir} not found")


def function3_cleanup_files():
    """
    Clean up unnecessary files, keeping only the .npy grasp files.
    Remove init_state.npz, raw/ directory, and any other files except .npy files.
    """
    assets_path = Path("/assets")
    object_path = assets_path / "object"
    
    # Iterate through all categories
    for category_dir in object_path.iterdir():
        if category_dir.is_dir() and category_dir.name not in ["glb_conversion_stats.json", "stats.json"]:
            # Iterate through all object IDs in this category
            for object_dir in category_dir.iterdir():
                if object_dir.is_dir():
                    grasp_dir = object_dir / "grasp"
                    
                    if grasp_dir.exists():
                        print(f"Cleaning up {category_dir.name}/{object_dir.name}")
                        
                        files_removed = 0
                        
                        # Remove init_state.npz if it exists
                        init_state_file = grasp_dir / "init_state.npz"
                        if init_state_file.exists():
                            init_state_file.unlink()
                            files_removed += 1
                            print(f"  Removed init_state.npz")
                        
                        # Remove raw/ directory if it exists
                        raw_dir = grasp_dir / "raw"
                        if raw_dir.exists():
                            shutil.rmtree(raw_dir)
                            files_removed += 1
                            print(f"  Removed raw/ directory")
                        
                        # Remove any other files that are not .npy
                        for item in grasp_dir.iterdir():
                            if item.is_file() and not item.name.endswith('.npy'):
                                item.unlink()
                                files_removed += 1
                                print(f"  Removed {item.name}")
                        
                        # Count remaining .npy files
                        npy_files = list(grasp_dir.glob("*.npy"))
                        print(f"  Cleanup complete: {files_removed} items removed, {len(npy_files)} .npy files remaining")


if __name__ == "__main__":
    print("Grasp Reorganization Script")
    print("0. Clean existing grasp directories (recommended before step 1)")
    print("1. Copy grasp folders to object directories (only instance '0')")
    print("2. Apply scaling and convert to 7D pose (.npy files)")
    print("3. Clean up unnecessary files (keep only .npy files)")
    print()
    
    choice = input("Enter function number to run (0, 1, 2, 3, or 'all'): ")
    
    if choice == "0" or choice == "all":
        print("Running Function 0: Cleaning existing grasp directories...")
        clean_existing_grasp_dirs()
        print("Function 0 completed.\n")
    
    if choice == "1" or choice == "all":        
        print("Running Function 1: Copying grasp folders...")
        function1_copy_grasp_to_object()
        print("Function 1 completed.\n")
    
    if choice == "2" or choice == "all":
        print("Running Function 2: Applying scaling and converting...")
        function2_apply_scaling_and_convert()
        print("Function 2 completed.\n")
    
    if choice == "3" or choice == "all":
        print("Running Function 3: Cleaning up files...")
        function3_cleanup_files()
        print("Function 3 completed.\n")
    
    if choice not in ["0", "1", "2", "3", "all"]:
        print("Invalid choice. Please run the script again with 0, 1, 2, 3, or 'all'")