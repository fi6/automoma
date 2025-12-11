#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path

def compare_grasp_files(old_pos_path, new_pos_path, translation_scale=0.3562990018302636):
    """
    Compare grasp files between old and new paths.
    
    Args:
        old_pos_path (str): Path to old position files (.npz)
        new_pos_path (str): Path to new grasp files (.npy)
        translation_scale (float): Scale factor for translation comparison
    """
    
    old_path = Path(old_pos_path)
    new_path = Path(new_pos_path)
    
    # Get all .npz files from old path
    old_files = sorted(old_path.glob("*.npz"))
    
    if not old_files:
        print(f"No .npz files found in {old_pos_path}")
        return
    
    print(f"Found {len(old_files)} files to compare")
    print("-" * 80)
    
    total_comparisons = 0
    mismatches = 0
    
    for old_file in old_files:
        # Construct corresponding new file path
        file_stem = old_file.stem  # e.g., "0000"
        new_file = new_path / f"{file_stem}.npy"
        
        if not new_file.exists():
            print(f"WARNING: {new_file} does not exist, skipping...")
            continue
        
        try:
            # Load old data
            old_data = np.load(old_file, allow_pickle=True)
            grasp_data = old_data["data"].item()  # Extract the dictionary from the array
            
            # Extract translation and rotation from old data
            old_pos_wf = grasp_data['pos_wf']  # Translation [x, y, z]
            old_quat_xyzw = grasp_data['after_grasp_quat_wf']  # Rotation [x, y, z, w]
            
            # Convert old quaternion from [x, y, z, w] → [w, x, y, z] to match new format
            old_quat_wxyz = np.array([
                old_quat_xyzw[3],  # w
                old_quat_xyzw[0],  # x
                old_quat_xyzw[1],  # y
                old_quat_xyzw[2]   # z
            ])
            
            # Apply scale to translation
            old_pos_scaled = old_pos_wf # / translation_scale
            
            # Load new data
            new_data = np.load(new_file)
            
            # Extract translation and rotation from new data
            # Format: [tx, ty, tz, qw, qx, qy, qz]
            new_translation = new_data[:3] * translation_scale
            new_quaternion = new_data[3:]  # [w, x, y, z]
            
            # Compare translations
            trans_diff = np.abs(old_pos_scaled - new_translation)
            trans_max_diff = np.max(trans_diff)
            trans_mean_diff = np.mean(trans_diff)
            
            # Compare rotations (quaternions)
            # Account for double cover: q and -q are same rotation
            rot_diff_direct = np.abs(old_quat_wxyz - new_quaternion)
            rot_diff_flipped = np.abs(old_quat_wxyz + new_quaternion)
            rot_diff = np.minimum(rot_diff_direct, rot_diff_flipped)
            
            rot_max_diff = np.max(rot_diff)
            rot_mean_diff = np.mean(rot_diff)
            
            # Tolerance for floating point comparison
            tolerance = 1e-6
            trans_match = trans_max_diff < tolerance
            rot_match = rot_max_diff < tolerance
            overall_match = trans_match and rot_match
            
            if not overall_match:
                mismatches += 1
            
            total_comparisons += 1
            
            print(f"File: {file_stem}")
            print(f"  Translation (scaled): {old_pos_scaled}")
            print(f"  Translation (new):    {new_translation}")
            print(f"  Translation Diff (max/mean): {trans_max_diff:.2e} / {trans_mean_diff:.2e} {'✓' if trans_match else '✗'}")
            
            print(f"  Rotation (old→wxyz):  {old_quat_wxyz}")
            print(f"  Rotation (new):       {new_quaternion}")
            print(f"  Rotation Diff (max/mean):    {rot_max_diff:.2e} / {rot_mean_diff:.2e} {'✓' if rot_match else '✗'}")
            
            if not overall_match:
                print(f"  RESULT: MISMATCH")
            else:
                print(f"  RESULT: MATCH")
            print("-" * 50)
            
        except Exception as e:
            print(f"ERROR processing {old_file.name}: {str(e)}")
            continue
    
    print(f"\nSUMMARY:")
    print(f"Total files compared: {total_comparisons}")
    print(f"Matches: {total_comparisons - mismatches}")
    print(f"Mismatches: {mismatches}")
    
    if mismatches == 0 and total_comparisons > 0:
        print("✅ All files match!")
    elif total_comparisons == 0:
        print("⚠️  No files were successfully compared")
    else:
        print("❌ Some files have mismatches")

if __name__ == "__main__":
    # Configuration
    old_pos_path = "assets/object/Microwave/7221/0/raw/pos"
    new_pos_path = "assets/object/Microwave/7221/grasp"
    translation_scale = 0.3562990018302636
    
    # Run comparison
    compare_grasp_files(old_pos_path, new_pos_path, translation_scale)