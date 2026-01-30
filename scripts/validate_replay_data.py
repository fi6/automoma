"""
Utility script for testing and validating the enhanced replay pipeline functionality.
This script provides tools for inspecting recorded data and evaluating policy performance.
"""

import os
import h5py
import torch
import numpy as np
import argparse
from typing import List, Dict, Any
import json
from pathlib import Path


def inspect_recorded_data(data_dir: str) -> Dict[str, Any]:
    """
    Inspect recorded trajectory data and provide summary statistics.
    
    Args:
        data_dir: Directory containing recorded HDF5 files
        
    Returns:
        Summary statistics dictionary
    """
    camera_data_dir = os.path.join(data_dir, "camera_data")
    
    if not os.path.exists(camera_data_dir):
        print(f"Camera data directory not found: {camera_data_dir}")
        return {}
    
    # Find all HDF5 files
    hdf5_files = [f for f in os.listdir(camera_data_dir) if f.endswith('.hdf5')]
    
    if not hdf5_files:
        print(f"No HDF5 files found in {camera_data_dir}")
        return {}
    
    print(f"Found {len(hdf5_files)} recorded episodes")
    
    stats = {
        "num_episodes": len(hdf5_files),
        "episodes": [],
        "cameras": set(),
        "joint_names": set(),
        "total_timesteps": 0
    }
    
    for file in sorted(hdf5_files):
        filepath = os.path.join(camera_data_dir, file)
        
        try:
            with h5py.File(filepath, "r") as f:
                episode_info = {
                    "filename": file,
                    "env_info": {},
                    "obs_structure": {},
                    "timesteps": 0
                }
                
                # Extract environment info
                if "env_info" in f:
                    for key in f["env_info"]:
                        value = f["env_info"][key][()]
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        episode_info["env_info"][key] = value
                
                # Analyze observation structure
                if "obs" in f:
                    obs = f["obs"]
                    
                    # Check joint data
                    if "joint" in obs and isinstance(obs["joint"], h5py.Group):
                        joint_names = list(obs["joint"].keys())
                        stats["joint_names"].update(joint_names)
                        if joint_names:
                            first_joint = obs["joint"][joint_names[0]]
                            episode_info["timesteps"] = len(first_joint)
                            episode_info["obs_structure"]["joint"] = {
                                "names": joint_names,
                                "shape": first_joint.shape
                            }
                    
                    # Check camera data
                    if "rgb" in obs and isinstance(obs["rgb"], h5py.Group):
                        camera_names = list(obs["rgb"].keys())
                        stats["cameras"].update(camera_names)
                        episode_info["obs_structure"]["cameras"] = camera_names
                        
                        for cam_name in camera_names:
                            rgb_data = obs["rgb"][cam_name]
                            episode_info["obs_structure"][f"rgb_{cam_name}"] = {
                                "shape": rgb_data.shape,
                                "dtype": str(rgb_data.dtype)
                            }
                    
                    # Check other observation types
                    for obs_type in ["eef", "point_cloud", "depth"]:
                        if obs_type in obs:
                            if isinstance(obs[obs_type], h5py.Group):
                                # Group structure
                                subkeys = list(obs[obs_type].keys())
                                episode_info["obs_structure"][obs_type] = {
                                    "type": "group",
                                    "keys": subkeys
                                }
                            else:
                                # Dataset structure
                                episode_info["obs_structure"][obs_type] = {
                                    "type": "dataset", 
                                    "shape": obs[obs_type].shape,
                                    "dtype": str(obs[obs_type].dtype)
                                }
                
                stats["episodes"].append(episode_info)
                stats["total_timesteps"] += episode_info["timesteps"]
                
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    # Convert sets to lists for JSON serialization
    stats["cameras"] = list(stats["cameras"])
    stats["joint_names"] = list(stats["joint_names"])
    
    return stats


def print_data_summary(stats: Dict[str, Any]) -> None:
    """Print a formatted summary of the data statistics."""
    if not stats:
        print("No data to summarize")
        return
    
    print("=" * 60)
    print("RECORDED DATA SUMMARY")
    print("=" * 60)
    
    print(f"Episodes: {stats['num_episodes']}")
    print(f"Total timesteps: {stats['total_timesteps']}")
    print(f"Average timesteps per episode: {stats['total_timesteps'] / stats['num_episodes']:.1f}")
    
    print(f"\nCameras: {', '.join(stats['cameras'])}")
    print(f"Joints: {len(stats['joint_names'])} joints")
    if len(stats['joint_names']) <= 10:
        print(f"  Joint names: {', '.join(stats['joint_names'])}")
    
    print(f"\nFirst Episode Details:")
    if stats['episodes']:
        episode = stats['episodes'][0]
        print(f"  File: {episode['filename']}")
        print(f"  Timesteps: {episode['timesteps']}")
        
        if 'env_info' in episode:
            print("  Environment Info:")
            for key, value in episode['env_info'].items():
                print(f"    {key}: {value}")
        
        if 'obs_structure' in episode:
            print("  Observation Structure:")
            for key, value in episode['obs_structure'].items():
                if isinstance(value, dict) and 'shape' in value:
                    print(f"    {key}: {value['shape']} ({value.get('dtype', 'unknown')})")
                else:
                    print(f"    {key}: {value}")


def validate_data_format(data_dir: str) -> bool:
    """
    Validate that recorded data matches expected format from collect_data.py.
    
    Args:
        data_dir: Directory containing recorded data
        
    Returns:
        True if data format is valid
    """
    print("=" * 60)
    print("DATA FORMAT VALIDATION")
    print("=" * 60)
    
    camera_data_dir = os.path.join(data_dir, "camera_data")
    
    if not os.path.exists(camera_data_dir):
        print("❌ Camera data directory missing")
        return False
    
    # Check for HDF5 files
    hdf5_files = [f for f in os.listdir(camera_data_dir) if f.endswith('.hdf5')]
    if not hdf5_files:
        print("❌ No HDF5 files found")
        return False
    
    print(f"✓ Found {len(hdf5_files)} HDF5 files")
    
    # Validate structure of first file
    first_file = os.path.join(camera_data_dir, hdf5_files[0])
    
    try:
        with h5py.File(first_file, "r") as f:
            # Check required top-level groups
            required_groups = ["env_info", "obs"]
            for group in required_groups:
                if group not in f:
                    print(f"❌ Missing required group: {group}")
                    return False
                print(f"✓ Found group: {group}")
            
            # Check env_info structure
            env_info = f["env_info"]
            expected_env_keys = ["scene_id", "robot_name", "object_id"]
            for key in expected_env_keys:
                if key not in env_info:
                    print(f"⚠️  Missing env_info key: {key}")
                else:
                    print(f"✓ Found env_info key: {key}")
            
            # Check obs structure
            obs = f["obs"]
            expected_obs_keys = ["joint", "eef", "rgb", "depth"]
            for key in expected_obs_keys:
                if key not in obs:
                    print(f"⚠️  Missing obs key: {key}")
                else:
                    print(f"✓ Found obs key: {key}")
            
            # Validate camera data
            if "rgb" in obs and isinstance(obs["rgb"], h5py.Group):
                camera_names = list(obs["rgb"].keys())
                expected_cameras = ["ego_topdown", "ego_wrist", "fix_local"]
                
                for cam in expected_cameras:
                    if cam in camera_names:
                        print(f"✓ Found camera: {cam}")
                        # Check data shape
                        rgb_data = obs["rgb"][cam]
                        if len(rgb_data.shape) == 4:  # (T, H, W, C)
                            print(f"  Shape: {rgb_data.shape} (timesteps, height, width, channels)")
                        else:
                            print(f"  ⚠️  Unexpected shape: {rgb_data.shape}")
                    else:
                        print(f"⚠️  Missing camera: {cam}")
    
    except Exception as e:
        print(f"❌ Error validating file: {e}")
        return False
    
    print("✓ Data format validation completed")
    return True


def compare_with_collect_data(recorded_dir: str, reference_dir: str) -> None:
    """
    Compare recorded data structure with reference data from collect_data.py.
    
    Args:
        recorded_dir: Directory with newly recorded data
        reference_dir: Directory with reference data from collect_data.py
    """
    print("=" * 60)
    print("COMPARISON WITH COLLECT_DATA.PY")
    print("=" * 60)
    
    # Inspect both directories
    recorded_stats = inspect_recorded_data(recorded_dir)
    reference_stats = inspect_recorded_data(reference_dir)
    
    if not recorded_stats or not reference_stats:
        print("❌ Could not load data for comparison")
        return
    
    # Compare structure
    print("Comparing data structures...")
    
    # Compare cameras
    recorded_cameras = set(recorded_stats.get("cameras", []))
    reference_cameras = set(reference_stats.get("cameras", []))
    
    if recorded_cameras == reference_cameras:
        print(f"✓ Cameras match: {recorded_cameras}")
    else:
        print(f"⚠️  Camera mismatch:")
        print(f"  Recorded: {recorded_cameras}")
        print(f"  Reference: {reference_cameras}")
    
    # Compare joint names
    recorded_joints = set(recorded_stats.get("joint_names", []))
    reference_joints = set(reference_stats.get("joint_names", []))
    
    if recorded_joints == reference_joints:
        print(f"✓ Joint names match ({len(recorded_joints)} joints)")
    else:
        print(f"⚠️  Joint name mismatch:")
        print(f"  Recorded: {len(recorded_joints)} joints")
        print(f"  Reference: {len(reference_joints)} joints")


def main():
    parser = argparse.ArgumentParser(description="Replay Pipeline Data Validation Tool")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing recorded data to inspect")
    parser.add_argument("--reference_dir", type=str, 
                       help="Reference directory for comparison (optional)")
    parser.add_argument("--save_stats", type=str,
                       help="Save statistics to JSON file (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return
    
    # Inspect recorded data
    stats = inspect_recorded_data(args.data_dir)
    
    if stats:
        print_data_summary(stats)
        
        # Save statistics if requested
        if args.save_stats:
            with open(args.save_stats, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\nStatistics saved to: {args.save_stats}")
        
        # Validate data format
        validate_data_format(args.data_dir)
        
        # Compare with reference if provided
        if args.reference_dir:
            if os.path.exists(args.reference_dir):
                compare_with_collect_data(args.data_dir, args.reference_dir)
            else:
                print(f"Warning: Reference directory does not exist: {args.reference_dir}")
    
    else:
        print("No valid data found to inspect")


if __name__ == "__main__":
    main()