#!/usr/bin/env python3
"""
Quick Reference: Index Tracking in Automoma Pipeline

This script shows how to access and verify index tracking across the pipeline.
"""

import json
import torch
from pathlib import Path


def track_collection_indices(scene_asset_dir: str):
    """
    Load and display indices recorded during collection phase.
    
    Args:
        scene_asset_dir: Path to scene+asset directory (e.g., output/summit_franka/scene_5_seed_5/7221)
    """
    print("=" * 70)
    print("COLLECTION PHASE - Index Tracking")
    print("=" * 70)
    
    # Method 1: Load from selected_indices.json (simplest)
    indices_file = Path(scene_asset_dir) / "selected_indices.json"
    if indices_file.exists():
        with open(indices_file) as f:
            indices = json.load(f)
        print(f"\n✓ Selected Indices File Found: {indices_file}")
        print(f"  - Total selected: {indices['total_count']}")
        print(f"  - Random seed: {indices['random_seed']}")
        print(f"  - Timestamp: {indices['timestamp']}")
        print(f"  - First 10 indices: {indices['selected_indices'][:10]}")
        return indices['selected_indices']
    
    # Method 2: Load from PyTorch file (machine-readable)
    traj_file = Path(scene_asset_dir) / "selected_filtered_traj_data.pt"
    if traj_file.exists():
        print(f"\n✓ Trajectory Data File Found: {traj_file}")
        data = torch.load(traj_file, weights_only=False)
        print(f"  - Total trajectories: {data['num_trajectories']}")
        print(f"  - Total available: {data['total_available']}")
        print(f"  - Random seed: {data['random_seed']}")
        print(f"  - First 10 indices: {data['selected_indices'][:10]}")
        return data['selected_indices']
    
    # Method 3: Load from collection metadata
    metadata_file = Path(scene_asset_dir) / "collection_metadata.json"
    if metadata_file.exists():
        print(f"\n✓ Collection Metadata File Found: {metadata_file}")
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(f"  - Total selected: {metadata['selected_indices_count']}")
        print(f"  - Random seed: {metadata['random_seed']}")
        print(f"  - Grasp sources: {metadata['grasp_source_distribution']}")
        print(f"  - First 10 indices: {metadata['selected_indices'][:10]}")
        return metadata['selected_indices']
    
    print("\n✗ No index tracking files found!")
    return None


def track_picking_indices(config_dir: str, robot_name: str, scene_name: str, object_id: str):
    """
    Load and display indices recorded during picking phase.
    
    Args:
        config_dir: Configuration directory (e.g., scripts/config)
        robot_name: Robot name (e.g., summit_franka)
        scene_name: Scene name (e.g., scene_5_seed_5)
        object_id: Object ID (e.g., 7221)
    """
    print("\n" + "=" * 70)
    print("PICKING PHASE - Index Tracking")
    print("=" * 70)
    
    # Load picking tracking log
    log_file = Path(config_dir) / f"automoma_manip_{robot_name}" / "automoma_camera_collection" / "pick_tracking_log.json"
    
    if log_file.exists():
        with open(log_file) as f:
            log = json.load(f)
        
        print(f"\n✓ Picking Tracking Log Found: {log_file}")
        print(f"  - Timestamp: {log['timestamp']}")
        print(f"  - Random pick: {log['random_pick']}")
        print(f"  - Copy files: {log['copy_files']}")
        
        # Access specific scene and object
        if scene_name in log["scenes_processed"] and object_id in log["scenes_processed"][scene_name]["objects"]:
            tracking = log["scenes_processed"][scene_name]["objects"][object_id]
            print(f"\n  Scene: {scene_name}, Object: {object_id}")
            print(f"    - Total available: {tracking['total_available']}")
            print(f"    - Picked for training: {tracking['picked_count']}")
            print(f"    - Reserved for testing: {tracking['test_count']}")
            print(f"    - Selection method: {tracking['method']}")
            print(f"    - First 5 train indices: {tracking['picked_indices'][:5]}")
            print(f"    - First 5 test indices: {tracking['test_indices'][:5]}")
            return tracking
    else:
        print(f"\n✗ Picking log not found: {log_file}")
    
    return None


def verify_data_consistency(collection_dir: str, picking_dir: str, robot_name: str, scene_name: str, object_id: str):
    """
    Verify that data selection is consistent across both phases.
    """
    print("\n" + "=" * 70)
    print("VERIFICATION - Data Consistency Check")
    print("=" * 70)
    
    try:
        # Load collection indices
        collection_file = Path(collection_dir) / "selected_indices.json"
        with open(collection_file) as f:
            collection = json.load(f)
        collection_set = set(collection['selected_indices'])
        print(f"\n✓ Loaded {len(collection_set)} indices from collection phase")
        
        # Load picking log
        log_file = Path(picking_dir) / f"automoma_manip_{robot_name}" / "automoma_camera_collection" / "pick_tracking_log.json"
        with open(log_file) as f:
            log = json.load(f)
        
        scene_obj = log["scenes_processed"][scene_name]["objects"][object_id]
        train_set = set(scene_obj['picked_indices'])
        test_set = set(scene_obj['test_indices'])
        print(f"✓ Loaded {len(train_set)} train indices from picking phase")
        print(f"✓ Loaded {len(test_set)} test indices from picking phase")
        
        # Verify consistency
        print("\n[Verification Results]")
        
        # Check 1: Train and test are mutually exclusive
        overlap = train_set & test_set
        if overlap:
            print(f"✗ ERROR: {len(overlap)} indices appear in both train and test!")
            return False
        else:
            print(f"✓ Train and test sets are mutually exclusive")
        
        # Check 2: All train/test indices came from collection
        train_not_in_collection = train_set - collection_set
        test_not_in_collection = test_set - collection_set
        
        if train_not_in_collection:
            print(f"✗ ERROR: {len(train_not_in_collection)} train indices not in collection!")
            return False
        if test_not_in_collection:
            print(f"✗ ERROR: {len(test_not_in_collection)} test indices not in collection!")
            return False
        
        print(f"✓ All train/test indices are from collection phase")
        
        # Check 3: All collected indices are used
        unused = collection_set - (train_set | test_set)
        if unused:
            print(f"⚠ WARNING: {len(unused)} collected indices not used in train/test!")
        else:
            print(f"✓ All collected indices are used in train/test splits")
        
        # Summary
        print(f"\n[Summary]")
        print(f"  Collection: {len(collection_set)} indices")
        print(f"  Train split: {len(train_set)} indices ({100*len(train_set)/len(collection_set):.1f}%)")
        print(f"  Test split: {len(test_set)} indices ({100*len(test_set)/len(collection_set):.1f}%)")
        print(f"  Total used: {len(train_set | test_set)} indices")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during verification: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("Automoma Pipeline - Index Tracking Quick Reference\n")
    
    # Track collection indices
    scene_asset_dir = "output/summit_franka/scene_5_seed_5/7221"
    collection_indices = track_collection_indices(scene_asset_dir)
    
    # Track picking indices
    picking_info = track_picking_indices(
        "scripts/config",
        "summit_franka",
        "scene_5_seed_5",
        "7221"
    )
    
    # Verify consistency
    if collection_indices and picking_info:
        verify_data_consistency(
            scene_asset_dir,
            "scripts/config",
            "summit_franka",
            "scene_5_seed_5",
            "7221"
        )
    
    print("\n" + "=" * 70)
    print("For more details, see: docs/INDEX_TRACKING_GUIDE.md")
    print("=" * 70)
