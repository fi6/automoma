#!/usr/bin/env python3
"""
Script to update object transformations in scene metadata.json files.
Updates matrix, position, and bbox_corners based on new translation and rotation.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
import os, sys

# ============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# ============================================================================

# Scene and object identification
SCENE_DIR = '''
/home/xinhai/Documents/automoma/assets/scene/infinigen/kitchen_1130/scene_55_seed_55/export/export_scene.blend/
'''
SCENE_DIR = SCENE_DIR.strip()
SCENE_PATH = SCENE_DIR.rsplit('/export/export_scene.blend/', 1)[0]
METADATA_FILE = "info/metadata.json"

# 46197 11622 101773 10944 103634 7221
OBJECT_ID = "10944"

# Backup configuration
CREATE_BACKUP = True  # Set to True to create a backup before modifying metadata

# New transformation parameters
NEW_TRANSLATION = (0.4333001077175141, 4.908326424331984, 0.9375704526901245)
NEW_ROTATION_DEGREES = (0, 0, 0)
NEW_ROTATION_RADIANS = tuple(np.radians(r) for r in NEW_ROTATION_DEGREES)

# ============================================================================
# TRANSFORMATION FUNCTIONS
# ============================================================================


def euler_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Convert Euler angles (in radians) to a 3x3 rotation matrix.
    Rotation order: Z-Y-X (commonly used in 3D graphics)
    
    Args:
        rx, ry, rz: Rotation angles in radians around X, Y, Z axes
        
    Returns:
        3x3 rotation matrix
    """
    # Rotation around X-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    # Rotation around Y-axis
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # Rotation around Z-axis
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: Rz * Ry * Rx
    return Rz @ Ry @ Rx


def create_transform_matrix(translation: Tuple[float, float, float],
                           rotation_radians: Tuple[float, float, float],
                           scale: Tuple[float, float, float]) -> List[List[float]]:
    """
    Create a 4x4 transformation matrix from translation, rotation, and scale.
    
    Args:
        translation: (x, y, z) translation
        rotation_radians: (rx, ry, rz) rotation in radians
        scale: (sx, sy, sz) scale factors
        
    Returns:
        4x4 transformation matrix as nested list
    """
    # Get rotation matrix
    R = euler_to_rotation_matrix(*rotation_radians)
    
    # Apply scale to rotation matrix
    scaled_R = R * np.array(scale)[:, np.newaxis]
    
    # Create 4x4 matrix
    matrix = np.eye(4)
    matrix[:3, :3] = scaled_R
    matrix[:3, 3] = translation
    
    return matrix.tolist()


def transform_bbox_corners(original_corners: List[List[float]],
                          old_matrix: np.ndarray,
                          new_matrix: np.ndarray) -> List[List[float]]:
    """
    Transform bounding box corners from old to new coordinate system.
    
    Args:
        original_corners: List of 8 corner points [x, y, z]
        old_matrix: Original 4x4 transformation matrix
        new_matrix: New 4x4 transformation matrix
        
    Returns:
        List of 8 transformed corner points
    """
    # Convert corners to homogeneous coordinates
    corners = np.array(original_corners)
    corners_homogeneous = np.hstack([corners, np.ones((len(corners), 1))])
    
    # Transform: first inverse of old transform, then apply new transform
    old_matrix_inv = np.linalg.inv(old_matrix)
    
    # Get local coordinates
    local_corners = (old_matrix_inv @ corners_homogeneous.T).T
    
    # Apply new transformation
    new_corners = (new_matrix @ local_corners.T).T
    
    # Convert back to 3D coordinates
    return new_corners[:, :3].tolist()


def update_object_transform(metadata_path: Path,
                           object_name: str,
                           new_translation: Tuple[float, float, float],
                           new_rotation_radians: Tuple[float, float, float]) -> None:
    """
    Update object transformation in metadata.json file.
    
    Args:
        metadata_path: Path to metadata.json file
        object_name: Name of the object to update
        new_translation: New (x, y, z) translation
        new_rotation_radians: New (rx, ry, rz) rotation in radians
    """
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Find the object
    if object_name not in metadata['static_objects']:
        raise ValueError(f"Object '{object_name}' not found in metadata")
    
    obj = metadata['static_objects'][object_name]
    
    # Get original scale (should remain unchanged)
    original_scale = tuple(obj['scale'])
    
    # Get original matrix for bbox transformation
    old_matrix = np.array(obj['matrix'])
    
    # Create new transformation matrix
    new_matrix = create_transform_matrix(new_translation, new_rotation_radians, original_scale)
    
    # Update matrix
    obj['matrix'] = new_matrix
    
    # Update position
    obj['position'] = list(new_translation)
    
    # Update rotation (store in radians)
    obj['rotation'] = list(new_rotation_radians)
    
    # Update bbox_corners
    original_corners = obj['bbox_corners']
    new_corners = transform_bbox_corners(original_corners, old_matrix, np.array(new_matrix))
    obj['bbox_corners'] = new_corners
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Successfully updated object '{object_name}'")
    print(f"  New position: {new_translation}")
    print(f"  New rotation (radians): {new_rotation_radians}")
    print(f"  Metadata saved to: {metadata_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def find_object_by_id(metadata_path: Path, object_id: str) -> str:
    """
    Find object name by its asset_id in metadata.
    
    Args:
        metadata_path: Path to metadata.json file
        object_id: Asset ID to search for
        
    Returns:
        Object name (key in static_objects dict)
        
    Raises:
        ValueError: If object with given ID not found
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    for object_name, obj_data in metadata['static_objects'].items():
        if obj_data.get('asset_id') == object_id:
            return object_name
    
    raise ValueError(f"No object found with asset_id '{object_id}'")


def create_backup(metadata_path: Path) -> None:
    """
    Create a backup of metadata.json if it doesn't already exist.
    
    Args:
        metadata_path: Path to metadata.json file
    """
    backup_path = metadata_path.with_name(metadata_path.stem + "_backup.json")
    
    if backup_path.exists():
        print(f"✓ Backup already exists: {backup_path}")
    else:
        with open(metadata_path, 'r') as src:
            content = src.read()
        with open(backup_path, 'w') as dst:
            dst.write(content)
        print(f"✓ Created backup: {backup_path}")


def main():
    """Main execution function."""
    # Construct full path to metadata file
    metadata_path = Path(SCENE_PATH) / METADATA_FILE
    
    # Verify file exists
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    print("=" * 70)
    print("Object Transform Update Script")
    print("=" * 70)
    
    # Create backup if enabled
    if CREATE_BACKUP:
        create_backup(metadata_path)
    
    # Find object name by ID
    object_name = find_object_by_id(metadata_path, OBJECT_ID)
    
    print(f"Scene path: {SCENE_PATH}")
    print(f"Object ID: {OBJECT_ID}")
    print(f"Object name: {object_name}")
    print(f"New translation: {NEW_TRANSLATION}")
    print(f"New rotation (degrees): {NEW_ROTATION_DEGREES}")
    print(f"New rotation (radians): {NEW_ROTATION_RADIANS}")
    print("=" * 70)
    
    # Perform update
    update_object_transform(
        metadata_path=metadata_path,
        object_name=object_name,
        new_translation=NEW_TRANSLATION,
        new_rotation_radians=NEW_ROTATION_RADIANS
    )
    
    print("=" * 70)
    print("Update complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
