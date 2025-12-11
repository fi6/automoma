#!/usr/bin/env python3
"""
Script to check and convert degree rotations to radians in all metadata.json files.
Scans all scenes under kitchen_1130 and converts rotation values if they appear to be in degrees.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple

# ============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# ============================================================================

BASE_PATH = "assets/scene/infinigen/kitchen_1130"
METADATA_FILENAME = "info/metadata.json"

# Threshold to detect if rotation is in degrees vs radians
# If any component is > pi/2 (1.5708), it's likely in degrees (except for special cases like 3.14159 for 180°)
# We'll use: if any value > 6.28 (2*pi), it's definitely in degrees
# Or if all values are between 0-360 (reasonable degree range), it's likely degrees
DEGREE_THRESHOLD = 6.29  # Any value > 2*pi is likely degrees
RADIAN_THRESHOLD = 3.15  # Values close to pi indicate radians

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def is_likely_degrees(rotation: List[float]) -> bool:
    """
    Determine if rotation values are likely in degrees rather than radians.
    
    Args:
        rotation: List of [rx, ry, rz] rotation values
        
    Returns:
        True if likely in degrees, False if likely in radians
    """
    # If any value is > 2*pi, definitely degrees
    if any(abs(r) > DEGREE_THRESHOLD for r in rotation):
        return True
    
    # If all values are reasonable degree values (0-360 range) and not close to typical radian values
    # Check if values are in the 0-360 range and don't match common radian values
    in_degree_range = all(0 <= abs(r) <= 360 for r in rotation)
    
    # Common radian values to check against
    common_radians = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6, np.pi, 3*np.pi/2, 2*np.pi, -np.pi, -np.pi/2]
    
    # Check if any value matches a common radian (within tolerance)
    matches_radian = any(abs(r - rad) < 0.01 or abs(r + rad) < 0.01 for r in rotation for rad in common_radians)
    
    # If in degree range and doesn't match common radians, likely degrees
    if in_degree_range and not matches_radian:
        # Additional check: if values are > 6.28, definitely degrees
        if any(abs(r) > 6.28 for r in rotation):
            return True
        # If all are small (< 2*pi), assume radians unless they're clearly degree values
        if all(abs(r) <= 2*np.pi for r in rotation) and any(abs(r) > 0.1 for r in rotation):
            return False
    
    return False


def convert_to_radians(rotation: List[float]) -> List[float]:
    """
    Convert rotation from degrees to radians.
    
    Args:
        rotation: Rotation values in degrees
        
    Returns:
        Rotation values in radians
    """
    return [np.radians(r) for r in rotation]


def process_metadata_file(metadata_path: Path) -> Tuple[bool, List[str]]:
    """
    Process a single metadata.json file and convert rotations if needed.
    
    Args:
        metadata_path: Path to metadata.json file
        
    Returns:
        Tuple of (was_modified, list_of_conversions)
    """
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return False, [f"Error reading file: {e}"]
    
    conversions = []
    was_modified = False
    
    if 'static_objects' not in metadata:
        return False, conversions
    
    for obj_name, obj_data in metadata['static_objects'].items():
        if 'rotation' not in obj_data:
            continue
        
        rotation = obj_data['rotation']
        
        # Check if rotation is in degrees
        if is_likely_degrees(rotation):
            old_rotation = rotation.copy()
            new_rotation = convert_to_radians(rotation)
            obj_data['rotation'] = new_rotation
            was_modified = True
            
            conversion_msg = (
                f"  {obj_name}:\n"
                f"    Object ID: {obj_data.get('asset_id', 'N/A')}\n"
                f"    Degrees: {old_rotation}\n"
                f"    Radians: {new_rotation}"
            )
            conversions.append(conversion_msg)
    
    # Save the modified metadata if changes were made
    if was_modified:
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            conversions.append("✓ File saved successfully")
        except IOError as e:
            conversions.append(f"✗ Error saving file: {e}")
            was_modified = False
    
    return was_modified, conversions


def main():
    """Main execution function."""
    base_path = Path(BASE_PATH)
    
    if not base_path.exists():
        print(f"Error: Base path not found: {base_path}")
        return
    
    print("=" * 80)
    print("Rotation Conversion Script - Scanning All Kitchen Scenes")
    print("=" * 80)
    print(f"Base path: {base_path}\n")
    
    # Find all metadata.json files
    metadata_files = list(base_path.glob(f"scene_*/info/metadata.json"))
    
    if not metadata_files:
        print(f"No metadata.json files found in {base_path}")
        return
    
    print(f"Found {len(metadata_files)} metadata.json files\n")
    
    total_conversions = 0
    scenes_modified = 0
    
    for metadata_path in sorted(metadata_files):
        scene_name = metadata_path.parent.parent.name
        was_modified, conversions = process_metadata_file(metadata_path)
        
        if conversions and conversions[0] != "✓ File saved successfully":
            if was_modified:
                print(f"Scene: {scene_name}")
                for conversion in conversions:
                    print(conversion)
                print()
                scenes_modified += 1
                total_conversions += len([c for c in conversions if c.startswith("  ")])
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total scenes scanned: {len(metadata_files)}")
    print(f"Scenes modified: {scenes_modified}")
    print(f"Total objects converted: {total_conversions}")
    print("=" * 80)


if __name__ == "__main__":
    main()
