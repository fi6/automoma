#!/usr/bin/env python3
"""
Analyze Scene Objects Script

This script analyzes scene metadata files and generates statistics about objects
contained in each scene across the infinigen_scene_100 directory.

Usage:
    python scripts/analyze_scene_objects.py --scene_dir output/collect/infinigen_scene_100
    python scripts/analyze_scene_objects.py --scene_dir output/collect/infinigen_scene_100 --output output/statistics/scene_objects.csv
    python scripts/analyze_scene_objects.py --scene_dir output/collect/infinigen_scene_100 --output output/statistics/scene_objects.json

Examples:
    python scripts/analyze_scene_objects.py --scene_dir output/collect/infinigen_scene_100
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


def load_scene_metadata(metadata_path: str) -> Dict[str, Any]:
    """Load metadata from a scene metadata.json file."""
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load metadata from {metadata_path}: {e}")
        return None


def extract_objects_from_metadata(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract object information from scene metadata."""
    objects = []
    
    if metadata is None or "static_objects" not in metadata:
        return objects
    
    static_objects = metadata["static_objects"]
    
    for obj_name, obj_data in static_objects.items():
        obj_info = {
            "name": obj_data.get("name", ""),
            "asset_type": obj_data.get("asset_type", ""),
            "asset_id": obj_data.get("asset_id", ""),
            "position": obj_data.get("position", []),
            "rotation": obj_data.get("rotation", []),
            "scale": obj_data.get("scale", []),
            "dimensions": obj_data.get("dimensions", []),
        }
        objects.append(obj_info)
    
    return objects


def analyze_scene_directory(scene_dir: str) -> Dict[str, Any]:
    """
    Analyze all scenes in a directory and collect object information.
    
    Returns a dictionary with scene-level statistics and a list of all scenes with their objects.
    """
    scene_path = Path(scene_dir)
    
    if not scene_path.exists():
        print(f"Error: Scene directory {scene_dir} does not exist")
        return None
    
    scenes_data = []
    object_counts = {}
    total_objects = 0
    
    # Find all scene directories
    scene_dirs = sorted([d for d in scene_path.iterdir() 
                        if d.is_dir() and d.name.startswith("scene_")])
    
    print(f"Found {len(scene_dirs)} scenes")
    print("=" * 120)
    
    for scene_subdir in scene_dirs:
        scene_name = scene_subdir.name
        metadata_path = scene_subdir / "info" / "metadata.json"
        
        if not metadata_path.exists():
            print(f"Warning: metadata.json not found for {scene_name}")
            continue
        
        # Load metadata
        metadata = load_scene_metadata(str(metadata_path))
        if metadata is None:
            continue
        
        # Extract objects
        objects = extract_objects_from_metadata(metadata)
        
        scene_data = {
            "scene_name": scene_name,
            "total_objects": len(objects),
            "objects": objects,
            "metadata_path": str(metadata_path),
        }
        scenes_data.append(scene_data)
        
        # Count object types
        for obj in objects:
            asset_type = obj.get("asset_type", "Unknown")
            asset_id = obj.get("asset_id", "")
            key = f"{asset_type} ({asset_id})"
            
            if key not in object_counts:
                object_counts[key] = 0
            object_counts[key] += 1
            total_objects += 1
    
    stats = {
        "scene_dir": str(scene_path),
        "total_scenes": len(scenes_data),
        "total_objects": total_objects,
        "unique_object_types": len(object_counts),
        "object_counts": object_counts,
        "scenes": scenes_data,
    }
    
    return stats


def print_scene_objects_table(stats: Dict[str, Any]):
    """Print a formatted table of objects in each scene."""
    print("\n" + "=" * 120)
    print("SCENE OBJECTS TABLE")
    print("=" * 120)
    print()
    
    if stats is None or not stats.get("scenes"):
        print("No scenes found")
        return
    
    for scene_data in stats["scenes"]:
        scene_name = scene_data["scene_name"]
        objects = scene_data["objects"]
        
        print(f"{'Scene:':<30} {scene_name}")
        print(f"{'Total Objects:':<30} {len(objects)}")
        print("-" * 120)
        print(f"{'Asset Type':<25} {'Asset ID':<15} {'Position (X, Y, Z)':<40} {'Scale':<20}")
        print("-" * 120)
        
        for obj in objects:
            asset_type = obj.get("asset_type", "")
            asset_id = obj.get("asset_id", "")
            position = obj.get("position", [])
            scale = obj.get("scale", [])
            
            pos_str = f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})" if position else "N/A"
            scale_str = f"({scale[0]:.4f}, {scale[1]:.4f}, {scale[2]:.4f})" if scale else "N/A"
            
            print(f"{asset_type:<25} {asset_id:<15} {pos_str:<40} {scale_str:<20}")
        
        print()


def print_object_summary(stats: Dict[str, Any]):
    """Print a summary of all objects across scenes."""
    print("\n" + "=" * 120)
    print("OBJECT TYPE SUMMARY")
    print("=" * 120)
    print()
    
    if stats is None:
        print("No statistics available")
        return
    
    print(f"Total Scenes: {stats['total_scenes']}")
    print(f"Total Objects: {stats['total_objects']}")
    print(f"Unique Object Types: {stats['unique_object_types']}")
    print()
    
    print(f"{'Object Type (ID)':<40} {'Count':<10} {'Percentage':<15}")
    print("-" * 120)
    
    for obj_type, count in sorted(stats['object_counts'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_objects'] * 100) if stats['total_objects'] > 0 else 0
        print(f"{obj_type:<40} {count:<10} {percentage:>6.2f}%")
    
    print()


def create_objects_dataframe(stats: Dict[str, Any]) -> pd.DataFrame:
    """Create a pandas DataFrame with all objects across scenes."""
    records = []
    
    if stats is None or not stats.get("scenes"):
        return None
    
    for scene_data in stats["scenes"]:
        scene_name = scene_data["scene_name"]
        
        for obj in scene_data["objects"]:
            record = {
                "scene_name": scene_name,
                "asset_type": obj.get("asset_type", ""),
                "asset_id": obj.get("asset_id", ""),
                "position_x": obj.get("position", [None])[0] if obj.get("position") else None,
                "position_y": obj.get("position", [None, None])[1] if obj.get("position") and len(obj.get("position", [])) > 1 else None,
                "position_z": obj.get("position", [None, None, None])[2] if obj.get("position") and len(obj.get("position", [])) > 2 else None,
                "scale_x": obj.get("scale", [None])[0] if obj.get("scale") else None,
                "scale_y": obj.get("scale", [None, None])[1] if obj.get("scale") and len(obj.get("scale", [])) > 1 else None,
                "scale_z": obj.get("scale", [None, None, None])[2] if obj.get("scale") and len(obj.get("scale", [])) > 2 else None,
            }
            records.append(record)
    
    if not records:
        return None
    
    df = pd.DataFrame(records)
    return df


def create_scene_summary_dataframe(stats: Dict[str, Any]) -> pd.DataFrame:
    """Create a pandas DataFrame with scene-level summaries."""
    records = []
    
    if stats is None or not stats.get("scenes"):
        return None
    
    for scene_data in stats["scenes"]:
        scene_name = scene_data["scene_name"]
        objects = scene_data["objects"]
        
        # Count by asset type
        asset_counts = {}
        for obj in objects:
            asset_type = obj.get("asset_type", "Unknown")
            asset_id = obj.get("asset_id", "")
            key = f"{asset_type} ({asset_id})"
            asset_counts[key] = asset_counts.get(key, 0) + 1
        
        record = {
            "scene_name": scene_name,
            "total_objects": len(objects),
        }
        
        # Add individual asset counts
        for asset_type, count in asset_counts.items():
            record[asset_type] = count
        
        records.append(record)
    
    if not records:
        return None
    
    df = pd.DataFrame(records)
    df = df.fillna(0).astype({col: int for col in df.columns if col != "scene_name"})
    return df


def save_to_csv(df: pd.DataFrame, output_path: str):
    """Save DataFrame to CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(output_path), index=False)
    print(f"Saved to CSV: {output_path}")


def save_to_json(stats: Dict[str, Any], output_path: str):
    """Save statistics to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    json_data = {
        "scene_dir": stats["scene_dir"],
        "total_scenes": stats["total_scenes"],
        "total_objects": stats["total_objects"],
        "unique_object_types": stats["unique_object_types"],
        "object_counts": stats["object_counts"],
        "scenes": [
            {
                "scene_name": scene["scene_name"],
                "total_objects": scene["total_objects"],
                "objects": scene["objects"],
            }
            for scene in stats["scenes"]
        ]
    }
    
    with open(str(output_path), 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved to JSON: {output_path}")


def main():
    """Main function to analyze scene objects."""
    parser = argparse.ArgumentParser(
        description="Analyze scene objects from metadata files"
    )
    parser.add_argument(
        "--scene_dir",
        type=str,
        default="output/collect/infinigen_scene_100",
        help="Directory containing scene subdirectories"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for statistics (CSV or JSON based on extension)"
    )
    parser.add_argument(
        "--scene_summary",
        action="store_true",
        help="Generate scene summary table (objects per scene)"
    )
    
    args = parser.parse_args()
    
    print(f"Analyzing scenes in: {args.scene_dir}")
    print()
    
    # Analyze scene directory
    stats = analyze_scene_directory(args.scene_dir)
    
    if stats is None:
        print("Failed to analyze scenes")
        return
    
    # Print tables
    print_scene_objects_table(stats)
    print_object_summary(stats)
    
    # Generate scene summary if requested
    if args.scene_summary:
        print("\n" + "=" * 120)
        print("SCENE SUMMARY TABLE")
        print("=" * 120)
        print()
        
        scene_summary_df = create_scene_summary_dataframe(stats)
        if scene_summary_df is not None:
            print(scene_summary_df.to_string(index=False))
            print()
    
    # Save output if specified
    if args.output:
        output_path = Path(args.output)
        
        if args.output.endswith('.csv'):
            # Create detailed objects dataframe
            df = create_objects_dataframe(stats)
            if df is not None:
                save_to_csv(df, args.output)
        elif args.output.endswith('.json'):
            save_to_json(stats, args.output)
        else:
            # Default to CSV
            df = create_objects_dataframe(stats)
            if df is not None:
                csv_path = str(output_path) + ".csv"
                save_to_csv(df, csv_path)
                
                # Also save JSON
                json_path = str(output_path) + ".json"
                save_to_json(stats, json_path)
    
    print("\n" + "=" * 120)
    print("Analysis complete!")
    print("=" * 120)


if __name__ == "__main__":
    main()
