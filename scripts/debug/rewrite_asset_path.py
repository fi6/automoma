#!/usr/bin/env python3
"""
Rewrite Asset Path Script

This script helps rewrite asset paths in files under object directories.
Given an object_id, it finds the corresponding directory under assets/object
and rewrites file contents with corrected asset paths.

Rewrite rules:
1. assets/robot/7221 -> assets/object/Microwave/7221 (or appropriate object type)
2. assets/robot/robot -> assets/robot

Usage:
    python scripts/debug/rewrite_asset_path.py --object_id 7221
    python scripts/debug/rewrite_asset_path.py --object_id 7221 --dry-run
    python scripts/debug/rewrite_asset_path.py --object_id 7221 --extensions .urdf,.yml

Examples:
    # Dry run to see what will be changed
    python scripts/debug/rewrite_asset_path.py --object_id 7221 --dry-run
    
    # Actually rewrite files
    python scripts/debug/rewrite_asset_path.py --object_id 7221
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional


def find_object_directory(object_id: str, assets_base: str = "assets/object") -> Optional[Tuple[str, str]]:
    """
    Find the object directory given an object_id.
    
    Args:
        object_id: Object ID (e.g., "7221")
        assets_base: Base path to search in (default: assets/object)
    
    Returns:
        Tuple of (object_type, object_path) or None if not found
        Example: ("Microwave", "assets/object/Microwave/7221")
    """
    assets_path = Path(assets_base)
    
    if not assets_path.exists():
        print(f"Error: Assets base path does not exist: {assets_base}")
        return None
    
    # Search for object_id in subdirectories
    for obj_type_dir in assets_path.iterdir():
        if not obj_type_dir.is_dir():
            continue
        
        obj_id_dir = obj_type_dir / object_id
        if obj_id_dir.exists() and obj_id_dir.is_dir():
            return (obj_type_dir.name, str(obj_id_dir))
    
    print(f"Error: Object directory not found for object_id {object_id}")
    return None


def get_rewrite_rules(object_id: str, object_type: str) -> List[Tuple[str, str, str]]:
    """
    Get the rewrite rules for a given object.
    
    Args:
        object_id: Object ID (e.g., "7221")
        object_type: Object type (e.g., "Microwave")
    
    Returns:
        List of tuples (pattern, replacement, description)
    """
    rules = [
        # Rule 1: assets/robot/{object_id} -> assets/object/{ObjectType}/{object_id}
        (
            f"assets/robot/{object_id}",
            f"assets/object/{object_type}/{object_id}",
            f"Replace robot-relative path with object-relative path (object_id)"
        ),
        # Rule 2: assets/robot/robot -> assets/robot
        (
            "assets/robot/robot",
            "assets/robot",
            "Replace redundant robot path (robot/robot -> robot)"
        ),
    ]
    return rules


def rewrite_file(file_path: str, rules: List[Tuple[str, str, str]], dry_run: bool = False) -> Tuple[bool, List[str]]:
    """
    Rewrite a single file with given rules.
    
    Args:
        file_path: Path to the file to rewrite
        rules: List of (pattern, replacement, description) tuples
        dry_run: If True, only show what would be changed without modifying
    
    Returns:
        Tuple of (was_modified, list_of_changes)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        print(f"Warning: Could not read file {file_path}: {e}")
        return False, []
    
    modified_content = original_content
    changes = []
    
    for pattern, replacement, description in rules:
        # Use regex with word boundaries to avoid partial matches
        # Escape special regex characters in the pattern
        escaped_pattern = re.escape(pattern)
        
        if re.search(escaped_pattern, modified_content):
            modified_content = re.sub(escaped_pattern, replacement, modified_content)
            changes.append(f"  - {description}: '{pattern}' -> '{replacement}'")
    
    if not changes:
        return False, []
    
    if not dry_run:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"✓ Modified: {file_path}")
        except Exception as e:
            print(f"Error: Could not write file {file_path}: {e}")
            return False, []
    else:
        print(f"[DRY-RUN] Would modify: {file_path}")
    
    return True, changes


def process_directory(obj_dir: str, object_id: str, object_type: str, extensions: List[str], dry_run: bool = False) -> None:
    """
    Process all files in a directory recursively.
    
    Args:
        obj_dir: Object directory to process
        object_id: Object ID
        object_type: Object type
        extensions: List of file extensions to process (e.g., ['.urdf', '.yml'])
        dry_run: If True, only show what would be changed
    """
    obj_path = Path(obj_dir)
    
    if not obj_path.exists():
        print(f"Error: Object directory does not exist: {obj_dir}")
        return
    
    rules = get_rewrite_rules(object_id, object_type)
    
    # Get all files matching extensions
    files_to_process = []
    for ext in extensions:
        files_to_process.extend(obj_path.rglob(f"*{ext}"))
    
    if not files_to_process:
        print(f"No files found with extensions {extensions} in {obj_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Processing object: {object_type}/{object_id}")
    print(f"Directory: {obj_dir}")
    print(f"Extensions: {extensions}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'LIVE'}")
    print(f"{'='*80}\n")
    
    total_files = len(files_to_process)
    modified_count = 0
    
    for file_path in sorted(files_to_process):
        was_modified, changes = rewrite_file(str(file_path), rules, dry_run)
        
        if was_modified:
            modified_count += 1
            for change in changes:
                print(change)
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Total files scanned: {total_files}")
    print(f"  Files modified: {modified_count}")
    if dry_run:
        print(f"  Mode: DRY-RUN (no files were actually modified)")
    print(f"{'='*80}\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Rewrite asset paths in object directory files"
    )
    parser.add_argument(
        "--object_id",
        type=str,
        required=True,
        help="Object ID to search for (e.g., 7221)"
    )
    parser.add_argument(
        "--assets_base",
        type=str,
        default="assets/object",
        help="Base path to search for objects (default: assets/object)"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".urdf,.yml,.yaml,.xml",
        help="File extensions to process, comma-separated (default: .urdf,.yml,.yaml,.xml)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    parser.add_argument(
        "--custom-pattern",
        type=str,
        help="Custom rewrite pattern in format 'old_pattern:new_pattern' (can use multiple times)",
        action="append"
    )
    
    args = parser.parse_args()
    
    # Parse extensions
    extensions = [ext.strip() for ext in args.extensions.split(",")]
    extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]
    
    # Find object directory
    result = find_object_directory(args.object_id, args.assets_base)
    if result is None:
        sys.exit(1)
    
    object_type, obj_dir = result
    
    # Process directory with base rules
    process_directory(obj_dir, args.object_id, object_type, extensions, args.dry_run)


if __name__ == "__main__":
    main()
