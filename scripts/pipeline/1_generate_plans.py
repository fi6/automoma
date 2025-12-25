#!/usr/bin/env python3
"""
Script 1: Generate motion plans for manipulation tasks.

This script runs the planning pipeline to:
1. Load scene and object configurations
2. Compute IK solutions for grasp poses
3. Plan trajectories for articulated manipulation
4. Filter and save valid trajectories

Usage:
    python 1_generate_plans.py --config configs/exps/multi_object_open/plan.yaml
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from automoma.planning.pipeline import PlanningPipeline, PlanningResult
from automoma.utils.file_utils import (
    load_robot_cfg,
    process_robot_cfg,
    get_grasp_poses,
    load_object_from_metadata,
)
from automoma.core.types import StageType


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_planning_for_scene_object(
    plan_cfg: Dict[str, Any],
    scene_name: str,
    scene_cfg: Dict[str, Any],
    object_id: str,
    object_cfg: Dict[str, Any],
    robot_cfg_path: str,
) -> List[PlanningResult]:
    """Run planning for a single scene-object combination."""
    
    print(f"\n{'='*80}")
    print(f"Processing: Scene={scene_name}, Object={object_id}")
    print(f"{'='*80}")
    
    # Initialize planning pipeline
    pipeline = PlanningPipeline(plan_cfg)
    
    # Load object from metadata if available
    metadata_path = scene_cfg.get("metadata_path")
    if metadata_path and os.path.exists(metadata_path):
        object_cfg = load_object_from_metadata(metadata_path, object_cfg.copy())
    
    # Setup pipeline
    pipeline.setup(
        scene_cfg=scene_cfg,
        object_cfg=object_cfg,
        robot_cfg_path=robot_cfg_path,
    )
    
    # Load grasp poses
    grasp_dir = object_cfg.get("grasp_dir") or f"assets/object/{object_cfg['asset_type']}/{object_id}/grasp"
    num_grasps = plan_cfg.get("num_grasps", 20)
    scaling_factor = object_cfg.get("scale", 1.0)
    
    grasp_poses = get_grasp_poses(
        grasp_dir=grasp_dir,
        num_grasps=num_grasps,
        scaling_factor=scaling_factor,
    )
    
    if not grasp_poses:
        print(f"No grasp poses found for object {object_id}")
        return []
    
    print(f"Loaded {len(grasp_poses)} grasp poses")
    
    # Get articulation angles
    start_angles = object_cfg.get("start_angles", [0.0])
    goal_angles = object_cfg.get("goal_angles", [1.57])
    
    # Run planning
    robot_name = plan_cfg.get("info_cfg", {}).get("robot", "summit_franka")
    
    results = pipeline.run_full_pipeline(
        robot_name=robot_name,
        scene_name=scene_name,
        object_id=object_id,
        grasp_poses=grasp_poses,
        start_angles=start_angles,
        goal_angles=goal_angles,
    )
    
    # Print summary
    successful = sum(1 for r in results if r.success)
    print(f"\nPlanning Summary: {successful}/{len(results)} successful grasps")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate motion plans")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exps/multi_object_open/plan.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Specific scene to process (optional)",
    )
    parser.add_argument(
        "--object",
        type=str,
        default=None,
        help="Specific object to process (optional)",
    )
    args = parser.parse_args()
    
    # Load configuration
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return 1
    
    config = load_config(str(config_path))
    
    # Extract configurations
    info_cfg = config.get("info_cfg", {})
    plan_cfg = config.get("plan_cfg", {})
    plan_cfg["info_cfg"] = info_cfg  # Include for reference
    
    scene_cfgs = config.get("scene_cfg", {})
    object_cfgs = config.get("object_cfg", {})
    robot_cfg = config.get("robot_cfg", {})
    
    # Get robot config path
    robot_cfg_path = robot_cfg.get("path", f"assets/robot/{info_cfg.get('robot', 'summit_franka')}/{info_cfg.get('robot', 'summit_franka')}.yml")
    robot_cfg_path = str(PROJECT_ROOT / robot_cfg_path)
    
    # Get scenes and objects to process
    scenes = [args.scene] if args.scene else info_cfg.get("scene", list(scene_cfgs.keys()))
    objects = [args.object] if args.object else info_cfg.get("object", list(object_cfgs.keys()))
    
    all_results = []
    
    for scene_name in scenes:
        if scene_name not in scene_cfgs:
            print(f"Scene '{scene_name}' not found in config, skipping")
            continue
        
        scene_cfg = scene_cfgs[scene_name]
        # Make paths absolute
        scene_cfg["path"] = str(PROJECT_ROOT / scene_cfg["path"])
        if "metadata_path" in scene_cfg:
            scene_cfg["metadata_path"] = str(PROJECT_ROOT / scene_cfg["metadata_path"])
        
        for object_id in objects:
            if object_id not in object_cfgs:
                print(f"Object '{object_id}' not found in config, skipping")
                continue
            
            object_cfg = object_cfgs[object_id].copy()
            # Make paths absolute
            object_cfg["path"] = str(PROJECT_ROOT / object_cfg["path"])
            
            try:
                results = run_planning_for_scene_object(
                    plan_cfg=plan_cfg,
                    scene_name=scene_name,
                    scene_cfg=scene_cfg,
                    object_id=object_id,
                    object_cfg=object_cfg,
                    robot_cfg_path=robot_cfg_path,
                )
                all_results.extend(results)
                
            except Exception as e:
                print(f"Error processing {scene_name}/{object_id}: {e}")
                import traceback
                traceback.print_exc()
    
    # Final summary
    print(f"\n{'='*80}")
    print("PLANNING COMPLETE")
    print(f"{'='*80}")
    total = len(all_results)
    successful = sum(1 for r in all_results if r.success)
    print(f"Total grasps processed: {total}")
    print(f"Successful plans: {successful}")
    print(f"Success rate: {successful/total*100:.1f}%" if total > 0 else "N/A")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
