#!/usr/bin/env python3
"""
Script: Filter existing trajectories with potentially stricter constraints.

Usage:
    python scripts/debug/filter_traj.py --exp single_object_reach --scene scene_0_seed_0 --object 7221
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "third_party/curobo/src"))
sys.path.append(str(PROJECT_ROOT / "third_party/lerobot/src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('curobo').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from automoma.core.config_loader import load_plan_config, Config
from automoma.tasks.factory import create_task
from automoma.utils.file_utils import load_object_from_metadata, load_traj, save_traj
from automoma.core.types import StageType

def run_filtering(cfg: Config, scene_name: str, object_id: str):
    # Validate scene and object (same as 1_generate_plans)
    info_cfg = cfg.info_cfg
    if scene_name not in info_cfg.scene:
        logger.error(f"Scene '{scene_name}' not found in info_cfg.scene")
        return 1
    
    if object_id not in info_cfg.object:
        logger.error(f"Object '{object_id}' not found in info_cfg.object")
        return 1

    # Create task
    task = create_task(cfg)
    
    # Get scene and object config
    if not (cfg.env_cfg and cfg.env_cfg.scene_cfg and scene_name in cfg.env_cfg.scene_cfg):
        logger.error(f"Scene config not found in env_cfg: {scene_name}")
        return 1
    scene_cfg = cfg.env_cfg.scene_cfg[scene_name]
    
    if not (cfg.env_cfg and cfg.env_cfg.object_cfg and object_id in cfg.env_cfg.object_cfg):
         logger.error(f"Object config not found in env_cfg: {object_id}")
         return 1
    object_cfg = cfg.env_cfg.object_cfg[object_id]
    
    metadata_path = scene_cfg.metadata_path
    object_cfg = load_object_from_metadata(metadata_path, object_cfg)
    
    logger.info(f"Filtering: Scene={scene_name}, Object={object_id}")
    
    # Setup planner (needed for filtering)
    robot_cfg = cfg.env_cfg.robot_cfg if cfg.env_cfg else cfg.robot_cfg
    print("Setting up planner...")
    task.setup_planner(scene_cfg, object_cfg, robot_cfg)
    
    # Iterate through grasps
    grasp_poses = task.get_grasp_poses(object_cfg)
    
    total_processed = 0
    total_filtered = 0
    
    for grasp_idx, _ in enumerate(grasp_poses):
        # Construct path matching ReachTask logic
        stage_output_dir = os.path.join(
            task.output_dir, "traj",
            cfg.env_cfg.robot_cfg.robot_type,
            scene_name, object_id,
            f"grasp_{grasp_idx:04d}",
            "stage_0",
        )
        
        traj_path = os.path.join(stage_output_dir, "traj_data.pt")
        
        if not os.path.exists(traj_path):
            # It's common that some grasps failed to plan, so just debug log or simple info
            # logger.warning(f"Trajectory file not found: {traj_path}") 
            continue
            
        logger.info(f"Processing grasp {grasp_idx + 1}/{len(grasp_poses)}")
        
        # Load trajectory
        try:
            # We must use load_traj from utils
            traj_result = load_traj(traj_path)
            
            # Count before filtering
            original_success = traj_result.success.sum().item()
            
            if original_success == 0:
                logger.info("  No successful trajectories to filter.")
                continue

            stage_type = task.STAGES[0] # Usually MOVE
            
            # Run filtering
            filtered_result = task._filter_trajectories(
                traj_result,
                stage_type,
                object_id=object_id
            )
            
            new_success = filtered_result.success.sum().item()
            logger.info(f"  Filtering: {original_success} -> {new_success}")
            
            # Always save back so the file on disk reflects the new filtered state
            save_traj(filtered_result, traj_path)
            
            # Specifically count how many were REMOVED
            total_filtered += (original_success - new_success)
            total_processed += 1
            
        except Exception as e:
            logger.error(f"Error processing grasp {grasp_idx}: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"Filtering completed. Trajectories processed: {total_processed}. Removed count: {total_filtered}.")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Filter existing trajectories")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")
    parser.add_argument("--scene", type=str, required=True, help="Scene name")
    parser.add_argument("--object", type=str, required=True, help="Object ID")
    args = parser.parse_args()
    
    from automoma.core.config_loader import load_plan_config
    cfg = load_plan_config(args.exp, PROJECT_ROOT)
    
    return run_filtering(cfg, args.scene, args.object)

if __name__ == "__main__":
    sys.exit(main())
