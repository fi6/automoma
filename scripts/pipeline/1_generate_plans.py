#!/usr/bin/env python3
"""
Script 1: Generate motion plans for manipulation tasks.

This script runs the planning pipeline using the task-based architecture:
1. Load configuration from experiment configs
2. Create the appropriate task
3. Run the planning pipeline for each scene/object

Usage:
    python 1_generate_plans.py --exp multi_object_open
    python 1_generate_plans.py --exp multi_object_open --scene scene_0_seed_0 --object 7221
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# --- CRITICAL INITIALIZATION ---
# Isaac Sim's SimulationApp MUST be initialized before any other imports that might touch pxr or omni.
# Planning with CuRobo requires a SimulationApp instance (even in headless mode).
# from automoma.simulation.sim_app_manager import get_simulation_app
# sim_app = get_simulation_app(headless=True)
# -------------------------------

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Reduce curobo logging to WARNING level
logging.getLogger('curobo').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from automoma.core.config_loader import load_config, Config
from automoma.tasks.factory import create_task
from automoma.utils.file_utils import load_object_from_metadata


def run_planning(cfg: Config, scene_filter: str = None, object_filter: str = None):
    """
    Run planning for all scenes and objects in configuration.
    
    Args:
        cfg: Configuration object
        scene_filter: Optional scene name to filter
        object_filter: Optional object ID to filter
    """
    # Create task from config
    task = create_task(cfg)
    
    # Get scenes and objects to process
    info_cfg = cfg.info_cfg
    scenes = info_cfg.scene if info_cfg.scene else []
    objects = info_cfg.object if info_cfg.object else []
    
    if scene_filter:
        scenes = [s for s in scenes if s == scene_filter]
    if object_filter:
        objects = [o for o in objects if o == object_filter]
    
    logger.info(f"Processing {len(scenes)} scenes x {len(objects)} objects")
    
    total_success = 0
    total_tasks = 0
    
    for scene_name in scenes:
        # Get scene config
        scene_cfg = cfg.scene_cfg[scene_name] if cfg.scene_cfg else None
        if scene_cfg is None:
            logger.warning(f"Scene config not found: {scene_name}")
            continue
        
        metadata_path = scene_cfg.metadata_path
        
        for object_id in objects:
            # Get object config
            object_cfg = cfg.object_cfg[object_id] if cfg.object_cfg else None
            if object_cfg is None:
                logger.warning(f"Object config not found: {object_id}")
                continue
            
            object_cfg = load_object_from_metadata(metadata_path, object_cfg)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Scene: {scene_name}, Object: {object_id}")
            logger.info(f"{'='*60}")
            
            try:
                # Setup planner with current scene/object
                task.setup_planner(scene_cfg, object_cfg, cfg.robot_cfg)
                
                # Run planning pipeline
                result = task.run_planning_pipeline(
                    scene_name=scene_name,
                    object_id=object_id,
                    object_cfg=object_cfg,
                )
                
                total_tasks += 1
                if result.success:
                    total_success += 1
                
                logger.info(f"Result: {'SUCCESS' if result.success else 'FAILED'}")
                logger.info(f"Trajectories: {result.successful_trajectories}/{result.total_trajectories}")
                
            except Exception as e:
                logger.error(f"Error processing {scene_name}/{object_id}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("PLANNING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total tasks: {total_tasks}")
    logger.info(f"Successful: {total_success}")
    if total_tasks > 0:
        logger.info(f"Success rate: {total_success/total_tasks*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Generate motion plans")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment name (e.g., 'multi_object_open')",
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
    logger.info(f"Loading configuration for experiment: {args.exp}")
    cfg = load_config(args.exp, PROJECT_ROOT)
    
    # Run planning
    run_planning(cfg, args.scene, args.object)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
