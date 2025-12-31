#!/usr/bin/env python3
"""
Script 1: Generate motion plans for manipulation tasks.

This script runs the planning pipeline for ONE scene-object combination:
1. Load configuration from experiment configs
2. Validate scene and object are in info_cfg
3. Create the appropriate task
4. Run the planning pipeline

Usage:
    python 1_generate_plans.py --exp single_object_open_test --scene scene_0_seed_0 --object 7221
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Configure logging
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


def run_planning(cfg: Config, scene_name: str, object_id: str):
    """
    Run planning for ONE scene-object combination.
    
    Args:
        cfg: Configuration object
        scene_name: Scene name
        object_id: Object ID
    
    Returns:
        int: 0 for success, 1 for failure
    """
    # Validate scene is in info_cfg
    info_cfg = cfg.info_cfg
    if scene_name not in info_cfg.scene:
        logger.error(f"Scene '{scene_name}' not found in info_cfg.scene: {info_cfg.scene}")
        return 1
    
    # Validate object is in info_cfg
    if object_id not in info_cfg.object:
        logger.error(f"Object '{object_id}' not found in info_cfg.object: {info_cfg.object}")
        return 1
    
    # Create task from config
    task = create_task(cfg)
    
    # Get scene config
    scene_cfg = cfg.env_cfg.scene_cfg[scene_name] if cfg.env_cfg and cfg.env_cfg.scene_cfg else None
    if scene_cfg is None:
        logger.error(f"Scene config not found in env_cfg: {scene_name}")
        return 1
    
    metadata_path = scene_cfg.metadata_path
    
    # Get object config
    object_cfg = cfg.env_cfg.object_cfg[object_id] if cfg.env_cfg and cfg.env_cfg.object_cfg else None
    if object_cfg is None:
        logger.error(f"Object config not found in env_cfg: {object_id}")
        return 1
    
    # Load object with metadata
    object_cfg = load_object_from_metadata(metadata_path, object_cfg)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Planning: Scene={scene_name}, Object={object_id}")
    logger.info(f"{'='*60}")
    
    try:
        # Setup planner
        robot_cfg = cfg.env_cfg.robot_cfg if cfg.env_cfg else cfg.robot_cfg
        task.setup_planner(scene_cfg, object_cfg, robot_cfg)
        
        # Run planning pipeline
        result = task.run_planning_pipeline(
            scene_name=scene_name,
            object_id=object_id,
            object_cfg=object_cfg,
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Result: {'SUCCESS' if result.success else 'FAILED'}")
        logger.info(f"Trajectories: {result.successful_trajectories}/{result.total_trajectories}")
        logger.info(f"{'='*60}")
        
        return 0 if result.success else 1
        
    except Exception as e:
        logger.error(f"Error processing {scene_name}/{object_id}: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description="Generate motion plans for ONE scene-object combination")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment name (e.g., 'single_object_open_test')",
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene name (must be in info_cfg.scene)",
    )
    parser.add_argument(
        "--object",
        type=str,
        required=True,
        help="Object ID (must be in info_cfg.object)",
    )
    args = parser.parse_args()
    
    # Load planning configuration
    logger.info(f"Loading planning configuration: {args.exp}")
    from automoma.core.config_loader import load_plan_config
    cfg = load_plan_config(args.exp, PROJECT_ROOT)
    
    # Run planning for this scene-object combination
    return run_planning(cfg, args.scene, args.object)


if __name__ == "__main__":
    sys.exit(main())
