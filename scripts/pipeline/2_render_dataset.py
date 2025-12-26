#!/usr/bin/env python3
"""
Script 2: Record dataset from planned trajectories.

This script runs the recording pipeline:
1. Load configuration from experiment
2. Initialize SimulationApp (required for Isaac Sim)
3. Iterate through each scene-object combination
4. For each combination: load scene, object, robot
5. Load planned trajectories and replay in simulation
6. Record observations and save in LeRobot format

Usage:
    python 2_render_dataset.py --exp single_object_open_test
    python 2_render_dataset.py --exp single_object_open_test --headless
    python 2_render_dataset.py --exp single_object_open_test --scene scene_0_seed_0 --object 7221
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
# We do a quick parse of sys.argv to get the headless flag before importing anything else.
import argparse
temp_parser = argparse.ArgumentParser(add_help=False)
temp_parser.add_argument("--headless", action="store_true")
temp_args, _ = temp_parser.parse_known_args()

from automoma.simulation.sim_app_manager import get_simulation_app
sim_app = get_simulation_app(headless=temp_args.headless)
# -------------------------------

# Now safe to import other modules
from automoma.core.config_loader import load_config, Config
from automoma.tasks.factory import create_task


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_recording(cfg: Config, headless: bool = False, max_episodes: int = None, dry_run: bool = False, scene_filter: str = None, object_filter: str = None):
    """
    Run recording pipeline.
    
    Args:
        cfg: Configuration object
        headless: Whether to run in headless mode
        max_episodes: Maximum episodes to record
        dry_run: If True, skip environment setup
        scene_filter: Optional scene name to filter
        object_filter: Optional object ID to filter
    """
    # Get scenes and objects to process
    info_cfg = cfg.info_cfg
    scenes = info_cfg.scene if info_cfg.scene else []
    objects = info_cfg.object if info_cfg.object else []
    
    if scene_filter:
        scenes = [s for s in scenes if s == scene_filter]
    if object_filter:
        objects = [o for o in objects if o == object_filter]
    
    logger.info(f"Processing {len(scenes)} scenes x {len(objects)} objects")
    
    # Create task
    task = create_task(cfg)
    
    # Import SimEnvWrapper
    from automoma.simulation.env_wrapper import SimEnvWrapper
    from automoma.utils.file_utils import load_object_from_metadata
    
    total_episodes_recorded = 0
    
    # Process each scene-object combination
    for scene_name in scenes:
        # Get scene config
        scene_cfg = cfg.scene_cfg[scene_name] if cfg.scene_cfg else None
        if scene_cfg is None:
            logger.warning(f"Scene config not found: {scene_name}")
            continue
        
        metadata_path = scene_cfg.metadata_path if hasattr(scene_cfg, 'metadata_path') else None
        
        for object_id in objects:
            # Get object config
            object_cfg = cfg.object_cfg[object_id] if cfg.object_cfg else None
            if object_cfg is None:
                logger.warning(f"Object config not found: {object_id}")
                continue
            
            # Load object with metadata if available
            if metadata_path:
                object_cfg = load_object_from_metadata(metadata_path, object_cfg)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Recording: Scene={scene_name}, Object={object_id}")
            logger.info(f"{'='*60}")
            
            # Setup environment (if not dry run)
            if not dry_run:
                try:
                    # Create environment wrapper
                    env = SimEnvWrapper(cfg)
                    
                    # Convert Config objects to dict for setup_env
                    scene_cfg_dict = scene_cfg.to_dict() if hasattr(scene_cfg, 'to_dict') else scene_cfg
                    object_cfg_dict = object_cfg.to_dict() if hasattr(object_cfg, 'to_dict') else object_cfg
                    
                    # Setup environment with specific scene and object
                    env.setup_env(object_cfg=object_cfg_dict, scene_cfg=scene_cfg_dict)
                    task.setup_env(env)
                    logger.info("Environment setup complete")
                    
                    # Warmup step
                    # env.sim.step(step=-1) # DEBUG
                    
                except Exception as e:
                    import traceback
                    logger.error(f"Could not setup environment: {e}")
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
                    logger.warning("Skipping this scene-object combination")
                    continue
            
            # Setup dataset wrapper for this scene-object combination
            from automoma.datasets.dataset import LeRobotDatasetWrapper
            
            # Use dataset_cfg as base if available, otherwise record_cfg
            if cfg.dataset_cfg:
                dataset_config = cfg.dataset_cfg.to_dict()
            else:
                record_cfg = cfg.record_cfg if cfg.record_cfg else cfg
                dataset_config = {
                    "repo_id": record_cfg.repo_id if record_cfg.repo_id else f"automoma/{cfg.info_cfg.task}",
                    "root": record_cfg.root if record_cfg.root else "./datasets",
                    "fps": record_cfg.fps if record_cfg.fps else 15,
                    "use_videos": record_cfg.use_videos if record_cfg.use_videos else True,
                    "robot_type": cfg.robot_cfg.robot_type if cfg.robot_cfg else "summit_franka",
                }
            
            # Ensure camera config is included as 'camera' attribute for LeRobotDatasetWrapper
            if "camera" not in dataset_config and cfg.camera_cfg:
                dataset_config["camera"] = cfg.camera_cfg.to_dict()
            
            dataset_wrapper = LeRobotDatasetWrapper(Config(dataset_config))
            dataset_wrapper.create()
            
            # Get trajectory directory for this specific scene-object combination
            traj_dir = cfg.plan_cfg.output_dir if cfg.plan_cfg else "data/output"
            traj_dir = str(PROJECT_ROOT / traj_dir / "traj" / cfg.robot_cfg.robot_type / scene_name / object_id)
            
            logger.info(f"Loading trajectories from: {traj_dir}")
            
            # Check if trajectory directory exists
            from pathlib import Path
            traj_path = Path(traj_dir)
            if not traj_path.exists():
                logger.warning(f"Trajectory directory not found: {traj_dir}")
                dataset_wrapper.close()
                continue
            
            # Count available trajectory files
            traj_files = list(traj_path.glob("**/traj_data.pt"))
            logger.info(f"Found {len(traj_files)} trajectory files")
            
            # Run recording pipeline for this scene-object combination
            if not dry_run and task.env is not None:
                logger.info("Starting recording pipeline...")
                episodes = task.run_recording_pipeline(traj_dir, dataset_wrapper)
                logger.info(f"Recorded {episodes} episodes for {scene_name}/{object_id}")
                total_episodes_recorded += episodes
            else:
                logger.info("Dry run - no recording performed")
            
            # Finalize dataset for this combination
            dataset_wrapper.close()
            
            # Clean up environment before next iteration
            if not dry_run and task.env is not None:
                task.env.close()
                task.env = None
                logger.info("Environment closed for next iteration")
    
    logger.info(f"\n{'='*60}")
    logger.info("RECORDING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total episodes recorded: {total_episodes_recorded}")


def main():
    parser = argparse.ArgumentParser(description="Record dataset from trajectories")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment name",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum episodes to record",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without simulation",
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
    
    try:
        # Load configuration
        logger.info(f"Loading configuration: {args.exp}")
        cfg = load_config(args.exp, PROJECT_ROOT)
        
        # Run recording
        logger.info("Starting run_recording function...")
        run_recording(cfg, args.headless, args.max_episodes, args.dry_run, args.scene, args.object)
        logger.info("run_recording completed successfully")
        
    except Exception as e:
        import traceback
        logger.error(f"Fatal error in main: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
