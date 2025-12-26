#!/usr/bin/env python3
"""
Script 2: Record dataset from planned trajectories.

This script runs the recording pipeline:
1. Load configuration from experiment
2. Initialize SimulationApp (required for Isaac Sim)
3. Load planned trajectories
4. Replay in simulation and record observations
5. Save in LeRobot format

Usage:
    python 2_render_dataset.py --exp multi_object_open
    python 2_render_dataset.py --exp multi_object_open --headless
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


def run_recording(cfg: Config, headless: bool = False, max_episodes: int = None, dry_run: bool = False):
    """
    Run recording pipeline.
    
    Args:
        cfg: Configuration object
        headless: Whether to run in headless mode
        max_episodes: Maximum episodes to record
        dry_run: If True, skip environment setup
    """
    # Create task
    task = create_task(cfg)
    
    # Setup environment (if not dry run)
    if not dry_run:
        try:
            # SimulationApp is already initialized at the top of the script
            
            # Now safe to import SimEnvWrapper
            from automoma.simulation.env_wrapper import SimEnvWrapper
            
            env = SimEnvWrapper(cfg)
            env.setup_env()
            task.setup_env(env)
            logger.info("Environment setup complete")
        except Exception as e:
            import traceback
            logger.error(f"Could not setup environment: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            logger.warning("Running in dry-run mode")
            dry_run = True
    
    # Setup dataset wrapper
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
    
    # Get trajectory directory
    traj_dir = cfg.plan_cfg.output_dir if cfg.plan_cfg else "data/output"
    traj_dir = str(PROJECT_ROOT / traj_dir / "traj")
    
    print(f"[RENDER] Loading trajectories from: {traj_dir}")
    logger.info(f"Loading trajectories from: {traj_dir}")
    
    # Count available trajectory files
    from pathlib import Path
    traj_path = Path(traj_dir)
    traj_files = list(traj_path.glob("**/traj_data.pt"))
    print(f"[RENDER] Found {len(traj_files)} trajectory files")
    logger.info(f"Found {len(traj_files)} trajectory files")
    
    # Run recording pipeline
    if not dry_run and task.env is not None:
        print("[RENDER] Starting recording pipeline...")
        logger.info("Starting recording pipeline...")
        episodes = task.run_recording_pipeline(traj_dir, dataset_wrapper)
        print(f"[RENDER] Recorded {episodes} episodes")
        logger.info(f"Recorded {episodes} episodes")
    else:
        print("[RENDER] Dry run - no recording performed")
        logger.info("Dry run - no recording performed")
    
    # Finalize dataset
    dataset_wrapper.close()
    logger.info("Dataset finalized")


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
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration: {args.exp}")
        cfg = load_config(args.exp, PROJECT_ROOT)
        
        # Run recording
        logger.info("Starting run_recording function...")
        run_recording(cfg, args.headless, args.max_episodes, args.dry_run)
        logger.info("run_recording completed successfully")
        
    except Exception as e:
        import traceback
        logger.error(f"Fatal error in main: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
