#!/usr/bin/env python3
"""
Script 2: Record dataset from planned trajectories.

This script runs the recording pipeline:
1. Load configuration from experiment
2. Load planned trajectories
3. Replay in simulation and record observations
4. Save in LeRobot format

Usage:
    python 2_render_dataset.py --exp multi_object_open
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from automoma.core.config_loader import load_config, Config
from automoma.tasks.task_factory import create_task
from automoma.simulation.env_wrapper import SimEnvWrapper
from automoma.datasets.dataset import LeRobotDatasetWrapper


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_recording(cfg: Config, max_episodes: int = None, dry_run: bool = False):
    """
    Run recording pipeline.
    
    Args:
        cfg: Configuration object
        max_episodes: Maximum episodes to record
        dry_run: If True, skip environment setup
    """
    # Create task
    task = create_task(cfg)
    
    # Setup environment (if not dry run)
    if not dry_run:
        try:
            env = SimEnvWrapper(cfg)
            env.setup_env()
            task.setup_env(env)
            logger.info("Environment setup complete")
        except Exception as e:
            logger.warning(f"Could not setup environment: {e}")
            logger.warning("Running in dry-run mode")
            dry_run = True
    
    # Setup dataset wrapper
    record_cfg = cfg.record_cfg if cfg.record_cfg else cfg
    
    dataset_config = {
        "repo_id": record_cfg.repo_id if record_cfg.repo_id else f"automoma/{cfg.info_cfg.task}",
        "root": record_cfg.root if record_cfg.root else "./datasets",
        "fps": record_cfg.fps if record_cfg.fps else 15,
        "use_videos": record_cfg.use_videos if record_cfg.use_videos else True,
        "robot_type": cfg.robot_cfg.robot_type if cfg.robot_cfg else "summit_franka",
    }
    
    dataset_wrapper = LeRobotDatasetWrapper(Config(dataset_config))
    dataset_wrapper.create()
    
    # Get trajectory directory
    traj_dir = cfg.plan_cfg.output_dir if cfg.plan_cfg else "data/output"
    traj_dir = str(PROJECT_ROOT / traj_dir / "traj")
    
    logger.info(f"Loading trajectories from: {traj_dir}")
    
    # Run recording pipeline
    if not dry_run and task.env is not None:
        episodes = task.run_recording_pipeline(traj_dir, dataset_wrapper)
        logger.info(f"Recorded {episodes} episodes")
    else:
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
    
    # Load configuration
    logger.info(f"Loading configuration: {args.exp}")
    cfg = load_config(args.exp, PROJECT_ROOT)
    
    # Run recording
    run_recording(cfg, args.max_episodes, args.dry_run)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
