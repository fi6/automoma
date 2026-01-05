#!/usr/bin/env python3
"""
Script 2: Record dataset from planned trajectories.

This script runs the recording pipeline:
1. Load configuration from experiment
2. Initialize SimulationApp (required for Isaac Sim)
3. Iterate through each scene-object combination
4. For each combination: load scene, object, robot
5. Load ALL planned trajectories (across all grasps)
6. Randomly sample max_episodes trajectories (if specified) using seed
7. Replay sampled trajectories in simulation and record observations
8. Save in LeRobot format

Usage:
    # Use max_episodes from config (record.yaml)
    python 2_render_dataset.py --exp single_object_open_test
    
    # Override max_episodes via command line
    python 2_render_dataset.py --exp single_object_open_test --max-episodes 10
    
    # Run in headless mode
    python 2_render_dataset.py --exp single_object_open_test --headless
    
    # Process specific scene-object combination
    python 2_render_dataset.py --exp single_object_open_test --scene scene_0_seed_0 --object 7221

Note:
    - max_episodes applies per scene-object combination
    - Random sampling uses seed from record_cfg.seed for reproducibility
    - All trajectories across grasps are loaded before sampling
"""

import os
import sys
import argparse
import logging
import torch
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

# Initialize SimulationApp early since this script always needs simulation
from automoma.utils.sim_utils import get_simulation_app
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


def load_or_generate_successful_trajs(traj_dir: str, seed: int = None) -> torch.Tensor:
    """
    Load successful trajectories from cache or generate and cache them.
    
    If all_successful_trajs.pt exists in traj_dir, load and return it.
    Otherwise, load all trajectory files, concatenate them, randomize order, save to cache, and return.
    
    Args:
        traj_dir: Directory containing trajectory files
        seed: Random seed for shuffling (if provided)
    
    Returns:
        torch.Tensor: Concatenated successful trajectories with randomized order
    """
    cache_file = Path(traj_dir) / "all_successful_trajs.pt"
    
    # Check if cache exists
    if cache_file.exists():
        logger.info(f"Loading cached trajectories from: {cache_file}")
        all_successful_trajs = torch.load(cache_file, weights_only=False)
        logger.info(f"Loaded {len(all_successful_trajs)} trajectories from cache")
        return all_successful_trajs
    
    # Load all trajectory files
    traj_path = Path(traj_dir)
    traj_files = list(traj_path.glob("**/traj_data.pt"))
    logger.info(f"Found {len(traj_files)} trajectory files")
    
    # Load and collect all successful trajectories
    all_successful_trajs = []
    for traj_file in traj_files:
        try:
            traj_data = torch.load(traj_file, weights_only=False)
            trajectories = traj_data["trajectories"]
            
            # Filter successful trajectories
            if "success" in traj_data:
                success_mask = traj_data["success"]
                successful_trajs = trajectories[success_mask]
                if len(successful_trajs) > 0:
                    all_successful_trajs.append(successful_trajs)
            else:
                if len(trajectories) > 0:
                    all_successful_trajs.append(trajectories)
        except Exception as e:
            logger.warning(f"Error loading {traj_file}: {e}")
    
    if not all_successful_trajs:
        raise RuntimeError(f"No successful trajectories found in {traj_dir}")
    
    # Concatenate all successful trajectories
    all_successful_trajs = torch.cat(all_successful_trajs, dim=0)
    logger.info(f"Total successful trajectories: {len(all_successful_trajs)}")
    
    # Randomize order
    if seed is not None:
        torch.manual_seed(seed)
        logger.info(f"Shuffling trajectories with seed={seed}")
    
    indices = torch.randperm(len(all_successful_trajs))
    print(f"Shuffled trajectory indices: {indices.tolist()}")
    all_successful_trajs = all_successful_trajs[indices]
    
    # Save to cache
    logger.info(f"Saving cached trajectories to: {cache_file}")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_successful_trajs, cache_file)
    
    return all_successful_trajs


def run_recording(cfg: Config, scene_name: str, object_id: str, max_episodes: int = None, dry_run: bool = False):
    """
    Run recording for ONE scene-object combination.
    
    Args:
        cfg: Configuration object
        scene_name: Scene name
        object_id: Object ID
        max_episodes: Maximum episodes to record (overrides config)
        dry_run: If True, skip environment setup
    
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
    
    # Get max_episodes from config or argument
    if max_episodes is None:
        if hasattr(cfg, 'record_cfg') and hasattr(cfg.record_cfg, 'max_episodes'):
            max_episodes = cfg.record_cfg.max_episodes
        else:
            max_episodes = None  # No limit
    
    # Get seed from config
    seed = None
    if hasattr(cfg, 'record_cfg') and hasattr(cfg.record_cfg, 'seed'):
        seed = cfg.record_cfg.seed
    
    logger.info(f"Max episodes: {max_episodes}")
    logger.info(f"Random seed: {seed}")
    
    # Create task
    task = create_task(cfg)
    
    # Import SimEnvWrapper
    from automoma.simulation.env_wrapper import SimEnvWrapper
    from automoma.utils.file_utils import load_object_from_metadata
    
    # Get scene config
    scene_cfg = cfg.env_cfg.scene_cfg[scene_name] if cfg.env_cfg and cfg.env_cfg.scene_cfg else None
    if scene_cfg is None:
        logger.error(f"Scene config not found: {scene_name}")
        return 1
    
    metadata_path = scene_cfg.metadata_path if hasattr(scene_cfg, 'metadata_path') else None
    
    # Get object config
    object_cfg = cfg.env_cfg.object_cfg[object_id] if cfg.env_cfg and cfg.env_cfg.object_cfg else None
    if object_cfg is None:
        logger.error(f"Object config not found: {object_id}")
        return 1
    
    # Load object with metadata if available
    if metadata_path:
        object_cfg = load_object_from_metadata(metadata_path, object_cfg)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Recording: Scene={scene_name}, Object={object_id}")
    logger.info(f"{'='*60}")
    
    episodes_recorded = 0
    
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
            
        except Exception as e:
            import traceback
            logger.error(f"Could not setup environment: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return 1
    
    # Setup dataset wrapper
    from automoma.datasets.dataset import LeRobotDatasetWrapper
    
    if not cfg.record_cfg or not cfg.record_cfg.dataset_cfg:
        logger.error("record_cfg.dataset_cfg is required in config")
        return 1
    
    dataset_config = cfg.record_cfg.dataset_cfg.to_dict()
    
    # Modify repo_id to include scene and object: {base_repo_id}_{object_id}_{scene_id}
    base_repo_id = dataset_config.get("repo_id", "default")
    modified_repo_id = f"{base_repo_id}_{object_id}_{scene_name}"
    dataset_config["repo_id"] = modified_repo_id
    logger.info(f"Dataset repo_id: {modified_repo_id}")
    
    if "camera" not in dataset_config and cfg.env_cfg and cfg.env_cfg.camera_cfg:
        dataset_config["camera"] = cfg.env_cfg.camera_cfg.to_dict()
    
    dataset_wrapper = LeRobotDatasetWrapper(Config(dataset_config))
    dataset_wrapper.create()
    
    # Get trajectory directory
    if not cfg.record_cfg or not cfg.record_cfg.data_cfg or not cfg.record_cfg.data_cfg.traj_dir:
        logger.error("record_cfg.data_cfg.traj_dir is required in config")
        dataset_wrapper.close()
        return 1
    
    traj_dir = cfg.record_cfg.data_cfg.traj_dir
    if not cfg.env_cfg or not cfg.env_cfg.robot_cfg or not cfg.env_cfg.robot_cfg.robot_type:
        logger.error("env_cfg.robot_cfg.robot_type is required in config")
        dataset_wrapper.close()
        return 1
    
    traj_dir = str(PROJECT_ROOT / traj_dir / cfg.env_cfg.robot_cfg.robot_type / scene_name / object_id)
    logger.info(f"Loading trajectories from: {traj_dir}")
    
    # Check if trajectory directory exists
    traj_path = Path(traj_dir)
    if not traj_path.exists():
        logger.warning(f"Trajectory directory not found: {traj_dir}")
        dataset_wrapper.close()
        return 1
    
    # Load or generate successful trajectories (with caching)
    try:
        all_successful_trajs = load_or_generate_successful_trajs(traj_dir, seed=seed)
    except RuntimeError as e:
        logger.warning(f"No successful trajectories found: {e}")
        dataset_wrapper.close()
        return 1
    
    # Trajectories if max_episodes is set
    if max_episodes is not None and len(all_successful_trajs) > max_episodes:
        sampled_trajs = all_successful_trajs[:max_episodes]
        logger.info(f"Randomly sampled {max_episodes} trajectories (seed={seed})")
    else:
        sampled_trajs = all_successful_trajs
        logger.info(f"Using all {len(sampled_trajs)} trajectories")
    
    # Run recording pipeline
    if not dry_run and task.env is not None:
        logger.info("Starting recording pipeline...")
        episodes_recorded = task.run_recording_pipeline_with_trajs(sampled_trajs, dataset_wrapper)
        logger.info(f"Recorded {episodes_recorded} episodes")
    else:
        logger.info("Dry run - no recording performed")
    
    # Finalize dataset
    dataset_wrapper.close()
    
    # Clean up environment
    if not dry_run and task.env is not None:
        task.env.close()
        task.env = None
    
    logger.info(f"Completed: {episodes_recorded} episodes recorded")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Record dataset from trajectories for ONE scene-object combination")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment name",
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
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum episodes to record (overrides config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without simulation",
    )
    args = parser.parse_args()
    
    try:
        # Load record configuration
        logger.info(f"Loading record configuration: {args.exp}")
        from automoma.core.config_loader import load_record_config
        cfg = load_record_config(args.exp, PROJECT_ROOT)
        
        # Run recording for this scene-object combination
        return run_recording(cfg, args.scene, args.object, args.max_episodes, args.dry_run)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
