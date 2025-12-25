#!/usr/bin/env python3
"""
Script 2: Render dataset by replaying trajectories in simulation.

This script:
1. Loads planned trajectories
2. Replays them in Isaac Sim
3. Records camera observations
4. Saves data in LeRobot format

Usage:
    python 2_render_dataset.py --config configs/exps/multi_object_open/record.yaml
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from automoma.core.types import DatasetFormat
from automoma.datasets.dataset import LeRobotDatasetWrapper
from automoma.utils.file_utils import load_robot_cfg, load_traj


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class DatasetRecorder:
    """Records trajectory data with camera observations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_cfg = config.get("dataset_cfg", {})
        self.env_cfg = config.get("env_cfg", {})
        self.camera_cfg = config.get("camera_cfg", {})
        
        self.env_wrapper = None
        self.dataset_wrapper = None
    
    def setup_environment(self) -> None:
        """Setup Isaac Sim environment (lazy import to avoid startup issues)."""
        try:
            from automoma.simulation.env_wrapper import SimEnvWrapper
            self.env_wrapper = SimEnvWrapper(self.env_cfg)
            self.env_wrapper.setup_env()
            print("Environment setup complete")
        except ImportError as e:
            print(f"Warning: Could not import simulation modules: {e}")
            print("Running in dry-run mode without simulation")
    
    def setup_dataset(self) -> None:
        """Setup dataset wrapper for recording."""
        dataset_config = {
            "repo_id": self.dataset_cfg.get("repo_id", "automoma_dataset"),
            "root": self.dataset_cfg.get("root", "./datasets"),
            "fps": self.dataset_cfg.get("fps", 15),
            "use_videos": self.dataset_cfg.get("use_videos", True),
            "push_to_hub": self.dataset_cfg.get("push_to_hub", False),
            "robot_type": self.dataset_cfg.get("robot_type", "summit_franka"),
            "state_dim": self.dataset_cfg.get("state_dim", 12),
            "state_names": self.dataset_cfg.get("state_names", []),
            "camera": {
                "names": self.camera_cfg.get("names", ["ego_topdown", "ego_wrist", "fix_local"]),
                "width": self.camera_cfg.get("width", 320),
                "height": self.camera_cfg.get("height", 240),
            },
            "task": self.dataset_cfg.get("task", "manipulation"),
        }
        
        # Convert to object-like access
        class ConfigObj:
            def __init__(self, d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, ConfigObj(v))
                    else:
                        setattr(self, k, v)
            def get(self, k, default=None):
                return getattr(self, k, default)
        
        self.dataset_wrapper = LeRobotDatasetWrapper(ConfigObj(dataset_config))
        self.dataset_wrapper.create()
        print("Dataset wrapper initialized")
    
    def record_trajectory(
        self,
        trajectory: torch.Tensor,
        robot_name: str,
        task_description: str = "",
    ) -> bool:
        """
        Record a single trajectory with observations.
        
        Args:
            trajectory: Trajectory tensor (T, num_joints)
            robot_name: Name of the robot
            task_description: Task description for the episode
            
        Returns:
            True if successful
        """
        if self.env_wrapper is None:
            print("No environment available, skipping recording")
            return False
        
        try:
            for step_idx, step_pose in enumerate(trajectory):
                # Set robot state
                robot_state = step_pose[:-1]  # All but last (handle angle)
                env_state = step_pose[-1:]    # Last (handle angle)
                
                self.env_wrapper.set_state(robot_state, env_state)
                self.env_wrapper.step()
                
                # Collect observation data
                data = self.env_wrapper.get_data()
                data["task"] = task_description
                
                # Add to dataset
                self.dataset_wrapper.add(data)
            
            # Save episode
            self.dataset_wrapper.save()
            return True
            
        except Exception as e:
            print(f"Error recording trajectory: {e}")
            return False
    
    def process_trajectory_files(
        self,
        traj_dir: str,
        robot_name: str,
        max_episodes: Optional[int] = None,
    ) -> int:
        """
        Process all trajectory files in a directory.
        
        Args:
            traj_dir: Directory containing trajectory files
            robot_name: Name of the robot
            max_episodes: Maximum number of episodes to record
            
        Returns:
            Number of episodes recorded
        """
        traj_dir = Path(traj_dir)
        if not traj_dir.exists():
            print(f"Trajectory directory not found: {traj_dir}")
            return 0
        
        # Find all trajectory files
        traj_files = list(traj_dir.glob("**/filtered_traj_data.pt"))
        if not traj_files:
            traj_files = list(traj_dir.glob("**/traj_data.pt"))
        
        print(f"Found {len(traj_files)} trajectory files")
        
        if max_episodes:
            traj_files = traj_files[:max_episodes]
        
        recorded = 0
        for traj_file in tqdm(traj_files, desc="Recording trajectories"):
            try:
                traj_data = torch.load(traj_file, weights_only=False)
                trajectories = traj_data["trajectories"]
                success = traj_data["success"]
                
                # Only record successful trajectories
                success_mask = success.bool()
                successful_trajs = trajectories[success_mask]
                
                for traj in successful_trajs:
                    if self.record_trajectory(
                        trajectory=traj,
                        robot_name=robot_name,
                        task_description=f"Manipulation task from {traj_file.parent.name}",
                    ):
                        recorded += 1
                        
                        if max_episodes and recorded >= max_episodes:
                            break
                
                if max_episodes and recorded >= max_episodes:
                    break
                    
            except Exception as e:
                print(f"Error processing {traj_file}: {e}")
        
        return recorded
    
    def finalize(self) -> None:
        """Finalize the dataset."""
        if self.dataset_wrapper is not None:
            self.dataset_wrapper.close()
            print("Dataset finalized")


def main():
    parser = argparse.ArgumentParser(description="Render dataset from trajectories")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exps/multi_object_open/record.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--traj-dir",
        type=str,
        default=None,
        help="Directory containing trajectory data (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for dataset (overrides config)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to record",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without simulation (for testing)",
    )
    args = parser.parse_args()
    
    # Load configuration
    config_path = PROJECT_ROOT / args.config
    
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        print(f"Config not found at {config_path}, using defaults")
        config = {}
    
    # Override with command line arguments
    if args.traj_dir:
        config["traj_dir"] = args.traj_dir
    if args.output_dir:
        config.setdefault("dataset_cfg", {})["root"] = args.output_dir
    
    # Get trajectory directory
    traj_dir = config.get("traj_dir", "data/run_1223/traj")
    traj_dir = str(PROJECT_ROOT / traj_dir)
    
    # Get robot name
    robot_name = config.get("info_cfg", {}).get("robot", "summit_franka")
    
    # Initialize recorder
    recorder = DatasetRecorder(config)
    
    try:
        # Setup
        if not args.dry_run:
            recorder.setup_environment()
        recorder.setup_dataset()
        
        # Record trajectories
        num_recorded = recorder.process_trajectory_files(
            traj_dir=traj_dir,
            robot_name=robot_name,
            max_episodes=args.max_episodes,
        )
        
        print(f"\nRecorded {num_recorded} episodes")
        
        # Finalize
        recorder.finalize()
        
    except Exception as e:
        print(f"Error during recording: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nDataset recording complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
