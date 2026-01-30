"""
Data loading utilities for IK and trajectory data.

This module provides robust functions to load trajectory data, IK data,
and handle missing files gracefully.

Author: Statistical Analysis Suite for AutoMoMA
Date: 2025-10-16
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import warnings

from config import (ROBOT_NAME, SCENE_NAMES, OBJECT_NAMES, GRASP_IDS, 
                    DATA_ROOT, get_traj_data_path, get_ik_data_path)

@dataclass
class TrajectoryData:
    """Container for trajectory data."""
    start_state: torch.Tensor  # [N, DoF]
    goal_state: torch.Tensor   # [N, DoF]
    traj: torch.Tensor         # [N, T, DoF]
    success: torch.Tensor      # [N]
    
    @property
    def num_trajectories(self) -> int:
        return self.start_state.shape[0]
    
    @property
    def num_timesteps(self) -> int:
        return self.traj.shape[1]
    
    @property
    def num_dof(self) -> int:
        return self.start_state.shape[1]
    
    @property
    def success_rate(self) -> float:
        return self.success.float().mean().item()
    
    @property
    def num_successful(self) -> int:
        return self.success.sum().item()
    
    def to_numpy(self) -> 'TrajectoryData':
        """Convert all tensors to numpy arrays."""
        return TrajectoryData(
            start_state=self.start_state.cpu().numpy(),
            goal_state=self.goal_state.cpu().numpy(),
            traj=self.traj.cpu().numpy(),
            success=self.success.cpu().numpy()
        )


@dataclass
class IKData:
    """Container for IK data."""
    start_iks: torch.Tensor  # [N, DoF]
    goal_iks: torch.Tensor   # [M, DoF]
    
    @property
    def num_start_iks(self) -> int:
        return self.start_iks.shape[0]
    
    @property
    def num_goal_iks(self) -> int:
        return self.goal_iks.shape[0]
    
    @property
    def num_dof(self) -> int:
        return self.start_iks.shape[1]
    
    def to_numpy(self) -> 'IKData':
        """Convert all tensors to numpy arrays."""
        return IKData(
            start_iks=self.start_iks.cpu().numpy(),
            goal_iks=self.goal_iks.cpu().numpy()
        )


def load_trajectory_data(file_path: Path, verbose: bool = True) -> Optional[TrajectoryData]:
    """
    Load trajectory data from a .pt file.
    
    Args:
        file_path: Path to the trajectory data file
        verbose: Whether to print loading information
        
    Returns:
        TrajectoryData object or None if loading fails
    """
    if not file_path.exists():
        if verbose:
            warnings.warn(f"Trajectory file not found: {file_path}")
        return None
    
    try:
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        # Validate required keys
        required_keys = ['start_state', 'goal_state', 'traj', 'success']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
        
        # Reverse object angle (dimension -1) in trajectory data to match IK convention
        # The object angle uses reversed kinematic chain convention in trajectories
        # data['traj'][:, -1] = -data['traj'][:, -1]
        
        traj_data = TrajectoryData(
            start_state=data['start_state'],
            goal_state=data['goal_state'],
            traj=data['traj'],
            success=data['success']
        )
        
        if verbose:
            print(f"✓ Loaded trajectory data from {file_path.name}")
            print(f"  Trajectories: {traj_data.num_trajectories}")
            print(f"  Timesteps: {traj_data.num_timesteps}")
            print(f"  DoF: {traj_data.num_dof}")
            print(f"  Success rate: {traj_data.success_rate:.2%}")
        
        return traj_data
        
    except Exception as e:
        if verbose:
            warnings.warn(f"Error loading trajectory data from {file_path}: {e}")
        return None


def load_ik_data(file_path: Path, verbose: bool = True) -> Optional[IKData]:
    """
    Load IK data from a .pt file.
    
    Args:
        file_path: Path to the IK data file
        verbose: Whether to print loading information
        
    Returns:
        IKData object or None if loading fails
    """
    if not file_path.exists():
        if verbose:
            warnings.warn(f"IK file not found: {file_path}")
        return None
    
    try:
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        # Validate required keys
        required_keys = ['start_iks', 'goal_iks']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
        
        ik_data = IKData(
            start_iks=data['start_iks'],
            goal_iks=data['goal_iks']
        )
        
        if verbose:
            print(f"✓ Loaded IK data from {file_path.name}")
            print(f"  Start IKs: {ik_data.num_start_iks}")
            print(f"  Goal IKs: {ik_data.num_goal_iks}")
            print(f"  DoF: {ik_data.num_dof}")
        
        return ik_data
        
    except Exception as e:
        if verbose:
            warnings.warn(f"Error loading IK data from {file_path}: {e}")
        return None


def load_all_trajectory_data(
    robot_name: str,
    scene_names: List[str],
    object_names: List[str],
    grasp_ids: List[int],
    data_root: Path,
    filtered: bool = False,
    verbose: bool = True
) -> Dict[str, Dict[int, TrajectoryData]]:
    """
    Load all trajectory data for multiple scenes and grasps.
    
    Args:
        robot_name: Name of the robot
        scene_names: List of scene names
        object_names: List of object names (one per scene)
        grasp_ids: List of grasp IDs
        data_root: Root directory containing trajectory data
        filtered: Whether to load filtered or raw trajectory data
        verbose: Whether to print loading progress
        
    Returns:
        Nested dictionary: {scene_name: {grasp_id: TrajectoryData}}
    """
    all_data = {}
    total_files = len(scene_names) * len(grasp_ids)
    loaded_files = 0
    
    if verbose:
        print(f"\nLoading {'filtered' if filtered else 'raw'} trajectory data...")
        print(f"Scenes: {len(scene_names)}, Grasps: {len(grasp_ids)}")
        print(f"Total files to load: {total_files}\n")
    
    for scene_name, object_name in zip(scene_names, object_names):
        scene_data = {}
        
        for grasp_id in grasp_ids:
            filename = "filtered_traj_data.pt" if filtered else "traj_data.pt"
            file_path = (data_root / robot_name / scene_name / object_name / 
                        f"grasp_{grasp_id:04d}" / filename)
            
            traj_data = load_trajectory_data(file_path, verbose=False)
            if traj_data is not None:
                scene_data[grasp_id] = traj_data
                loaded_files += 1
        
        if scene_data:
            all_data[scene_name] = scene_data
        
        if verbose and len(scene_data) > 0:
            print(f"✓ {scene_name}: {len(scene_data)}/{len(grasp_ids)} grasps loaded")
    
    if verbose:
        print(f"\n✓ Total: {loaded_files}/{total_files} files loaded successfully")
    
    return all_data


def load_all_ik_data(
    robot_name: str,
    scene_names: List[str],
    object_names: List[str],
    grasp_ids: List[int],
    data_root: Path,
    verbose: bool = True
) -> Dict[str, Dict[int, IKData]]:
    """
    Load all IK data for multiple scenes and grasps.
    
    Args:
        robot_name: Name of the robot
        scene_names: List of scene names
        object_names: List of object names (one per scene)
        grasp_ids: List of grasp IDs
        data_root: Root directory containing IK data
        verbose: Whether to print loading progress
        
    Returns:
        Nested dictionary: {scene_name: {grasp_id: IKData}}
    """
    all_data = {}
    total_files = len(scene_names) * len(grasp_ids)
    loaded_files = 0
    
    if verbose:
        print(f"\nLoading IK data...")
        print(f"Scenes: {len(scene_names)}, Grasps: {len(grasp_ids)}")
        print(f"Total files to load: {total_files}\n")
    
    for scene_name, object_name in zip(scene_names, object_names):
        scene_data = {}
        
        for grasp_id in grasp_ids:
            file_path = (data_root / robot_name / scene_name / object_name / 
                        f"grasp_{grasp_id:04d}" / "ik_data.pt")
            
            ik_data = load_ik_data(file_path, verbose=False)
            if ik_data is not None:
                scene_data[grasp_id] = ik_data
                loaded_files += 1
        
        if scene_data:
            all_data[scene_name] = scene_data
        
        if verbose and len(scene_data) > 0:
            print(f"✓ {scene_name}: {len(scene_data)}/{len(grasp_ids)} grasps loaded")
    
    if verbose:
        print(f"\n✓ Total: {loaded_files}/{total_files} files loaded successfully")
    
    return all_data


def extract_ik_from_trajectories(traj_data: TrajectoryData) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract start and goal IK solutions from trajectory data.
    
    Note: The object angle (dimension 10, last DoF) is reversed in the trajectory data
    due to the VKC (reversed kinematic chain). We reverse it back here to match the
    IK data convention.
    
    Args:
        traj_data: TrajectoryData object
        
    Returns:
        Tuple of (start_iks, goal_iks) as numpy arrays with corrected object angles
    """
    # First timestep is start IK, last timestep is goal IK
    start_iks = traj_data.traj[:, 0, :].cpu().numpy()
    goal_iks = traj_data.traj[:, -1, :].cpu().numpy()
    
    # Reverse object angle (dimension 10) to match IK data convention
    # The trajectory data stores object angles with reversed sign due to VKC
    start_iks[:, 10] = -start_iks[:, 10]
    goal_iks[:, 10] = -goal_iks[:, 10]
    
    return start_iks, goal_iks


def filter_successful_trajectories(traj_data: TrajectoryData) -> TrajectoryData:
    """
    Filter trajectory data to keep only successful trajectories.
    
    Args:
        traj_data: TrajectoryData object
        
    Returns:
        New TrajectoryData object containing only successful trajectories
    """
    success_mask = traj_data.success.bool()
    
    return TrajectoryData(
        start_state=traj_data.start_state[success_mask],
        goal_state=traj_data.goal_state[success_mask],
        traj=traj_data.traj[success_mask],
        success=traj_data.success[success_mask]
    )


if __name__ == "__main__":
    """Test data loading functionality."""
    
    print("=" * 80)
    print("Testing Data Loader")
    print("=" * 80)
    
    # Test single file loading
    print("\n1. Testing single trajectory file loading...")
    traj_path = get_traj_data_path(ROBOT_NAME, SCENE_NAMES[0], OBJECT_NAMES[0], 
                                   GRASP_IDS[0], filtered=False)
    traj_data = load_trajectory_data(traj_path)
    
    if traj_data:
        print(f"\n   Start IKs extracted: {extract_ik_from_trajectories(traj_data)[0].shape}")
        successful = filter_successful_trajectories(traj_data)
        print(f"   After filtering: {successful.num_trajectories} successful trajectories")
    
    # Test single IK file loading
    print("\n2. Testing single IK file loading...")
    ik_path = get_ik_data_path(ROBOT_NAME, SCENE_NAMES[0], OBJECT_NAMES[0], GRASP_IDS[0])
    ik_data = load_ik_data(ik_path)
    
    # Test batch loading (limited for testing)
    print("\n3. Testing batch loading (first 2 scenes, first 2 grasps)...")
    test_scenes = SCENE_NAMES[:2]
    test_objects = OBJECT_NAMES[:2]
    test_grasps = GRASP_IDS[:2]
    
    all_traj = load_all_trajectory_data(
        ROBOT_NAME, test_scenes, test_objects, test_grasps, 
        DATA_ROOT, filtered=False, verbose=True
    )
    
    all_ik = load_all_ik_data(
        ROBOT_NAME, test_scenes, test_objects, test_grasps,
        DATA_ROOT, verbose=True
    )
    
    print("\n" + "=" * 80)
    print("✓ Data loader test completed successfully")
    print("=" * 80)
