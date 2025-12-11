#!/usr/bin/env python3
"""
Simple trajectory re-filtering script

Processes all scene directories under a given path and re-filters trajectory files
with tighter tolerances.
"""

import os
import glob
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Union

# Third Party
from curobo.util_file import load_yaml
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.geom.types import WorldConfig

# Local imports
from automoma.utils.file import process_robot_cfg
from cuakr.utils.math import quaternion_distance


@dataclass
class TrajResult:
    """Simple trajectory result container"""
    start_states: torch.Tensor
    goal_states: torch.Tensor
    trajectories: torch.Tensor
    success: torch.Tensor


def load_akr_robot_config(akr_robot_cfg_path: str) -> Dict:
    """Load and process AKR robot configuration"""
    akr_robot_cfg = load_yaml(akr_robot_cfg_path)["robot_cfg"]
    akr_robot_cfg = process_robot_cfg(akr_robot_cfg)
    return akr_robot_cfg


def init_motion_gen_akr(akr_robot_cfg: Dict) -> MotionGen:
    """Initialize AKR motion generator"""
    tensor_args = TensorDeviceType()
    
    motion_gen_config_akr = MotionGenConfig.load_from_robot_config(
        akr_robot_cfg,
        WorldConfig(),
        tensor_args,
        world_coll_checker=None,  # No collision checking for filtering
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=0.05,
        collision_cache={"obb": 30, "mesh": 100},
        optimize_dt=True,
        trajopt_dt=None,
        trajopt_tsteps=32,
        trim_steps=None,
        use_cuda_graph=False,
        gradient_trajopt_file="gradient_trajopt_fixbase.yml",
    )
    
    motion_gen_akr = MotionGen(motion_gen_config_akr)
    return motion_gen_akr


def load_traj(path: str) -> TrajResult:
    """Load trajectory data from file"""
    traj_data = torch.load(path, weights_only=False)
    start_state = traj_data["start_state"]
    goal_state = traj_data["goal_state"]
    traj = traj_data["traj"]
    success = traj_data["success"]
    return TrajResult(
        start_states=start_state,
        goal_states=goal_state,
        trajectories=traj,
        success=success
    )


def save_traj(traj_result: TrajResult, path: str) -> None:
    """Save trajectory data to file"""
    traj_data = {
        "start_state": traj_result.start_states.cpu(),
        "goal_state": traj_result.goal_states.cpu(),
        "traj": traj_result.trajectories.cpu(),
        "success": traj_result.success.cpu(),
    }
    torch.save(traj_data, path)


def akr_fk_filter(js: JointState, pose, motion_gen_akr: MotionGen,
                  position_tolerance: float = 0.005, 
                  rotation_tolerance: float = 0.005) -> bool:
    """
    Check if FK of joint state is within specified pose tolerance
    """
    fk_result = motion_gen_akr.ik_solver.fk(js.position)
    
    # Convert to CPU numpy arrays
    pose_position_cpu = (
        pose.position.cpu().numpy()
        if pose.position.is_cuda
        else pose.position.numpy()
    )
    pose_quaternion_cpu = (
        pose.quaternion.cpu().numpy()
        if pose.quaternion.is_cuda
        else pose.quaternion.numpy()
    )
    fk_position_cpu = (
        fk_result.ee_pose.position.cpu().numpy()
        if fk_result.ee_pose.position.is_cuda
        else fk_result.ee_pose.position.numpy()
    )
    fk_quaternion_cpu = (
        fk_result.ee_pose.quaternion.cpu().numpy()
        if fk_result.ee_pose.quaternion.is_cuda
        else fk_result.ee_pose.quaternion.numpy()
    )
    
    # Check position and rotation differences
    position_diff = np.linalg.norm(pose_position_cpu - fk_position_cpu)
    rotation_diff = quaternion_distance(pose_quaternion_cpu, fk_quaternion_cpu)
    
    return position_diff < position_tolerance and rotation_diff < rotation_tolerance


def filter_trajectories(traj_result: TrajResult, motion_gen_akr: MotionGen,
                       position_tolerance: float = 0.005,
                       rotation_tolerance: float = 0.005) -> TrajResult:
    """
    Filter trajectory data based on FK validation with tighter tolerances
    """
    tensor_args = TensorDeviceType()
    
    if traj_result.trajectories.shape[0] == 0:
        return traj_result
    
    # Step 1: Filter based on success
    success_indices = traj_result.success.nonzero(as_tuple=True)[0]
    
    if success_indices.shape[0] == 0:
        # Return empty result
        dof = traj_result.start_states.shape[1]
        return TrajResult(
            start_states=torch.zeros((0, dof)),
            goal_states=torch.zeros((0, dof)),
            trajectories=torch.zeros((0, 0, dof)),
            success=torch.zeros((0,), dtype=torch.bool)
        )
    
    # Filter by success
    start_states = traj_result.start_states[success_indices]
    goal_states = traj_result.goal_states[success_indices]
    trajectories = traj_result.trajectories[success_indices]
    success_flags = traj_result.success[success_indices]
    
    # Step 2: FK filtering with tighter tolerances
    fk_valid_indices = []
    
    for i in tqdm(range(goal_states.shape[0]), desc="FK Filtering"):
        goal_pose = JointState.from_position(tensor_args.to_device(goal_states[i]))
        
        # Get target object pose from goal
        goal_object_pose = motion_gen_akr.ik_solver.fk(goal_pose.position).ee_pose
        
        # Check each waypoint in trajectory
        trajectory_valid = True
        for j in range(trajectories.shape[1]):
            traj_pose = JointState.from_position(
                tensor_args.to_device(trajectories[i][j])
            )
            if not akr_fk_filter(traj_pose, goal_object_pose, motion_gen_akr,
                                position_tolerance, rotation_tolerance):
                trajectory_valid = False
                break
        
        if trajectory_valid:
            fk_valid_indices.append(i)
    
    if not fk_valid_indices:
        # Return empty result
        dof = start_states.shape[1]
        return TrajResult(
            start_states=torch.zeros((0, dof)),
            goal_states=torch.zeros((0, dof)),
            trajectories=torch.zeros((0, 0, dof)),
            success=torch.zeros((0,), dtype=torch.bool)
        )
    
    # Apply FK filtering
    fk_indices = torch.tensor(fk_valid_indices, device=start_states.device)
    start_states = start_states[fk_indices]
    goal_states = goal_states[fk_indices]
    trajectories = trajectories[fk_indices]
    success_flags = success_flags[fk_indices]
    
    return TrajResult(
        start_states=start_states,
        goal_states=goal_states,
        trajectories=trajectories,
        success=success_flags
    )


def process_scene_directory(scene_dir: str, motion_gen_akr: MotionGen,
                          position_tolerance: float = 0.005,
                          rotation_tolerance: float = 0.005) -> None:
    """Process all trajectory files in a scene directory"""
    
    # Find all filtered_traj_data.pt files
    traj_files = glob.glob(os.path.join(scene_dir, "**/filtered_traj_data.pt"), recursive=True)
    
    for traj_file in traj_files:
        print(f"\nProcessing: {traj_file}")
        
        try:
            # Load existing trajectory data
            traj_result = load_traj(traj_file)
            original_count = traj_result.success.sum().item()
            total_count = len(traj_result.success)
            
            print(f"  Original: {original_count}/{total_count} successful trajectories")
            
            # Re-filter with tighter tolerances
            filtered_result = filter_trajectories(
                traj_result, motion_gen_akr, 
                position_tolerance, rotation_tolerance
            )
            
            filtered_count = filtered_result.success.sum().item()
            filtered_total = len(filtered_result.success)
            
            print(f"  After re-filtering: {filtered_count}/{filtered_total} successful trajectories")
            print(f"  Reduction: {original_count - filtered_count} trajectories")
            
            # Save back to same file
            save_traj(filtered_result, traj_file)
            print(f"  ✓ Saved to {traj_file}")
            
        except Exception as e:
            print(f"  ✗ Error processing {traj_file}: {e}")


def main():
    """Main function to re-filter all trajectory files"""
    
    # Configuration
    base_path = "output/summit_franka"
    akr_robot_cfg_path = "assets/object/Microwave/7221/summit_franka_7221_0_grasp_0000.yml"
    
    # Tighter tolerances than before
    position_tolerance = 0.005  # 5mm instead of 10mm
    rotation_tolerance = 0.005  # ~0.3 degrees instead of ~0.6 degrees
    
    print("=== Trajectory Re-filtering Script ===")
    print(f"Base path: {base_path}")
    print(f"Position tolerance: {position_tolerance}m")
    print(f"Rotation tolerance: {rotation_tolerance} rad")
    
    # Load AKR robot configuration
    print("\nLoading AKR robot configuration...")
    akr_robot_cfg = load_akr_robot_config(akr_robot_cfg_path)
    
    # Initialize motion generator
    print("Initializing AKR motion generator...")
    motion_gen_akr = init_motion_gen_akr(akr_robot_cfg)
    
    # Find all scene directories
    scene_dirs = [d for d in glob.glob(os.path.join(base_path, "scene_*")) 
                  if os.path.isdir(d)]
    
    print(f"\nFound {len(scene_dirs)} scene directories")
    
    # Process each scene directory
    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        print(f"\n{'='*50}")
        print(f"Processing scene: {scene_name}")
        print(f"{'='*50}")
        
        process_scene_directory(scene_dir, motion_gen_akr, 
                              position_tolerance, rotation_tolerance)
    
    print(f"\n{'='*50}")
    print("✓ Re-filtering completed for all scenes!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()