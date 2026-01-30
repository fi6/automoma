#!/usr/bin/env python3
import os
import sys
import argparse
import torch
from typing import List
from pathlib import Path

# Add src and third_party to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))
sys.path.insert(0, os.path.join(root_dir, "third_party", "cuakr", "src"))

from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.types.robot import JointState
from curobo.types.base import TensorDeviceType
from curobo.geom.types import WorldConfig
from curobo.util_file import load_yaml

from automoma.utils.file import process_robot_cfg


# Robot configuration mapping
ROBOT_CONFIG_MAP = {
    "summit_franka": "assets/robot/summit_franka/summit_franka.yml",
    "summit_franka_fixed_base": "assets/robot/summit_franka/summit_franka_fixed_base.yml",
    "r1": "assets/robot/r1/r1.yml",
}

def load_traj(path):
    traj_data = torch.load(path, weights_only=False)
    start_state = traj_data["start_state"]
    goal_state = traj_data["goal_state"]
    traj = traj_data["traj"]
    success = traj_data["success"]
    return start_state, goal_state, traj, success
def save_traj(path, goal_state, start_state, traj, success):
    torch.save(
        {
            "goal_state": goal_state.cpu(),
            "start_state": start_state.cpu(),
            "traj": traj.cpu(),
            "success": success.cpu(),
        },
        path,
    )
    print(f"[INFO] Saved interpolated trajectory to: {path}")

class TrajectoryInterpolator:
    def __init__(self, robot_cfg):
        self.tensor_args = TensorDeviceType()
        
        # Initialize MotionGen
        # We use a minimal configuration for interpolation
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            WorldConfig(),
            self.tensor_args,
            interpolation_dt=0.01,
            use_cuda_graph=False,
        )
        self.motion_gen = MotionGen(motion_gen_config)

    def interpolate_traj(self, input_path, output_path):
        """
        Interpolate trajectory from input path and save to output path.
        """
        if not os.path.exists(input_path):
            print(f"[WARN] Input file does not exist: {input_path}")
            return False

        # Check if the trajectory folder already exists
        output_folder = os.path.dirname(output_path)
        os.makedirs(output_folder, exist_ok=True)

        # Load the trajectory data
        start_state, goal_state, traj, success = load_traj(input_path)
        
        # Interpolate the trajectory
        # self.motion_gen.js_trajopt_solver.interpolation_dt = 0.01
        self.motion_gen.js_trajopt_solver.interpolation_dt = 0.002
        
        interpolated_result = (
            self.motion_gen.js_trajopt_solver.get_interpolated_trajectory(
                JointState.from_position(
                    self.tensor_args.to_device(traj)
                )
            )
        )
        
        L = interpolated_result[1].min().item()  # in case there's variation
        interpolated_trajectory = interpolated_result[0][:, :L, :]
        interpolated_trajectory = interpolated_trajectory.position

        print(f"[INFO] Interpolated {input_path}")
        print(f"       Shape changed from {traj.shape} to {interpolated_trajectory.shape}")

        # Save the interpolated trajectory
        save_traj(
            output_path,
            goal_state,
            start_state,
            interpolated_trajectory,
            success
        )
        return True

def get_robot_config_path(robot_name: str) -> str:
    """Get the configuration path for a robot by name."""
    if robot_name not in ROBOT_CONFIG_MAP:
        # Try to find it in assets/robot/{robot_name}/{robot_name}.yml
        potential_path = f"assets/robot/{robot_name}/{robot_name}.yml"
        if os.path.exists(os.path.join(root_dir, potential_path)):
            return potential_path
        raise ValueError(f"Unknown robot: {robot_name}. Available robots: {list(ROBOT_CONFIG_MAP.keys())}")
    return ROBOT_CONFIG_MAP[robot_name]

def get_akr_robot_config_path(object_category: str, object_id: str, robot_name: str, grasp_id: int) -> str:
    """Get the configuration path for a robot by name."""
    # "assets/object/Microwave/7221/summit_franka_7221_0_grasp_0000.yml"
    return f"assets/object/{object_category}/{object_id}/{robot_name}_{object_id}_0_grasp_{grasp_id:04d}.yml"

def main():
    parser = argparse.ArgumentParser(description="Interpolate trajectories in batch")
    parser.add_argument("--dir", type=str, default="output/collect_table1/traj", help="Base directory containing trajectories")
    parser.add_argument("--robot", type=str, default="summit_franka", help="Robot name")
    parser.add_argument("--scene", nargs='+', default=[f"scene_{i}_seed_{i}" for i in range(0, 1)], help="List of scenes")
    parser.add_argument("--object", nargs='+', default=["7221"], help="List of objects")
    parser.add_argument("--grasp_id", nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 18], help="List of grasp IDs (integers or strings)")
    
    args = parser.parse_args()
    
    # Process batch
    total = 0
    success_count = 0
    
    for scene in args.scene:
        for obj in args.object:
            for grasp in args.grasp_id:
                # Handle grasp ID formatting
                grasp_int = int(grasp)
                grasp_str = f"grasp_{grasp_int:04d}"
                
                # Load robot config for this specific grasp
                robot_config_path = get_akr_robot_config_path("Microwave", obj, args.robot, grasp_int)
                abs_robot_config_path = os.path.join(root_dir, robot_config_path)
                
                if not os.path.exists(abs_robot_config_path):
                    print(f"[WARN] Robot config does not exist: {abs_robot_config_path}")
                    total += 1
                    continue
                
                print(f"Loading robot config from: {abs_robot_config_path}")
                robot_cfg = load_yaml(abs_robot_config_path)["robot_cfg"]
                robot_cfg = process_robot_cfg(robot_cfg)
                
                # Initialize interpolator for this grasp
                interpolator = TrajectoryInterpolator(robot_cfg)
                
                # Construct paths
                # Path structure: {dir}/{robot}/{scene}/{object}/{grasp_id}/filtered_traj_data.pt
                
                base_path = os.path.join(args.dir, args.robot, scene, obj, grasp_str)
                input_file = os.path.join(base_path, "filtered_traj_data.pt")
                output_file = os.path.join(base_path, "filtered_traj_data_interpolated.pt")
                
                print(f"Processing: {input_file}")
                if interpolator.interpolate_traj(input_file, output_file):
                    success_count += 1
                total += 1
                
    print(f"Batch processing complete. Success: {success_count}/{total}")

if __name__ == "__main__":
    main()
