"""
Data structures for camera and trajectory recording in the replay system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import torch
import numpy as np


@dataclass
class CameraResult:
    """
    Dataclass to store camera data and trajectory information in the same format 
    as collect_data.py for consistency with RoboTwin data collection pipeline.
    """
    
    # Environment information
    env_info: Dict[str, Any] = field(default_factory=dict)
    
    # Observation data following collect_data.py structure
    obs: Dict[str, Any] = field(default_factory=lambda: {
        "joint": {},      # Joint state data
        "eef": [],        # End-effector pose data  
        "point_cloud": [], # Point cloud observations
        "rgb": {},        # RGB camera images
        "depth": {}       # Depth camera images
    })
    
    def initialize_joint_structure(self, robot_name: str) -> None:
        """Initialize joint data structure based on robot configuration from collect_data.py format."""
        # Define joint configs locally to avoid import issues
        joint_configs = {
            "summit_franka": {
                "output_joints": {
                    "mobile_base": 3,
                    "arm": 7,
                    "gripper": 1,
                }
            },
            "summit_franka_fixed_base": {
                "output_joints": {
                    "mobile_base": 3,
                    "arm": 7,
                    "gripper": 1,
                }
            },
            "r1": {
                "output_joints": {
                    "mobile_base": 3,
                    "torso": 4,
                    "left_arm": 6,
                    "left_gripper": 1,
                    "right_arm": 6,
                    "right_gripper": 1,
                }
            }
        }
        
        if robot_name in joint_configs:
            joint_config = joint_configs[robot_name]
            output_joints = joint_config["output_joints"]
            
            # Initialize joint structure with grouped data
            for joint_group_name in output_joints.keys():
                self.obs["joint"][joint_group_name] = []
        else:
            # Fallback for unknown robot types - use individual joint names
            joint_names = self._get_fallback_joint_names(robot_name)
            for joint_name in joint_names:
                self.obs["joint"][joint_name] = []
    
    def _get_fallback_joint_names(self, robot_name: str) -> List[str]:
        """Fallback joint names for robots not in CollectConfig."""
        joint_configs = {
            "summit_franka": [
                "base_x", "base_y", "base_z",
                "panda_joint1", "panda_joint2", "panda_joint3", 
                "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",
                "panda_finger_joint1", "panda_finger_joint2"
            ],
            "r1": [
                "base_x", "base_y", "base_z",
                "torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4",
                "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
                "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
                "left_gripper_axis1", "left_gripper_axis2"
            ]
        }
        return joint_configs.get(robot_name, [])
    
    def initialize_camera_structure(self, camera_types: List[str]) -> None:
        """Initialize camera data structures for RGB and depth."""
        for camera_type in camera_types:
            self.obs["rgb"][camera_type] = []
            self.obs["depth"][camera_type] = []
    
    def add_observation(self, 
                       joint_data: Optional[Dict[str, np.ndarray]] = None,
                       eef_data: Optional[np.ndarray] = None,
                       point_cloud_data: Optional[np.ndarray] = None,
                       rgb_data: Optional[Dict[str, np.ndarray]] = None,
                       depth_data: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Add observation data for current timestep."""
        
        if joint_data is not None:
            for joint_name, joint_value in joint_data.items():
                if joint_name in self.obs["joint"]:
                    self.obs["joint"][joint_name].append(joint_value)
        
        if eef_data is not None:
            self.obs["eef"].append(eef_data)
            
        if point_cloud_data is not None:
            self.obs["point_cloud"].append(point_cloud_data)
            
        if rgb_data is not None:
            for camera_type, rgb_image in rgb_data.items():
                if camera_type in self.obs["rgb"]:
                    self.obs["rgb"][camera_type].append(rgb_image)
                    
        if depth_data is not None:
            for camera_type, depth_image in depth_data.items():
                if camera_type in self.obs["depth"]:
                    self.obs["depth"][camera_type].append(depth_image)
    
    def add_grouped_joint_observation(self, raw_joint_positions: np.ndarray, robot_name: str) -> None:
        """Add joint observation data grouped according to robot configuration."""
        # Define joint configs and input joint dimensions locally
        joint_configs = {
            "summit_franka": {
                "input_joints": {"mobile_base": 3, "arm": 7, "gripper": 2},
                "output_joints": {"mobile_base": 3, "arm": 7, "gripper": 1},
            },
            "summit_franka_fixed_base": {
                "input_joints": {"mobile_base": 3, "arm": 7, "gripper": 2},
                "output_joints": {"mobile_base": 3, "arm": 7, "gripper": 1},
            },
            "r1": {
                "input_joints": {"mobile_base": 3, "torso": 4, "left_arm": 6, "left_gripper": 2, "right_arm": 6, "right_gripper": 2},
                "output_joints": {"mobile_base": 3, "torso": 4, "left_arm": 6, "left_gripper": 1, "right_arm": 6, "right_gripper": 1},
            }
        }
        
        if robot_name not in joint_configs:
            # Fallback: add raw joint positions as individual joints
            for i, pos in enumerate(raw_joint_positions):
                joint_name = f"joint_{i}"
                if joint_name not in self.obs["joint"]:
                    self.obs["joint"][joint_name] = []
                self.obs["joint"][joint_name].append(float(pos))
            return
        
        config = joint_configs[robot_name]
        input_joints = config["input_joints"]
        output_joints = config["output_joints"]
        
        # Process joint data according to input/output configuration
        start_idx = 0
        for joint_group_name in output_joints:
            output_dim = output_joints[joint_group_name]
            if joint_group_name in input_joints:
                input_dim = input_joints[joint_group_name]
                # Extract values from raw_joint_positions
                joint_values = raw_joint_positions[start_idx : start_idx + input_dim]
                
                # For gripper, combine 2 input values to 1 output value by averaging
                if "gripper" in joint_group_name and input_dim == 2 and output_dim == 1:
                    joint_values = [(joint_values[0] + joint_values[1]) / 2]
                else:
                    joint_values = joint_values.tolist()
                    
                start_idx += input_dim
            else:
                # For joints not in input, use zero values
                joint_values = [0.0] * output_dim

            # Add to observation - store as array if multi-dimensional, scalar if single
            if output_dim > 1:
                self.obs["joint"][joint_group_name].append(joint_values)
            else:
                self.obs["joint"][joint_group_name].append(joint_values[0])
    
    def set_env_info(self, 
                     scene_id: str,
                     robot_name: str,
                     object_id: str,
                     pose_id: str,
                     **kwargs) -> None:
        """Set environment information matching collect_data.py format."""
        self.env_info = {
            "robot_name": robot_name,
            "scene_id": scene_id,
            "object_id": object_id,
            "grasp_pose": pose_id,  # Using pose_id as grasp_pose to match collect_data.py
        }
    
    def finalize(self) -> None:
        """Finalize the data structure by setting the number of timesteps."""
        # Determine number of timesteps from any non-empty observation
        if self.obs["eef"]:
            self.env_info["num_timesteps"] = len(self.obs["eef"])
        elif any(self.obs["joint"].values()):
            first_joint = next(iter(self.obs["joint"].values()))
            self.env_info["num_timesteps"] = len(first_joint)
        elif self.obs["point_cloud"]:
            self.env_info["num_timesteps"] = len(self.obs["point_cloud"])


@dataclass 
class TrajectoryEvaluationResult:
    """
    Dataclass for storing trajectory evaluation results from policy inference.
    """
    
    # Model predictions vs ground truth
    eef_poses: torch.Tensor  # Shape: (num_steps, 2, 7) - [model, gt], 7D pose
    open_angles: torch.Tensor  # Shape: (num_steps,) - handle open angles
    
    # Evaluation metrics
    success: bool = False
    num_steps: int = 0
    
    # Additional trajectory information
    trajectory_idx: int = 0
    grasp_id: int = 0
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute evaluation metrics from the trajectory data."""
        if self.eef_poses.size(0) == 0:
            return {"position_error": float('inf'), "rotation_error": float('inf')}
        
        # Extract model and ground truth poses
        model_poses = self.eef_poses[:, 0, :]  # Shape: (num_steps, 7)
        gt_poses = self.eef_poses[:, 1, :]     # Shape: (num_steps, 7)
        
        # Position error (L2 norm)
        pos_error = torch.norm(model_poses[:, :3] - gt_poses[:, :3], dim=1).mean().item()
        
        # Rotation error (quaternion distance - simplified)
        # For quaternion q1, q2: error = 1 - |dot(q1, q2)|
        model_quats = model_poses[:, 3:7]  # (num_steps, 4)
        gt_quats = gt_poses[:, 3:7]        # (num_steps, 4)
        
        # Normalize quaternions
        model_quats = model_quats / torch.norm(model_quats, dim=1, keepdim=True)
        gt_quats = gt_quats / torch.norm(gt_quats, dim=1, keepdim=True)
        
        # Dot product and rotation error
        dot_products = torch.sum(model_quats * gt_quats, dim=1)
        rot_error = (1 - torch.abs(dot_products)).mean().item()
        
        return {
            "position_error": pos_error,
            "rotation_error": rot_error,
            "success": self.success,
            "num_steps": self.num_steps
        }