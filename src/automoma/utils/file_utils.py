import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from curobo.util_file import load_yaml
from automoma.core.types import IKResult, TrajResult
from scipy.spatial.transform import Rotation as R


def get_project_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def get_abs_path(path: str) -> str:
    return os.path.join(get_project_dir(), path)

def process_robot_cfg(robot_cfg: Dict) -> Dict:
    if robot_cfg.get("kinematics", {}).get("urdf_path", ""):
        robot_cfg["kinematics"]["urdf_path"] = os.path.join(get_project_dir(), robot_cfg["kinematics"]["urdf_path"])
    if robot_cfg.get("kinematics", {}).get("external_asset_path", ""):
        robot_cfg["kinematics"]["external_asset_path"] = os.path.join(
            get_project_dir(), robot_cfg["kinematics"]["external_asset_path"]
        )
    if robot_cfg.get("kinematics", {}).get("asset_root_path", ""):
        robot_cfg["kinematics"]["asset_root_path"] = os.path.join(
            get_project_dir(), robot_cfg["kinematics"]["asset_root_path"]
        )
    return robot_cfg
def load_robot_cfg(robot_cfg_path: Union[str, Dict]) -> Dict[str, Any]:
    """Load robot configuration from YAML file or return dict."""
    if isinstance(robot_cfg_path, str):
        robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]
        print(f"Robot configuration loaded from {robot_cfg_path}")
    else:
        robot_cfg = robot_cfg_path
    return robot_cfg

def save_ik(ik_result: IKResult, path: str) -> None:
    """Save IKResult to a file."""
    ik_data = {
        "target_poses": ik_result.target_poses,
        "iks": ik_result.iks,
    }
    torch.save(ik_data, path)
    print(f"IK data saved to {path}")

def load_ik(path: str) -> IKResult:
    """Load IKResult from a file."""
    ik_data = torch.load(path, weights_only=False)
    print(f"IK data loaded from {path}")
    return IKResult(
        start_iks=ik_data["start_iks"],
        goal_iks=ik_data.get("goal_iks")
    )

def save_traj(traj_result: TrajResult, path: str) -> None:
    """Save TrajResult to a file."""
    traj_data = {
        "start_states": traj_result.start_states.cpu(),
        "goal_states": traj_result.goal_states.cpu(),
        "trajectories": traj_result.trajectories.cpu(),
        "success": traj_result.success.cpu()
    }
    torch.save(traj_data, path)
    print(f"Trajectory data saved to {path}")

def load_traj(path: str) -> TrajResult:
    """Load TrajResult from a file."""
    traj_data = torch.load(path, weights_only=False)
    print(f"Trajectory data loaded from {path}")
    return TrajResult.from_dict(traj_data)

def get_grasp_poses(
    grasp_dir: str, 
    num_grasps: int = 10, 
    scaling_factor: float = 1.0
) -> List[np.ndarray]:
    """Load and scale grasp poses from directory."""
    def grasp_scale(grasp: np.ndarray, scale: float) -> np.ndarray:
        scaled_grasp = np.copy(grasp)
        scaled_grasp[:3] *= scale
        return scaled_grasp
    
    grasp_poses = []
    for i in range(num_grasps):
        grasp_file = os.path.join(grasp_dir, f"{i:04d}.npy")
        if os.path.exists(grasp_file):
            grasp = np.load(grasp_file)
            grasp_poses.append(grasp_scale(grasp, scaling_factor))
    
    print(f"Loaded {len(grasp_poses)} grasp poses from {grasp_dir}")
    return grasp_poses

def load_object_from_metadata(metadata_path: str, object_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load object configuration from scene metadata."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    objects = metadata["static_objects"]
    
    # Find target object
    object_info = None
    target_asset_type = object_cfg.get("asset_type")
    target_asset_id = object_cfg.get("asset_id")
    
    for value in objects.values():
        if (value.get("asset_type") == target_asset_type and 
            value.get("asset_id") == target_asset_id):
            object_info = value
            break
    
    if object_info is None:
        raise ValueError(f"Object with asset_type={target_asset_type} and asset_id={target_asset_id} not found in metadata")

    from automoma.utils.math_utils import matrix_to_pose, single_axis_self_rotation
    
    # Process pose from matrix
    matrix = np.array(object_info["matrix"])
    
    # Apply rotation: single_axis_self_rotation
    matrix = single_axis_self_rotation(matrix, axis='z', angle=np.pi)
    
    # Convert matrix to 7D pose [x, y, z, qw, qx, qy, qz]
    processed_pose = matrix_to_pose(matrix).tolist()
    
    # Update object configuration
    object_cfg.update({
        "name": object_info["name"],
        "asset_type": object_info.get("asset_type", target_asset_type),
        "asset_id": object_info.get("asset_id", target_asset_id),
        "dimensions": object_info["dimensions"],
        "pose": processed_pose
    })
    
    print(f"Object loaded from metadata: {object_info['name']} with dimensions {object_info['dimensions']}")
    return object_cfg
