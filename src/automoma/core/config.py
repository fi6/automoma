"""Configuration dataclasses for AutoMoMa framework."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


@dataclass
class SceneConfig:
    """Scene configuration."""
    path: str = ""
    pose: List[float] = field(default_factory=lambda: [0, 0, 0, 1, 0, 0, 0])
    metadata_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "pose": self.pose,
            "metadata_path": self.metadata_path,
        }


@dataclass
class ObjectConfig:
    """Object configuration."""
    path: str = ""
    asset_type: str = ""
    asset_id: str = ""
    pose: List[float] = field(default_factory=lambda: [0, 0, 0, 1, 0, 0, 0])
    dimensions: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    scale: float = 1.0
    joint_id: int = 0
    start_angles: List[float] = field(default_factory=lambda: [0.0])
    goal_angles: List[float] = field(default_factory=lambda: [1.57])
    grasp_dir: str = ""
    num_grasps: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "asset_type": self.asset_type,
            "asset_id": self.asset_id,
            "pose": self.pose,
            "dimensions": self.dimensions,
            "scale": self.scale,
            "joint_id": self.joint_id,
            "start_angles": self.start_angles,
            "goal_angles": self.goal_angles,
            "grasp_dir": self.grasp_dir,
            "num_grasps": self.num_grasps,
        }


@dataclass
class RobotConfig:
    """Robot configuration."""
    robot_type: str = "summit_franka"
    path: str = ""
    mobile_base_dof: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_type": self.robot_type,
            "path": self.path,
            "mobile_base_dof": self.mobile_base_dof,
        }


@dataclass
class CameraConfig:
    """Camera configuration."""
    name: str = ""
    prim_path: str = ""
    resolution: List[int] = field(default_factory=lambda: [320, 240])
    frequency: int = 30
    focal_length: Optional[float] = None
    pose: List[float] = field(default_factory=lambda: [0, 0, 0, 1, 0, 0, 0])
    pose_type: str = "local"  # "local" or "world"


@dataclass  
class PlanConfig:
    """Planning configuration."""
    voxel_dims: List[float] = field(default_factory=lambda: [5.0, 5.0, 5.0])
    voxel_size: float = 0.02
    expanded_dims: List[float] = field(default_factory=lambda: [1.0, 0.2, 0.2])
    collision_checker_type: str = "VOXEL"
    output_dir: str = "data/output"
    
    # Clustering parameters
    ap_fallback_clusters: int = 30
    ap_clusters_upperbound: int = 80
    ap_clusters_lowerbound: int = 10
    
    # IK planning parameters
    ik_limit: List[int] = field(default_factory=lambda: [50, 50])
    
    # Trajectory planning parameters
    stage_type: str = "MOVE_ARTICULATED"
    batch_size: int = 10
    expand_to_pairs: bool = True
    
    # Filter parameters
    position_tolerance: float = 0.01
    rotation_tolerance: float = 0.05
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "voxel_dims": self.voxel_dims,
            "voxel_size": self.voxel_size,
            "expanded_dims": self.expanded_dims,
            "collision_checker_type": self.collision_checker_type,
            "output_dir": self.output_dir,
            "cluster": {
                "ap_fallback_clusters": self.ap_fallback_clusters,
                "ap_clusters_upperbound": self.ap_clusters_upperbound,
                "ap_clusters_lowerbound": self.ap_clusters_lowerbound,
            },
            "plan_ik": {"limit": self.ik_limit},
            "plan_traj": {
                "stage_type": self.stage_type,
                "batch_size": self.batch_size,
                "expand_to_pairs": self.expand_to_pairs,
            },
            "filter": {
                "stage_type": self.stage_type,
                "position_tolerance": self.position_tolerance,
                "rotation_tolerance": self.rotation_tolerance,
            },
        }


@dataclass
class RecordConfig:
    """Recording configuration."""
    output_dir: str = "data/datasets"
    repo_id: str = "automoma_dataset"
    root: str = "./datasets"
    fps: int = 15
    use_videos: bool = True
    push_to_hub: bool = False
    robot_type: str = "summit_franka"
    
    # State configuration
    state_dim: int = 12
    state_names: List[str] = field(default_factory=list)
    
    # Camera configuration
    camera_names: List[str] = field(default_factory=lambda: ["ego_topdown", "ego_wrist", "fix_local"])
    camera_width: int = 320
    camera_height: int = 240
    
    # Recording parameters
    max_episodes: int = 100
    steps_per_episode: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "repo_id": self.repo_id,
            "root": self.root,
            "fps": self.fps,
            "use_videos": self.use_videos,
            "push_to_hub": self.push_to_hub,
            "robot_type": self.robot_type,
            "state_dim": self.state_dim,
            "state_names": self.state_names,
            "camera": {
                "names": self.camera_names,
                "width": self.camera_width,
                "height": self.camera_height,
            },
            "max_episodes": self.max_episodes,
            "steps_per_episode": self.steps_per_episode,
        }


@dataclass
class TrainConfig:
    """Training configuration."""
    # Dataset
    dataset_repo_id: str = ""
    dataset_root: str = "./datasets"
    
    # Model
    policy_type: str = "diffusion"  # "diffusion", "act", "vq_bet"
    pretrained_path: Optional[str] = None
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 500
    grad_clip: float = 10.0
    
    # Checkpointing
    output_dir: str = "outputs/train"
    save_freq: int = 10000
    eval_freq: int = 5000
    log_freq: int = 100
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "automoma"
    wandb_entity: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": {
                "repo_id": self.dataset_repo_id,
                "root": self.dataset_root,
            },
            "policy": {
                "type": self.policy_type,
                "pretrained_path": self.pretrained_path,
            },
            "training": {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "warmup_steps": self.warmup_steps,
                "grad_clip": self.grad_clip,
            },
            "checkpoint": {
                "output_dir": self.output_dir,
                "save_freq": self.save_freq,
                "eval_freq": self.eval_freq,
                "log_freq": self.log_freq,
            },
            "hardware": {
                "device": self.device,
                "num_workers": self.num_workers,
            },
            "wandb": {
                "use_wandb": self.use_wandb,
                "project": self.wandb_project,
                "entity": self.wandb_entity,
            },
        }


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # Model
    checkpoint_path: str = ""
    policy_type: str = "diffusion"
    
    # Environment
    scene_config: Optional[SceneConfig] = None
    object_config: Optional[ObjectConfig] = None
    robot_config: Optional[RobotConfig] = None
    
    # Evaluation parameters
    num_episodes: int = 50
    max_steps_per_episode: int = 500
    success_threshold: float = 0.05
    
    # Async inference settings (for LeRobot)
    use_async_inference: bool = True
    inference_host: str = "localhost"
    inference_port: int = 50051
    inference_timeout: float = 30.0
    
    # Output
    output_dir: str = "outputs/eval"
    save_videos: bool = True
    save_trajectories: bool = True
    
    # Hardware
    device: str = "cuda"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": {
                "checkpoint_path": self.checkpoint_path,
                "policy_type": self.policy_type,
            },
            "evaluation": {
                "num_episodes": self.num_episodes,
                "max_steps_per_episode": self.max_steps_per_episode,
                "success_threshold": self.success_threshold,
            },
            "async_inference": {
                "enabled": self.use_async_inference,
                "host": self.inference_host,
                "port": self.inference_port,
                "timeout": self.inference_timeout,
            },
            "output": {
                "output_dir": self.output_dir,
                "save_videos": self.save_videos,
                "save_trajectories": self.save_trajectories,
            },
            "hardware": {
                "device": self.device,
            },
        }


def load_config_from_yaml(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def save_config_to_yaml(config: Dict[str, Any], yaml_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
