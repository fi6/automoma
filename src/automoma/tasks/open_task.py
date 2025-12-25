"""Open task for articulated objects (doors, drawers, microwaves, etc.)."""

import logging
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np

from automoma.core.types import TaskType, StageType, IKResult
from automoma.core.config_loader import Config
from automoma.tasks.base_task_new import BaseTask, TaskResult, StageResult


logger = logging.getLogger(__name__)


class OpenTask(BaseTask):
    """
    Open task for articulated objects.
    
    This task involves:
    1. Moving to grasp position on handle (at start angle)
    2. Opening the articulated object (from start to goal angle)
    
    The robot starts grasping the handle and moves while the object articulates.
    """
    
    STAGES = [StageType.MOVE_ARTICULATED]
    TASK_TYPE = TaskType.REACH_OPEN
    
    def __init__(self, cfg: Config):
        """Initialize open task."""
        super().__init__(cfg)
        self.name = "open"
    
    def get_grasp_poses(self, object_cfg: Config, **kwargs) -> List[List[float]]:
        """
        Get grasp poses for the object handle.
        
        Args:
            object_cfg: Object configuration
            
        Returns:
            List of grasp poses
        """
        from automoma.utils.file_utils import get_grasp_poses
        
        # Get grasp directory
        grasp_dir = object_cfg.grasp_dir
        if not grasp_dir:
            grasp_dir = f"assets/object/{object_cfg.asset_type}/{object_cfg.asset_id}/grasp"
        
        grasp_dir = str(self.project_root / grasp_dir)
        
        num_grasps = self.cfg.plan_cfg.num_grasps
        scale = object_cfg.scale if object_cfg.scale else 1.0
        
        return get_grasp_poses(
            grasp_dir=grasp_dir,
            num_grasps=num_grasps,
            scaling_factor=scale,
        )
    
    def get_target_pose_for_stage(
        self,
        stage_index: int,
        grasp_pose: List[float],
        angle: float,
        object_cfg: Config,
    ) -> torch.Tensor:
        """
        Get target end-effector pose for opening motion.
        
        The target pose accounts for the object's articulation angle.
        
        Args:
            stage_index: Stage index (always 0 for single-stage open)
            grasp_pose: Base grasp pose on handle
            angle: Current articulation angle
            object_cfg: Object configuration
            
        Returns:
            Target pose tensor [x, y, z, qw, qx, qy, qz]
        """
        from automoma.utils.math_utils import get_open_ee_pose
        from curobo.types.math import Pose
        
        # Get object pose from planner
        object_pose = Pose.from_list(self.planner.object_pose)
        grasp_Pose = Pose.from_list(grasp_pose)
        
        # Default and current joint configuration
        default_joint_cfg = {"joint_0": 0.0}
        joint_cfg = {"joint_0": angle}
        
        # Compute target pose considering articulation
        target_Pose = get_open_ee_pose(
            object_pose=object_pose,
            grasp_pose=grasp_Pose,
            object_urdf=self.planner.object_urdf,
            handle="link_0",
            joint_cfg=joint_cfg,
            default_joint_cfg=default_joint_cfg,
        )
        
        return torch.tensor(target_Pose.to_list())
    
    def get_stage_type(self, stage_index: int) -> StageType:
        """Get stage type - always MOVE_ARTICULATED for open task."""
        return StageType.MOVE_ARTICULATED
    
    def _get_stage_angles(self, stage_index: int, object_cfg: Config) -> Tuple[List[float], List[float]]:
        """
        Get start and goal angles for the open stage.
        
        For the open task:
        - Start angles: Handle at closed position (e.g., 0)
        - Goal angles: Handle at open position (e.g., 1.57)
        """
        start_angles = object_cfg.start_angles if object_cfg.start_angles else [0.0]
        goal_angles = object_cfg.goal_angles if object_cfg.goal_angles else [1.57]
        return start_angles, goal_angles
    
    def _check_task_complete(self, obs: Dict[str, Any]) -> bool:
        """
        Check if the open task is complete.
        
        Task is complete when object angle is close to goal angle.
        """
        if "env_state" not in obs:
            return False
        
        current_angle = obs.get("env_state", 0.0)
        goal_angle = self.cfg.object_cfg.goal_angles[-1] if self.cfg.object_cfg.goal_angles else 1.57
        
        return abs(current_angle - goal_angle) < 0.1  # 0.1 radian tolerance
    
    def get_test_initial_states(self, test_data_dir: str) -> List[torch.Tensor]:
        """
        Load test initial states for open task evaluation.
        
        For open task, the robot should start already grasping the handle
        at the start angle position.
        """
        initial_states = super().get_test_initial_states(test_data_dir)
        
        if not initial_states:
            logger.warning("No test initial states found, using start IKs")
        
        return initial_states


class ReachOpenTask(BaseTask):
    """
    Two-stage reach and open task.
    
    Stage 1: Reach to grasp position
    Stage 2: Open the articulated object
    
    The goal IKs of Stage 1 become the start IKs of Stage 2.
    """
    
    STAGES = [StageType.MOVE, StageType.MOVE_ARTICULATED]
    TASK_TYPE = TaskType.REACH_OPEN
    
    def __init__(self, cfg: Config):
        """Initialize reach-open task."""
        super().__init__(cfg)
        self.name = "reach_open"
    
    def get_grasp_poses(self, object_cfg: Config, **kwargs) -> List[List[float]]:
        """Get grasp poses for the object handle."""
        from automoma.utils.file_utils import get_grasp_poses
        
        grasp_dir = object_cfg.grasp_dir
        if not grasp_dir:
            grasp_dir = f"assets/object/{object_cfg.asset_type}/{object_cfg.asset_id}/grasp"
        
        grasp_dir = str(self.project_root / grasp_dir)
        
        num_grasps = self.cfg.plan_cfg.num_grasps
        scale = object_cfg.scale if object_cfg.scale else 1.0
        
        return get_grasp_poses(
            grasp_dir=grasp_dir,
            num_grasps=num_grasps,
            scaling_factor=scale,
        )
    
    def get_target_pose_for_stage(
        self,
        stage_index: int,
        grasp_pose: List[float],
        angle: float,
        object_cfg: Config,
    ) -> torch.Tensor:
        """
        Get target pose based on stage.
        
        Stage 0 (Reach): Target is the grasp pose at start angle
        Stage 1 (Open): Target is the grasp pose at current angle
        """
        from automoma.utils.math_utils import get_open_ee_pose
        from curobo.types.math import Pose
        
        object_pose = Pose.from_list(self.planner.object_pose)
        grasp_Pose = Pose.from_list(grasp_pose)
        
        default_joint_cfg = {"joint_0": 0.0}
        joint_cfg = {"joint_0": angle}
        
        target_Pose = get_open_ee_pose(
            object_pose=object_pose,
            grasp_pose=grasp_Pose,
            object_urdf=self.planner.object_urdf,
            handle="link_0",
            joint_cfg=joint_cfg,
            default_joint_cfg=default_joint_cfg,
        )
        
        return torch.tensor(target_Pose.to_list())
    
    def get_stage_type(self, stage_index: int) -> StageType:
        """Get stage type based on index."""
        return self.STAGES[stage_index]
    
    def _get_stage_angles(self, stage_index: int, object_cfg: Config) -> Tuple[List[float], List[float]]:
        """
        Get angles for each stage.
        
        Stage 0 (Reach): Start from anywhere, goal is closed position
        Stage 1 (Open): Start from closed, goal is open position
        """
        start_angles = object_cfg.start_angles if object_cfg.start_angles else [0.0]
        goal_angles = object_cfg.goal_angles if object_cfg.goal_angles else [1.57]
        
        if stage_index == 0:
            # Reach stage: both start and goal at closed position
            return start_angles, start_angles
        else:
            # Open stage: from closed to open
            return start_angles, goal_angles
    
    def _check_task_complete(self, obs: Dict[str, Any]) -> bool:
        """Check if reach-open task is complete."""
        if "env_state" not in obs:
            return False
        
        current_angle = obs.get("env_state", 0.0)
        goal_angle = self.cfg.object_cfg.goal_angles[-1] if self.cfg.object_cfg.goal_angles else 1.57
        
        return abs(current_angle - goal_angle) < 0.1
