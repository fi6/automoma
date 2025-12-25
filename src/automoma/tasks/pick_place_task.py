"""Pick and place task implementation."""

import logging
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np

from automoma.core.types import TaskType, StageType, IKResult
from automoma.core.config_loader import Config
from automoma.tasks.base_task_new import BaseTask, TaskResult, StageResult


logger = logging.getLogger(__name__)


class PickTask(BaseTask):
    """
    Pick task - reach and grasp an object.
    
    Stages:
    1. MOVE - Reach to pre-grasp position
    2. MOVE - Move to grasp position
    3. GRASP - Close gripper
    """
    
    STAGES = [StageType.MOVE, StageType.MOVE, StageType.GRASP]
    TASK_TYPE = TaskType.PICK
    
    def __init__(self, cfg: Config):
        """Initialize pick task."""
        super().__init__(cfg)
        self.name = "pick"
    
    def get_grasp_poses(self, object_cfg: Config, **kwargs) -> List[List[float]]:
        """Get grasp poses for picking."""
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
        Get target pose for pick stages.
        
        Stage 0: Pre-grasp position (offset from grasp)
        Stage 1: Grasp position
        Stage 2: Same as grasp (for gripper close)
        """
        grasp_tensor = torch.tensor(grasp_pose)
        
        if stage_index == 0:
            # Pre-grasp: offset in approach direction (typically -z)
            offset = torch.tensor([0, 0, -0.1, 0, 0, 0, 0])  # 10cm approach
            return grasp_tensor + offset
        else:
            return grasp_tensor
    
    def get_stage_type(self, stage_index: int) -> StageType:
        """Get stage type based on index."""
        return self.STAGES[stage_index]
    
    def _get_stage_angles(self, stage_index: int, object_cfg: Config) -> Tuple[List[float], List[float]]:
        """Pick task doesn't use articulation angles."""
        return [0.0], [0.0]
    
    def _check_task_complete(self, obs: Dict[str, Any]) -> bool:
        """Check if pick is complete (object grasped)."""
        # Could check gripper state or contact sensors
        return obs.get("gripper_closed", False)


class PlaceTask(BaseTask):
    """
    Place task - move and release an object.
    
    Assumes robot starts holding the object.
    
    Stages:
    1. MOVE_HOLDING - Move to pre-place position
    2. MOVE_HOLDING - Lower to place position  
    3. RELEASE - Open gripper
    4. MOVE - Retract
    """
    
    STAGES = [StageType.MOVE_HOLDING, StageType.MOVE_HOLDING, StageType.RELEASE, StageType.MOVE]
    TASK_TYPE = TaskType.PLACE
    
    def __init__(self, cfg: Config):
        """Initialize place task."""
        super().__init__(cfg)
        self.name = "place"
    
    def get_grasp_poses(self, object_cfg: Config, **kwargs) -> List[List[float]]:
        """Get place poses (target positions)."""
        # For place task, "grasp_poses" are actually place poses
        place_poses = object_cfg.place_poses if object_cfg.place_poses else []
        return place_poses
    
    def get_target_pose_for_stage(
        self,
        stage_index: int,
        grasp_pose: List[float],  # Actually place pose for this task
        angle: float,
        object_cfg: Config,
    ) -> torch.Tensor:
        """Get target pose for place stages."""
        place_tensor = torch.tensor(grasp_pose)
        
        if stage_index == 0:
            # Pre-place position (above place location)
            offset = torch.tensor([0, 0, 0.1, 0, 0, 0, 0])
            return place_tensor + offset
        elif stage_index == 3:
            # Retract position
            offset = torch.tensor([0, 0, 0.15, 0, 0, 0, 0])
            return place_tensor + offset
        else:
            return place_tensor
    
    def get_stage_type(self, stage_index: int) -> StageType:
        """Get stage type based on index."""
        return self.STAGES[stage_index]
    
    def _get_stage_angles(self, stage_index: int, object_cfg: Config) -> Tuple[List[float], List[float]]:
        """Place task doesn't use articulation angles."""
        return [0.0], [0.0]
    
    def _check_task_complete(self, obs: Dict[str, Any]) -> bool:
        """Check if place is complete."""
        return obs.get("gripper_open", False) and obs.get("retracted", False)


class PickPlaceTask(BaseTask):
    """
    Full pick and place task.
    
    Combines pick and place in sequence:
    1. MOVE - Reach to pre-grasp
    2. MOVE - Move to grasp  
    3. GRASP - Close gripper
    4. LIFT - Lift object
    5. MOVE_HOLDING - Move to place location
    6. MOVE_HOLDING - Lower to place
    7. RELEASE - Open gripper
    8. MOVE - Retract
    
    Stage chaining:
    - Stage 1 goal IKs -> Stage 2 start IKs
    - Stage 2 goal IKs -> Stage 3 start IKs
    - etc.
    """
    
    STAGES = [
        StageType.MOVE,          # Pre-grasp
        StageType.MOVE,          # Grasp
        StageType.GRASP,         # Close gripper
        StageType.LIFT,          # Lift
        StageType.MOVE_HOLDING,  # Move to place
        StageType.MOVE_HOLDING,  # Lower
        StageType.RELEASE,       # Open gripper
        StageType.MOVE,          # Retract
    ]
    TASK_TYPE = TaskType.PICK_PLACE
    
    def __init__(self, cfg: Config):
        """Initialize pick-place task."""
        super().__init__(cfg)
        self.name = "pick_place"
    
    def get_grasp_poses(self, object_cfg: Config, **kwargs) -> List[List[float]]:
        """Get grasp poses for the object."""
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
        """Get target pose based on stage."""
        grasp_tensor = torch.tensor(grasp_pose)
        
        # Get place pose if available
        place_pose = object_cfg.place_pose if object_cfg.place_pose else grasp_pose
        place_tensor = torch.tensor(place_pose)
        
        lift_height = object_cfg.lift_height if object_cfg.lift_height else 0.1
        
        if stage_index == 0:  # Pre-grasp
            return grasp_tensor + torch.tensor([0, 0, -0.1, 0, 0, 0, 0])
        elif stage_index in [1, 2]:  # Grasp position
            return grasp_tensor
        elif stage_index == 3:  # Lift
            return grasp_tensor + torch.tensor([0, 0, lift_height, 0, 0, 0, 0])
        elif stage_index == 4:  # Move to place (above)
            return place_tensor + torch.tensor([0, 0, lift_height, 0, 0, 0, 0])
        elif stage_index in [5, 6]:  # Place position
            return place_tensor
        elif stage_index == 7:  # Retract
            return place_tensor + torch.tensor([0, 0, 0.15, 0, 0, 0, 0])
        else:
            return grasp_tensor
    
    def get_stage_type(self, stage_index: int) -> StageType:
        """Get stage type based on index."""
        return self.STAGES[stage_index]
    
    def _get_stage_angles(self, stage_index: int, object_cfg: Config) -> Tuple[List[float], List[float]]:
        """Pick-place doesn't use articulation angles."""
        return [0.0], [0.0]
    
    def _check_task_complete(self, obs: Dict[str, Any]) -> bool:
        """Check if pick-place is complete."""
        return obs.get("object_placed", False) and obs.get("retracted", False)
