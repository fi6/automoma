"""Pick and place task implementation."""

from typing import Dict, List, Optional, Any
from automoma.core.types import TaskType, StageType, TrajResult
from automoma.tasks.base_task import BaseTask


class PickPlaceTask(BaseTask):
    """
    Pick and place manipulation task.
    
    This task involves:
    1. Moving to approach position
    2. Reaching to grasp position
    3. Grasping the object
    4. Lifting the object
    5. Moving to place position
    6. Lowering the object
    7. Releasing the object
    8. Returning to home position
    """
    
    DEFAULT_STAGES = [
        StageType.MOVE,           # Move to approach
        StageType.MOVE,           # Reach to grasp
        StageType.GRASP,          # Close gripper
        StageType.LIFT,           # Lift object
        StageType.MOVE_HOLDING,   # Move to place
        StageType.MOVE_HOLDING,   # Lower object
        StageType.RELEASE,        # Open gripper
        StageType.HOME,           # Return home
    ]
    
    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        """
        Initialize pick and place task.
        
        Args:
            config: Task configuration
            **kwargs: Additional arguments
        """
        stages = config.get("stages", self.DEFAULT_STAGES) if config else self.DEFAULT_STAGES
        
        super().__init__(
            task_type=TaskType.PICK_PLACE,
            stages=stages,
            name="pick_place",
        )
        
        self.config = config or {}
        
        # Task parameters
        self.grasp_pose = self.config.get("grasp_pose", None)
        self.place_pose = self.config.get("place_pose", None)
        self.home_pose = self.config.get("home_pose", None)
        self.approach_offset = self.config.get("approach_offset", [0, 0, 0.1])
        self.lift_height = self.config.get("lift_height", 0.1)
        self.gripper_close_value = self.config.get("gripper_close_value", 0.0)
        self.gripper_open_value = self.config.get("gripper_open_value", 0.04)
    
    def get_stage_config(self, stage: StageType) -> Dict[str, Any]:
        """Get configuration for a specific stage."""
        base_config = {
            "batch_size": self.config.get("batch_size", 10),
            "expand_to_pairs": self.config.get("expand_to_pairs", True),
        }
        
        stage_configs = {
            StageType.MOVE: {
                **base_config,
            },
            StageType.GRASP: {
                "gripper_value": self.gripper_close_value,
            },
            StageType.LIFT: {
                **base_config,
                "lift_height": self.lift_height,
            },
            StageType.MOVE_HOLDING: {
                **base_config,
            },
            StageType.RELEASE: {
                "gripper_value": self.gripper_open_value,
            },
            StageType.HOME: {
                **base_config,
            },
        }
        
        return stage_configs.get(stage, base_config)
    
    def validate_stage(self, stage: StageType, result: Any) -> bool:
        """Validate the result of a stage."""
        if result is None:
            return False
        
        if stage in [StageType.MOVE, StageType.LIFT, StageType.MOVE_HOLDING, StageType.HOME]:
            # For trajectory stages, check if any trajectories succeeded
            if isinstance(result, TrajResult):
                return result.success.any()
            return False
        
        elif stage in [StageType.GRASP, StageType.RELEASE]:
            # For gripper stages, check if gripper state is correct
            if isinstance(result, dict):
                return "gripper_state" in result
            return False
        
        elif stage == StageType.WAIT:
            return True
        
        return True
    
    def get_success_criteria(self) -> Dict[str, Any]:
        """Get the success criteria for the task."""
        return {
            "position_threshold": self.config.get("position_threshold", 0.02),
            "orientation_threshold": self.config.get("orientation_threshold", 0.1),
            "object_placed": True,
        }
