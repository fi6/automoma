"""Reach and open task implementation for articulated objects."""

from typing import Dict, List, Optional, Any
from automoma.core.types import TaskType, StageType, TrajResult
from automoma.tasks.base_task import BaseTask


class ReachOpenTask(BaseTask):
    """
    Reach and open manipulation task for articulated objects.
    
    This task involves:
    1. Moving to approach position
    2. Reaching to grasp position on handle
    3. Grasping the handle
    4. Moving while articulating the object (opening motion)
    5. Releasing the handle
    6. Returning to home position
    """
    
    DEFAULT_STAGES = [
        StageType.MOVE,               # Move to approach
        StageType.MOVE,               # Reach to handle
        StageType.GRASP,              # Close gripper on handle
        StageType.MOVE_ARTICULATED,   # Open the articulated object
        StageType.RELEASE,            # Release handle
        StageType.HOME,               # Return home
    ]
    
    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        """
        Initialize reach and open task.
        
        Args:
            config: Task configuration
            **kwargs: Additional arguments
        """
        stages = config.get("stages", self.DEFAULT_STAGES) if config else self.DEFAULT_STAGES
        
        super().__init__(
            task_type=TaskType.REACH_OPEN,
            stages=stages,
            name="reach_open",
        )
        
        self.config = config or {}
        
        # Task parameters
        self.handle_grasp_pose = self.config.get("handle_grasp_pose", None)
        self.home_pose = self.config.get("home_pose", None)
        self.approach_offset = self.config.get("approach_offset", [0, 0, 0.1])
        
        # Articulation parameters
        self.start_angle = self.config.get("start_angle", 0.0)
        self.goal_angle = self.config.get("goal_angle", 1.57)  # ~90 degrees
        self.handle_link = self.config.get("handle_link", "link_0")
        
        # Gripper parameters
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
            StageType.MOVE_ARTICULATED: {
                **base_config,
                "start_angle": self.start_angle,
                "goal_angle": self.goal_angle,
                "handle_link": self.handle_link,
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
        
        if stage in [StageType.MOVE, StageType.MOVE_ARTICULATED, StageType.HOME]:
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
            "angle_threshold": self.config.get("angle_threshold", 0.1),  # Articulation angle
            "target_angle": self.goal_angle,
        }
    
    def get_intermediate_angles(self, num_waypoints: int = 5) -> List[float]:
        """
        Get intermediate angles for smooth articulation motion.
        
        Args:
            num_waypoints: Number of intermediate waypoints
            
        Returns:
            List of angles from start to goal
        """
        import numpy as np
        return np.linspace(self.start_angle, self.goal_angle, num_waypoints).tolist()
