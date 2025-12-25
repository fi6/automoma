"""Base task class for manipulation tasks."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch

from automoma.core.types import TaskType, StageType, PlanResult


logger = logging.getLogger(__name__)


@dataclass
class TaskState:
    """State of the task execution."""
    current_stage: StageType
    stage_index: int
    is_complete: bool
    success: bool
    error_message: str = ""
    stage_results: List[Any] = field(default_factory=list)


class BaseTask(ABC):
    """
    Base class for manipulation tasks.
    
    A task defines a sequence of stages (e.g., reach, grasp, lift, move, release)
    and the logic for executing and validating each stage.
    """
    
    def __init__(
        self,
        task_type: TaskType,
        stages: List[StageType],
        name: str = "",
    ):
        """
        Initialize the task.
        
        Args:
            task_type: Type of the task
            stages: Ordered list of stages to execute
            name: Optional task name
        """
        self.task_type = task_type
        self.stages = stages
        self.name = name or task_type.name
        
        self.state = TaskState(
            current_stage=stages[0] if stages else StageType.MOVE,
            stage_index=0,
            is_complete=False,
            success=False,
        )
        
        self.planner = None
        self.env = None
    
    def setup(self, planner=None, env=None) -> None:
        """
        Setup the task with planner and environment.
        
        Args:
            planner: Motion planner instance
            env: Environment wrapper instance
        """
        self.planner = planner
        self.env = env
    
    @abstractmethod
    def get_stage_config(self, stage: StageType) -> Dict[str, Any]:
        """
        Get configuration for a specific stage.
        
        Args:
            stage: The stage type
            
        Returns:
            Configuration dictionary for the stage
        """
        pass
    
    @abstractmethod
    def validate_stage(self, stage: StageType, result: Any) -> bool:
        """
        Validate the result of a stage.
        
        Args:
            stage: The stage type
            result: Result from executing the stage
            
        Returns:
            True if the stage was successful
        """
        pass
    
    @abstractmethod
    def get_success_criteria(self) -> Dict[str, Any]:
        """
        Get the success criteria for the task.
        
        Returns:
            Dictionary of success criteria
        """
        pass
    
    def execute_stage(self, stage: StageType, **kwargs) -> Tuple[Any, bool]:
        """
        Execute a single stage of the task.
        
        Args:
            stage: The stage to execute
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (result, success)
        """
        stage_config = self.get_stage_config(stage)
        stage_config.update(kwargs)
        
        try:
            # Execute based on stage type
            if stage in [StageType.MOVE, StageType.MOVE_HOLDING, StageType.MOVE_ARTICULATED, StageType.HOME]:
                result = self._execute_move_stage(stage, stage_config)
            elif stage == StageType.GRASP:
                result = self._execute_grasp_stage(stage_config)
            elif stage == StageType.RELEASE:
                result = self._execute_release_stage(stage_config)
            elif stage == StageType.LIFT:
                result = self._execute_lift_stage(stage_config)
            elif stage == StageType.WAIT:
                result = self._execute_wait_stage(stage_config)
            else:
                result = None
            
            success = self.validate_stage(stage, result)
            return result, success
            
        except Exception as e:
            self.state.error_message = str(e)
            return None, False
    
    def _execute_move_stage(self, stage: StageType, config: Dict[str, Any]) -> Any:
        """Execute a move stage using the planner."""
        if self.planner is None:
            raise ValueError("Planner not set up")
        
        start_iks = config.get("start_iks")
        goal_iks = config.get("goal_iks")
        
        if start_iks is None or goal_iks is None:
            raise ValueError("start_iks and goal_iks required for move stage")
        
        plan_cfg = {
            "stage_type": stage,
            "batch_size": config.get("batch_size", 10),
            "expand_to_pairs": config.get("expand_to_pairs", True),
        }
        
        return self.planner.plan_traj(
            start_iks=start_iks,
            goal_iks=goal_iks,
            plan_cfg=plan_cfg,
        )
    
    def _execute_grasp_stage(self, config: Dict[str, Any]) -> Any:
        """Execute a grasp stage."""
        if self.env is None:
            return None
        
        gripper_value = config.get("gripper_value", 0.0)
        
        if hasattr(self.env, "set_gripper"):
            self.env.set_gripper(gripper_value)
        
        return {"gripper_state": "closed", "gripper_value": gripper_value}
    
    def _execute_release_stage(self, config: Dict[str, Any]) -> Any:
        """Execute a release stage."""
        if self.env is None:
            return None
        
        gripper_value = config.get("gripper_value", 0.04)
        
        if hasattr(self.env, "set_gripper"):
            self.env.set_gripper(gripper_value)
        
        return {"gripper_state": "open", "gripper_value": gripper_value}
    
    def _execute_lift_stage(self, config: Dict[str, Any]) -> Any:
        """Execute a lift stage."""
        return self._execute_move_stage(StageType.LIFT, config)
    
    def _execute_wait_stage(self, config: Dict[str, Any]) -> Any:
        """Execute a wait stage."""
        wait_steps = config.get("wait_steps", 10)
        
        if self.env is not None and hasattr(self.env, "step"):
            for _ in range(wait_steps):
                self.env.step()
        
        return {"wait_steps": wait_steps}
    
    def run(self, **kwargs) -> TaskState:
        """
        Run the complete task.
        
        Args:
            **kwargs: Additional arguments for stages
            
        Returns:
            Final task state
        """
        for i, stage in enumerate(self.stages):
            self.state.current_stage = stage
            self.state.stage_index = i
            
            logger.info(f"Executing stage {i + 1}/{len(self.stages)}: {stage.name}")
            
            stage_kwargs = kwargs.get(stage.name.lower(), {})
            result, success = self.execute_stage(stage, **stage_kwargs)
            
            self.state.stage_results.append({
                "stage": stage,
                "result": result,
                "success": success,
            })
            
            if not success:
                self.state.success = False
                self.state.is_complete = True
                logger.warning(f"Stage {stage.name} failed: {self.state.error_message}")
                return self.state
        
        self.state.is_complete = True
        self.state.success = self._check_success()
        
        return self.state
    
    def _check_success(self) -> bool:
        """Check if the overall task was successful."""
        criteria = self.get_success_criteria()
        
        # All stages must have succeeded
        all_stages_success = all(
            r["success"] for r in self.state.stage_results
        )
        
        return all_stages_success
    
    def reset(self) -> None:
        """Reset the task state."""
        self.state = TaskState(
            current_stage=self.stages[0] if self.stages else StageType.MOVE,
            stage_index=0,
            is_complete=False,
            success=False,
        )
