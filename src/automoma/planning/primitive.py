"""Planning primitives for motion planning tasks."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np

from automoma.core.types import TaskType, StageType, IKResult, TrajResult


class PlanningPrimitive(ABC):
    """Base class for planning primitives."""
    
    def __init__(self, name: str, stage_type: StageType):
        self.name = name
        self.stage_type = stage_type
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the planning primitive."""
        pass
    
    @abstractmethod
    def validate(self, result: Any) -> bool:
        """Validate the result of the planning primitive."""
        pass


class IKPlanningPrimitive(PlanningPrimitive):
    """Primitive for inverse kinematics planning."""
    
    def __init__(self, planner, robot_cfg: Dict[str, Any]):
        super().__init__("ik_planning", StageType.MOVE)
        self.planner = planner
        self.robot_cfg = robot_cfg
        self.motion_gen = None
    
    def setup(self, motion_gen=None) -> None:
        """Setup the motion generator."""
        if motion_gen is not None:
            self.motion_gen = motion_gen
        else:
            self.motion_gen = self.planner.init_motion_gen(self.robot_cfg)
    
    def execute(
        self,
        target_pose: torch.Tensor,
        joint_cfg: Optional[Dict[str, float]] = None,
        enable_collision: bool = True,
    ) -> IKResult:
        """
        Execute IK planning for the target pose.
        
        Args:
            target_pose: Target end-effector pose [x, y, z, qw, qx, qy, qz]
            joint_cfg: Joint configuration for articulated objects
            enable_collision: Whether to enable collision checking
            
        Returns:
            IKResult containing the IK solutions
        """
        plan_cfg = {
            "joint_cfg": joint_cfg,
            "enable_collision": enable_collision,
        }
        
        return self.planner.plan_ik(
            target_pose=target_pose,
            robot_cfg=self.robot_cfg,
            plan_cfg=plan_cfg,
            motion_gen=self.motion_gen,
        )
    
    def validate(self, result: IKResult) -> bool:
        """Validate IK result has valid solutions."""
        return result is not None and result.iks.shape[0] > 0


class TrajectoryPlanningPrimitive(PlanningPrimitive):
    """Primitive for trajectory planning."""
    
    def __init__(
        self,
        planner,
        robot_cfg: Dict[str, Any],
        stage_type: StageType = StageType.MOVE,
    ):
        super().__init__("traj_planning", stage_type)
        self.planner = planner
        self.robot_cfg = robot_cfg
        self.motion_gen = None
        self.batch_size = 10
    
    def setup(self, motion_gen=None, batch_size: int = 10) -> None:
        """Setup the motion generator and batch size."""
        if motion_gen is not None:
            self.motion_gen = motion_gen
        else:
            self.motion_gen = self.planner.init_motion_gen(self.robot_cfg)
        self.batch_size = batch_size
    
    def execute(
        self,
        start_iks: torch.Tensor,
        goal_iks: torch.Tensor,
        expand_to_pairs: bool = True,
    ) -> TrajResult:
        """
        Execute trajectory planning from start to goal IK solutions.
        
        Args:
            start_iks: Start IK solutions
            goal_iks: Goal IK solutions
            expand_to_pairs: Whether to expand start/goal to all pairs
            
        Returns:
            TrajResult containing the planned trajectories
        """
        plan_cfg = {
            "stage_type": self.stage_type,
            "batch_size": self.batch_size,
            "expand_to_pairs": expand_to_pairs,
        }
        
        return self.planner.plan_traj(
            start_iks=start_iks,
            goal_iks=goal_iks,
            robot_cfg=self.robot_cfg,
            plan_cfg=plan_cfg,
            motion_gen=self.motion_gen,
        )
    
    def validate(self, result: TrajResult) -> bool:
        """Validate trajectory result has successful trajectories."""
        return result is not None and result.success.sum() > 0


class GraspPrimitive(PlanningPrimitive):
    """Primitive for grasp execution."""
    
    def __init__(self, gripper_close_value: float = 0.0):
        super().__init__("grasp", StageType.GRASP)
        self.gripper_close_value = gripper_close_value
    
    def execute(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        Execute grasp by closing gripper.
        
        Args:
            current_state: Current robot state
            
        Returns:
            New state with closed gripper
        """
        new_state = current_state.clone()
        # Assuming last 2 dimensions are gripper fingers
        new_state[-2:] = self.gripper_close_value
        return new_state
    
    def validate(self, result: torch.Tensor) -> bool:
        """Validate gripper is closed."""
        return result[-2:].sum() <= 2 * self.gripper_close_value + 0.01


class ReleasePrimitive(PlanningPrimitive):
    """Primitive for release execution."""
    
    def __init__(self, gripper_open_value: float = 0.04):
        super().__init__("release", StageType.RELEASE)
        self.gripper_open_value = gripper_open_value
    
    def execute(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        Execute release by opening gripper.
        
        Args:
            current_state: Current robot state
            
        Returns:
            New state with open gripper
        """
        new_state = current_state.clone()
        new_state[-2:] = self.gripper_open_value
        return new_state
    
    def validate(self, result: torch.Tensor) -> bool:
        """Validate gripper is open."""
        return result[-2:].sum() >= 2 * self.gripper_open_value - 0.01


class WaitPrimitive(PlanningPrimitive):
    """Primitive for waiting in place."""
    
    def __init__(self, num_steps: int = 10):
        super().__init__("wait", StageType.WAIT)
        self.num_steps = num_steps
    
    def execute(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        Execute wait by repeating current state.
        
        Args:
            current_state: Current robot state
            
        Returns:
            Trajectory of repeated states
        """
        return current_state.unsqueeze(0).repeat(self.num_steps, 1)
    
    def validate(self, result: torch.Tensor) -> bool:
        """Validate wait trajectory."""
        return result is not None and result.shape[0] == self.num_steps


class ComposePrimitive(PlanningPrimitive):
    """Compose multiple primitives into a sequence."""
    
    def __init__(self, primitives: List[PlanningPrimitive]):
        super().__init__("compose", StageType.COMPOSITE)
        self.primitives = primitives
    
    def execute(self, initial_state: torch.Tensor, **kwargs) -> List[Any]:
        """
        Execute all primitives in sequence.
        
        Args:
            initial_state: Initial robot state
            **kwargs: Additional arguments for primitives
            
        Returns:
            List of results from each primitive
        """
        results = []
        current_state = initial_state
        
        for primitive in self.primitives:
            result = primitive.execute(current_state, **kwargs.get(primitive.name, {}))
            results.append(result)
            
            # Update current state based on result type
            if isinstance(result, torch.Tensor):
                if result.dim() == 1:
                    current_state = result
                elif result.dim() == 2:
                    current_state = result[-1]
            elif isinstance(result, TrajResult):
                if result.success.any():
                    current_state = result.trajectories[result.success][-1, -1]
        
        return results
    
    def validate(self, results: List[Any]) -> bool:
        """Validate all primitive results."""
        return all(
            primitive.validate(result)
            for primitive, result in zip(self.primitives, results)
        )
