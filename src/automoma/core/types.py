from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum, auto
import numpy as np
import torch
from scipy.spatial.transform import Rotation

class TaskType(Enum):
    PICK_PLACE = auto()
    REACH_OPEN = auto()
    PICK = auto()
    PLACE = auto()
    REACH = auto()
    OPEN = auto()
    CLOSE = auto()
    PULL = auto()
    PUSH = auto()
    COMPOSITE = auto()
    
class StageType(Enum):
    REACH = auto()              # Reach to the object
    GRASP = auto()              # Grasp (close gripper)
    LIFT = auto()               # Lift the object
    RELEASE = auto()            # Release (open gripper)
    MOVE = auto()               # Move
    MOVE_HOLDING = auto()       # Move while holding an object
    MOVE_ARTICULATED = auto()   # Move articulated parts
    WAIT = auto()               # Wait in place
    HOME = auto()               # Back to initial position

class GripperState(Enum):
    OPEN = auto()
    CLOSED = auto()
    MOVING = auto()    

@dataclass
class IKResult:
    start_iks: torch.Tensor
    goal_iks: torch.Tensor = None

    @classmethod
    def cat(cls, results: List[IKResult]) -> IKResult:
        """Merge multiple IKResult objects into one."""
        return cls(
            start_iks=torch.cat([r.start_iks for r in results], dim=0),
            goal_iks=torch.cat([r.goal_iks for r in results], dim=0) if results[0].goal_iks is not None else None
        )

@dataclass
class TrajResult:
    start_states: torch.Tensor
    goal_states: torch.Tensor
    trajectories: torch.Tensor
    success: torch.Tensor
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrajResult:
        return cls(
            start_states=data["start_states"],
            goal_states=data["goal_states"],
            trajectories=data["trajectories"],
            success=data["success"]
        )
    @classmethod
    def cat(cls, results: List[TrajResult]) -> TrajResult:
        """Merge multiple TrajResult objects into one."""
        if not results:
            return None
        return cls(
            start_states=torch.cat([r.start_states for r in results], dim=0),
            goal_states=torch.cat([r.goal_states for r in results], dim=0),
            trajectories=torch.cat([r.trajectories for r in results], dim=0),
            success=torch.cat([r.success for r in results], dim=0)
        )

    def __getitem__(self, indices: Union[torch.Tensor, np.ndarray, slice, List[int]]) -> TrajResult:
        """Allow indexing and masking of the result object."""
        return TrajResult(
            start_states=self.start_states[indices],
            goal_states=self.goal_states[indices],
            trajectories=self.trajectories[indices],
            success=self.success[indices]
        )

    @property
    def num_samples(self) -> int:
        return self.success.shape[0]
    
@dataclass
class PlanResult:
    task_type: TaskType
    stages: List[StageType]
    ik_results: List[IKResult]
    traj_results: List[TrajResult]



    