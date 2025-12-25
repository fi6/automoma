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
    GRASP = auto()              # Grasp (close gripper)
    LIFT = auto()               # Lift the object
    RELEASE = auto()            # Release (open gripper)
    MOVE = auto()               # Move
    MOVE_HOLDING = auto()       # Move while holding an object
    MOVE_ARTICULATED = auto()   # Move articulated parts
    WAIT = auto()               # Wait in place
    HOME = auto()               # Back to initial position
    COMPOSITE = auto()          # Composite stage (multiple primitives)

class GripperState(Enum):
    OPEN = auto()
    CLOSED = auto()
    MOVING = auto()    
    
class DatasetType(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()
    
class DatasetFormat(Enum):
    LEROBOT = auto()
    HDF5 = auto()
    ZARR = auto()
    
class PoseType(Enum):
    LOCAL = auto()
    WORLD = auto()
    
class CameraType(Enum):
    EGO_TOPDOWN = auto()
    EGO_WRIST = auto()
    FIX_LOCAL = auto()

@dataclass
class IKResult:
    target_poses: torch.Tensor
    iks: torch.Tensor
    
    @classmethod
    def cat(cls, results: List["IKResult"]) -> "IKResult":
        """Merge multiple IKResult objects into one."""
        if not results:
            # Return empty IKResult with zero-sized tensors for consistent type
            return cls(
                target_poses=torch.empty((0, 7)),
                iks=torch.empty((0, 0))
            )
        return cls(
            target_poses=torch.cat([r.target_poses for r in results], dim=0),
            iks=torch.cat([r.iks for r in results], dim=0)
        )
        
    @classmethod
    def fallback(cls, robot_dof: int, num_samples: int=0) -> "IKResult":
        """Create a fallback IKResult with failed IK solutions."""
        return cls(
            target_poses=torch.zeros((num_samples, 7)),
            iks=torch.zeros((num_samples, robot_dof))
        )
        
    def __getitem__(self, indices: Union[torch.Tensor, np.ndarray, slice, List[int]]) -> IKResult:
        """Allow indexing and masking of the result object."""
        return IKResult(
            target_poses=self.target_poses[indices],
            iks=self.iks[indices]
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
    @classmethod
    def fallback(cls, robot_dof: int, num_samples: int=0) -> TrajResult:
        """Create a fallback TrajResult with failed trajectories."""
        return cls(
            start_states=torch.zeros((num_samples, robot_dof)),
            goal_states=torch.zeros((num_samples, robot_dof)),
            trajectories=torch.zeros((num_samples, 1, robot_dof)),
            success=torch.zeros((num_samples,), dtype=torch.bool)
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



    