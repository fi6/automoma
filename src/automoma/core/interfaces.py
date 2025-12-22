from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Callable
from automoma.core.types import IKResult, TrajResult, PlanResult
import numpy as np
import torch

class MotionPlannerInterface(ABC):
    @abstractmethod
    def setup_env(self, scene_cfg: Dict[str, Any], object_cfg: Dict[str, Any]) -> None: 
        pass
    @abstractmethod
    def plan_ik(self, target_pose: torch.Tensor, robot_cfg: Dict[str, Any] = None, plan_cfg: Dict[str, Any] = None, motion_gen = None) -> torch.Tensor:
        pass
    @abstractmethod
    def plan_traj(self, start_states: torch.Tensor, goal_states: torch.Tensor = None, robot_cfg: Dict[str, Any] = None, plan_cfg: Dict[str, Any] = None, motion_gen = None) -> TrajResult:
        pass
        