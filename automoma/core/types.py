# Copyright (c) 2024-2025, AutoMoMa Authors. All rights reserved.
# SPDX-License-Identifier: MIT
"""Core data types for the AutoMoMa planning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskType(Enum):
    """High-level manipulation task types."""
    REACH_OPEN = auto()
    PICK_PLACE = auto()
    PICK = auto()
    PLACE = auto()
    REACH = auto()
    OPEN = auto()
    CLOSE = auto()
    COMPOSITE = auto()


class StageType(Enum):
    """Individual stage within a task."""
    GRASP = auto()
    LIFT = auto()
    RELEASE = auto()
    MOVE = auto()
    MOVE_HOLDING = auto()
    MOVE_ARTICULATED = auto()
    WAIT = auto()
    HOME = auto()


class GripperState(Enum):
    OPEN = auto()
    CLOSED = auto()


# ---------------------------------------------------------------------------
# IKResult
# ---------------------------------------------------------------------------

@dataclass
class IKResult:
    """Inverse-kinematics solution container.

    Attributes:
        target_poses: ``[N, 7]`` end-effector goal poses (``x y z qw qx qy qz``).
        iks:          ``[N, D]`` joint-space solutions.
    """

    target_poses: torch.Tensor
    iks: torch.Tensor

    # -- Factory helpers -----------------------------------------------------

    @classmethod
    def cat(cls, results: List[IKResult]) -> IKResult:
        non_empty = [r for r in results if r.iks.shape[0] > 0]
        if not non_empty:
            dof = results[0].iks.shape[1] if results and results[0].iks.ndim > 1 else 0
            return cls(target_poses=torch.empty(0, 7), iks=torch.empty(0, dof))
        dev = non_empty[0].iks.device
        return cls(
            target_poses=torch.cat([r.target_poses.to(dev) for r in non_empty]),
            iks=torch.cat([r.iks.to(dev) for r in non_empty]),
        )

    @classmethod
    def fallback(cls, robot_dof: int = 0, num_samples: int = 0) -> IKResult:
        return cls(
            target_poses=torch.zeros(num_samples, 7),
            iks=torch.zeros(num_samples, robot_dof),
        )

    # -- Indexing / slicing --------------------------------------------------

    def __getitem__(self, idx: Union[torch.Tensor, np.ndarray, slice, List[int]]) -> IKResult:
        return IKResult(target_poses=self.target_poses[idx], iks=self.iks[idx])

    def __len__(self) -> int:
        return self.iks.shape[0]

    def downsample(self, max_samples: int) -> IKResult:
        if len(self) <= max_samples:
            return self
        return self[torch.randperm(len(self))[:max_samples]]


# ---------------------------------------------------------------------------
# TrajResult
# ---------------------------------------------------------------------------

@dataclass
class TrajResult:
    """Trajectory-optimisation result container.

    Attributes:
        start_states:  ``[N, D]`` start joint positions.
        goal_states:   ``[N, D]`` goal joint positions.
        trajectories:  ``[N, T, D]`` planned waypoints.
        success:       ``[N]``  boolean success flag per trajectory.
    """

    start_states: torch.Tensor
    goal_states: torch.Tensor
    trajectories: torch.Tensor
    success: torch.Tensor

    # -- Factory helpers -----------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrajResult:
        return cls(
            start_states=data["start_states"],
            goal_states=data["goal_states"],
            trajectories=data["trajectories"],
            success=data["success"],
        )

    @classmethod
    def cat(cls, results: List[TrajResult]) -> TrajResult:
        non_empty = [r for r in results if r.trajectories.shape[0] > 0]
        if not non_empty:
            dof = results[0].trajectories.shape[2] if results and results[0].trajectories.ndim > 2 else 0
            steps = results[0].trajectories.shape[1] if results and results[0].trajectories.ndim > 1 else 1
            return cls(
                start_states=torch.empty(0, dof),
                goal_states=torch.empty(0, dof),
                trajectories=torch.empty(0, steps, dof),
                success=torch.empty(0, dtype=torch.bool),
            )
        dev = non_empty[0].trajectories.device
        return cls(
            start_states=torch.cat([r.start_states.to(dev) for r in non_empty]),
            goal_states=torch.cat([r.goal_states.to(dev) for r in non_empty]),
            trajectories=torch.cat([r.trajectories.to(dev) for r in non_empty]),
            success=torch.cat([r.success.to(dev) for r in non_empty]),
        )

    @classmethod
    def fallback(cls, robot_dof: int = 0, num_samples: int = 0) -> TrajResult:
        return cls(
            start_states=torch.zeros(num_samples, robot_dof),
            goal_states=torch.zeros(num_samples, robot_dof),
            trajectories=torch.zeros(num_samples, 1, robot_dof),
            success=torch.zeros(num_samples, dtype=torch.bool),
        )

    # -- Indexing / slicing --------------------------------------------------

    def __getitem__(self, idx: Union[torch.Tensor, np.ndarray, slice, List[int]]) -> TrajResult:
        return TrajResult(
            start_states=self.start_states[idx],
            goal_states=self.goal_states[idx],
            trajectories=self.trajectories[idx],
            success=self.success[idx],
        )

    @property
    def num_samples(self) -> int:
        return self.success.shape[0]


# ---------------------------------------------------------------------------
# PlanResult (aggregated output)
# ---------------------------------------------------------------------------

@dataclass
class PlanResult:
    """Aggregated planning output."""

    task_type: TaskType
    stages: List[StageType]
    ik_results: List[IKResult]
    traj_results: List[TrajResult]
