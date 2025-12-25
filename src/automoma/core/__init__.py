"""Core module for AutoMoMa framework."""

from automoma.core.types import (
    TaskType,
    StageType,
    GripperState,
    DatasetType,
    DatasetFormat,
    PoseType,
    CameraType,
    IKResult,
    TrajResult,
    PlanResult,
)
from automoma.core.interfaces import MotionPlannerInterface
from automoma.core.config import (
    PlanConfig,
    RecordConfig,
    TrainConfig,
    EvalConfig,
    SceneConfig,
    ObjectConfig,
    RobotConfig,
    CameraConfig,
)
from automoma.core.registry import Registry, register_component, get_component

__all__ = [
    # Types
    "TaskType",
    "StageType",
    "GripperState",
    "DatasetType",
    "DatasetFormat",
    "PoseType",
    "CameraType",
    "IKResult",
    "TrajResult",
    "PlanResult",
    # Interfaces
    "MotionPlannerInterface",
    # Config
    "PlanConfig",
    "RecordConfig",
    "TrainConfig",
    "EvalConfig",
    "SceneConfig",
    "ObjectConfig",
    "RobotConfig",
    "CameraConfig",
    # Registry
    "Registry",
    "register_component",
    "get_component",
]
