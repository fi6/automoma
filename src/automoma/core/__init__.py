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
from automoma.core.config_loader import (
    Config,
    ConfigLoader,
    load_config,
    load_plan_config,
    load_record_config,
    load_train_config,
    load_eval_config,
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
    # Config (dataclasses - legacy)
    "PlanConfig",
    "RecordConfig",
    "TrainConfig",
    "EvalConfig",
    "SceneConfig",
    "ObjectConfig",
    "RobotConfig",
    "CameraConfig",
    # Config loader (new)
    "Config",
    "ConfigLoader",
    "load_config",
    "load_plan_config",
    "load_record_config", 
    "load_train_config",
    "load_eval_config",
    # Registry
    "Registry",
    "register_component",
    "get_component",
]
