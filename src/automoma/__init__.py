"""AutoMoMa: A Python package for robot trajectory generation and simulation."""

__version__ = "0.1.0"

# Core modules
from automoma.core import (
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
    MotionPlannerInterface,
    PlanConfig,
    RecordConfig,
    TrainConfig,
    EvalConfig,
    SceneConfig,
    ObjectConfig,
    RobotConfig,
    CameraConfig,
    Registry,
    register_component,
    get_component,
)

# Planning modules
from automoma.planning import (
    BasePlanner,
    CuroboPlanner,
    PlanningPipeline,
    PlanningPrimitive,
    IKPlanningPrimitive,
    TrajectoryPlanningPrimitive,
)

# Dataset modules
from automoma.datasets import (
    BaseDatasetWrapper,
    LeRobotDatasetWrapper,
    DatasetConverter,
    Recorder,
)

# Evaluation modules
from automoma.evaluation import (
    EvaluationMetrics,
    MetricsCalculator,
    PolicyRunner,
    AsyncModelClient,
    LeRobotModelClient,
    get_model,
)

# Task modules
from automoma.tasks import (
    BaseTask,
    TaskFactory,
    create_task,
    PickPlaceTask,
    ReachOpenTask,
)

__all__ = [
    # Version
    "__version__",
    # Core types
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
    # Core interfaces
    "MotionPlannerInterface",
    # Core config
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
    # Planning
    "BasePlanner",
    "CuroboPlanner",
    "PlanningPipeline",
    "PlanningPrimitive",
    "IKPlanningPrimitive",
    "TrajectoryPlanningPrimitive",
    # Datasets
    "BaseDatasetWrapper",
    "LeRobotDatasetWrapper",
    "DatasetConverter",
    "Recorder",
    # Evaluation
    "EvaluationMetrics",
    "MetricsCalculator",
    "PolicyRunner",
    "AsyncModelClient",
    "LeRobotModelClient",
    "get_model",
    # Tasks
    "BaseTask",
    "TaskFactory",
    "create_task",
    "PickPlaceTask",
    "ReachOpenTask",
]
