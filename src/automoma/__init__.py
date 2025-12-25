"""AutoMoMa: A Python package for robot trajectory generation and simulation."""

__version__ = "0.1.0"

# Core modules - always safe to import
from automoma.core import (
    # Types
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
    # Config dataclasses (legacy)
    PlanConfig,
    RecordConfig,
    TrainConfig,
    EvalConfig,
    SceneConfig,
    ObjectConfig,
    RobotConfig,
    CameraConfig,
    # Config loader (new)
    Config,
    ConfigLoader,
    load_config,
    load_plan_config,
    load_record_config,
    load_train_config,
    load_eval_config,
    # Registry
    Registry,
    register_component,
    get_component,
)

# Planning modules - safe to import without Isaac Sim
from automoma.planning import (
    BasePlanner,
    CuroboPlanner,
    PlanningPipeline,
)

# Dataset modules - safe to import without Isaac Sim
from automoma.datasets import (
    BaseDatasetWrapper,
    LeRobotDatasetWrapper,
    DatasetConverter,
    Recorder,
)

# Evaluation modules - safe to import without Isaac Sim
from automoma.evaluation import (
    EvaluationMetrics,
    MetricsCalculator,
    PolicyRunner,
    AsyncModelClient,
    LeRobotModelClient,
    get_model,
)

# Task modules - safe to import without Isaac Sim
from automoma.tasks import (
    BaseTask,
    TaskResult,
    StageResult,
    TaskFactory,
    create_task,
    create_task_from_exp,
    OpenTask,
    ReachOpenTask,
    PickTask,
    PlaceTask,
    PickPlaceTask,
)

# Simulation module management - safe to import
from automoma.simulation import (
    get_simulation_app,
    is_sim_app_initialized,
    close_simulation_app,
    require_simulation_app,
)


# Note: SimEnvWrapper and other simulation classes require SimulationApp
# They can be imported after calling get_simulation_app():
#
#     from automoma import get_simulation_app
#     sim_app = get_simulation_app(headless=False)
#     from automoma.simulation import SimEnvWrapper


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
    # Core config (dataclasses - legacy)
    "PlanConfig",
    "RecordConfig",
    "TrainConfig",
    "EvalConfig",
    "SceneConfig",
    "ObjectConfig",
    "RobotConfig",
    "CameraConfig",
    # Config loader (new - recommended)
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
    # Planning
    "BasePlanner",
    "CuroboPlanner",
    "PlanningPipeline",
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
    "TaskResult",
    "StageResult",
    "TaskFactory",
    "create_task",
    "create_task_from_exp",
    "OpenTask",
    "ReachOpenTask",
    "PickTask",
    "PlaceTask",
    "PickPlaceTask",
    # Simulation management
    "get_simulation_app",
    "is_sim_app_initialized",
    "close_simulation_app",
    "require_simulation_app",
]
