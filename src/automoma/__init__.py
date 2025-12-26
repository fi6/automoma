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

# Note: Planning, Dataset, Evaluation, and Task modules are NOT safe to import 
# before SimulationApp is initialized if they depend on curobo or pxr.
# They should be imported from their respective submodules when needed.

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
    # Simulation management
    "get_simulation_app",
    "is_sim_app_initialized",
    "close_simulation_app",
    "require_simulation_app",
]
