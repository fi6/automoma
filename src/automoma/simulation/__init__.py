"""Simulation module for AutoMoMa framework.

IMPORTANT: For modules that require Isaac Sim (SimEnvWrapper, SceneBuilder, 
SensorRig), you must first initialize SimulationApp:

    from automoma.simulation.sim_app_manager import get_simulation_app
    
    # Initialize SimulationApp (only call once)
    sim_app = get_simulation_app(headless=False)
    
    # Now you can import and use simulation modules
    from automoma.simulation import SimEnvWrapper
    env = SimEnvWrapper(cfg)
    env.setup_env()
    
For planning-only code that doesn't need Isaac Sim, you can directly use:
    from automoma.planning import CuroboPlanner
"""

from automoma.simulation.sim_app_manager import (
    get_simulation_app,
    is_sim_app_initialized,
    close_simulation_app,
    require_simulation_app,
)

# Lazy imports - these should only be used after SimulationApp is initialized
# We define __getattr__ to provide helpful error messages


def __getattr__(name):
    """Lazy loading with helpful error messages for simulation modules."""
    sim_classes = {
        "SimEnvWrapper": "env_wrapper",
        "SceneBuilder": "scene_builder",
        "InfinigenBuilder": "scene_builder",
        "SensorRig": "sensors",
    }
    
    if name in sim_classes:
        # Check if SimulationApp is initialized
        if not is_sim_app_initialized():
            raise RuntimeError(
                f"Cannot import {name}: SimulationApp is not initialized. "
                f"Call get_simulation_app() before importing simulation modules.\n\n"
                f"Example:\n"
                f"    from automoma.simulation import get_simulation_app\n"
                f"    sim_app = get_simulation_app(headless=False)\n"
                f"    from automoma.simulation import {name}\n"
            )
        
        # Import the module and get the class
        module_name = sim_classes[name]
        import importlib
        module = importlib.import_module(f"automoma.simulation.{module_name}")
        return getattr(module, name)
    
    raise AttributeError(f"module 'automoma.simulation' has no attribute '{name}'")


__all__ = [
    # SimulationApp management
    "get_simulation_app",
    "is_sim_app_initialized",
    "close_simulation_app",
    "require_simulation_app",
    # Simulation classes (lazy loaded)
    "SimEnvWrapper",
    "SceneBuilder",
    "InfinigenBuilder",
    "SensorRig",
]
