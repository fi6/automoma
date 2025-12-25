"""
Singleton manager for Isaac Sim's SimulationApp.

This module ensures that:
1. SimulationApp is only initialized once per process
2. SimulationApp is initialized lazily (only when needed)
3. Code that doesn't need Isaac Sim can run without it

Usage:
    from automoma.simulation.sim_app_manager import get_simulation_app, is_sim_app_initialized
    
    # Check if already initialized
    if not is_sim_app_initialized():
        # Initialize with custom settings
        sim_app = get_simulation_app(headless=True, width=1920, height=1080)
    else:
        # Get existing instance
        sim_app = get_simulation_app()
"""

import logging
from typing import Optional, Dict, Any


logger = logging.getLogger(__name__)


# Global singleton state
_simulation_app: Optional[Any] = None
_is_initialized: bool = False


def is_sim_app_initialized() -> bool:
    """Check if SimulationApp has been initialized."""
    return _is_initialized


def get_simulation_app(
    headless: bool = False,
    width: int = 1920,
    height: int = 1080,
    **kwargs
) -> Any:
    """
    Get or create the SimulationApp singleton.
    
    Args:
        headless: Whether to run in headless mode
        width: Window width
        height: Window height
        **kwargs: Additional arguments passed to SimulationApp
        
    Returns:
        The SimulationApp instance
        
    Note:
        Settings are only applied on first initialization.
        Subsequent calls return the existing instance.
    """
    global _simulation_app, _is_initialized
    
    if _is_initialized:
        logger.debug("Returning existing SimulationApp instance")
        return _simulation_app
    
    logger.info(f"Initializing SimulationApp (headless={headless}, {width}x{height})")
    
    # Import isaacsim first to set up the environment
    import isaacsim
    from omni.isaac.kit import SimulationApp
    
    # Create the app with settings
    app_config = {
        "headless": headless,
        "width": width,
        "height": height,
        **kwargs
    }
    
    _simulation_app = SimulationApp(app_config)
    _is_initialized = True
    
    logger.info("SimulationApp initialized successfully")
    return _simulation_app


def close_simulation_app() -> None:
    """Close the SimulationApp if initialized."""
    global _simulation_app, _is_initialized
    
    if _is_initialized and _simulation_app is not None:
        logger.info("Closing SimulationApp")
        _simulation_app.close()
        _simulation_app = None
        _is_initialized = False


def require_simulation_app() -> None:
    """
    Decorator or context check to ensure SimulationApp is initialized.
    
    Raises:
        RuntimeError: If SimulationApp is not initialized
    """
    if not _is_initialized:
        raise RuntimeError(
            "SimulationApp is not initialized. "
            "Call get_simulation_app() before using simulation features, or use "
            "the planning-only functions which don't require Isaac Sim."
        )
