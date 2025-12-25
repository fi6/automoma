"""Task factory for creating task instances."""

import logging
from typing import Dict, Any, Optional, Type

from automoma.core.types import TaskType
from automoma.core.config_loader import Config
from automoma.tasks.base_task import BaseTask


logger = logging.getLogger(__name__)


# Task registry
_TASK_REGISTRY: Dict[str, Type[BaseTask]] = {}


def register_task(name: str):
    """Decorator to register a task class."""
    def decorator(cls: Type[BaseTask]) -> Type[BaseTask]:
        _TASK_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_task_class(task_name: str) -> Optional[Type[BaseTask]]:
    """Get task class by name."""
    return _TASK_REGISTRY.get(task_name.lower())


def list_available_tasks():
    """List all registered tasks."""
    return list(_TASK_REGISTRY.keys())


# Register built-in tasks
def _register_builtin_tasks():
    """Register all built-in task implementations."""
    from automoma.tasks.open_task import OpenTask, ReachOpenTask
    from automoma.tasks.pick_place_task import PickTask, PlaceTask, PickPlaceTask
    
    _TASK_REGISTRY["open"] = OpenTask
    _TASK_REGISTRY["reach_open"] = ReachOpenTask
    _TASK_REGISTRY["pick"] = PickTask
    _TASK_REGISTRY["place"] = PlaceTask
    _TASK_REGISTRY["pick_place"] = PickPlaceTask
    
    # Aliases
    _TASK_REGISTRY["multi_object_open"] = OpenTask
    _TASK_REGISTRY["single_object_open_test"] = OpenTask


class TaskFactory:
    """
    Factory for creating task instances from configuration.
    
    Usage:
        cfg = load_config("multi_object_open")
        task = TaskFactory.create(cfg)
    """
    
    @staticmethod
    def create(cfg: Config, task_name: Optional[str] = None) -> BaseTask:
        """
        Create a task instance from configuration.
        
        Args:
            cfg: Configuration object
            task_name: Optional task name override
            
        Returns:
            Task instance
        """
        # Ensure tasks are registered
        if not _TASK_REGISTRY:
            _register_builtin_tasks()
        
        # Get task name from config or parameter
        if task_name is None:
            if cfg.info_cfg and cfg.info_cfg.task:
                task_name = cfg.info_cfg.task
            else:
                raise ValueError("Task name not specified in config or parameter")
        
        # Get task class
        task_class = get_task_class(task_name)
        if task_class is None:
            available = list_available_tasks()
            raise ValueError(f"Unknown task: {task_name}. Available: {available}")
        
        logger.info(f"Creating task: {task_name}")
        return task_class(cfg)
    
    @staticmethod
    def create_from_exp(exp_name: str, project_root: str = None) -> BaseTask:
        """
        Create a task from experiment name.
        
        Args:
            exp_name: Experiment name (e.g., "multi_object_open")
            project_root: Optional project root path
            
        Returns:
            Task instance
        """
        from automoma.core.config_loader import load_config
        
        cfg = load_config(exp_name, project_root)
        return TaskFactory.create(cfg)


def create_task(cfg: Config, task_name: Optional[str] = None) -> BaseTask:
    """Convenience function to create a task."""
    return TaskFactory.create(cfg, task_name)


def create_task_from_exp(exp_name: str, project_root: str = None) -> BaseTask:
    """Convenience function to create a task from experiment name."""
    return TaskFactory.create_from_exp(exp_name, project_root)
