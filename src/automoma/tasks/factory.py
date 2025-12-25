"""Task factory for creating task instances."""

from typing import Dict, Any, Optional, Type
from automoma.core.types import TaskType
from automoma.core.registry import TASK_REGISTRY, register_task


class TaskFactory:
    """Factory for creating task instances."""
    
    @staticmethod
    def create(
        task_type: TaskType,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Create a task instance based on task type.
        
        Args:
            task_type: Type of task to create
            config: Optional configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Task instance
        """
        from automoma.tasks.pick_place import PickPlaceTask
        from automoma.tasks.reach_open import ReachOpenTask
        
        if config is None:
            config = {}
        
        task_mapping = {
            TaskType.PICK_PLACE: PickPlaceTask,
            TaskType.REACH_OPEN: ReachOpenTask,
            TaskType.PICK: PickPlaceTask,
            TaskType.PLACE: PickPlaceTask,
            TaskType.REACH: ReachOpenTask,
            TaskType.OPEN: ReachOpenTask,
        }
        
        if task_type not in task_mapping:
            raise ValueError(f"Unknown task type: {task_type}")
        
        task_class = task_mapping[task_type]
        return task_class(config, **kwargs)
    
    @staticmethod
    def register(task_type: TaskType, task_class: Type) -> None:
        """
        Register a new task class.
        
        Args:
            task_type: Task type to register
            task_class: Task class to register
        """
        TASK_REGISTRY.register(task_type.name, task_class)
    
    @staticmethod
    def get_available_tasks():
        """Get list of available task types."""
        return [TaskType.PICK_PLACE, TaskType.REACH_OPEN, TaskType.PICK, TaskType.PLACE, TaskType.REACH, TaskType.OPEN]


def create_task(
    task_type: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    Convenience function to create a task.
    
    Args:
        task_type: Task type as string (e.g., "pick_place", "reach_open")
        config: Optional configuration
        **kwargs: Additional arguments
        
    Returns:
        Task instance
    """
    task_type_enum = TaskType[task_type.upper()]
    return TaskFactory.create(task_type_enum, config, **kwargs)
