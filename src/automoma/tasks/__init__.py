"""Tasks module for AutoMoMa framework."""

from automoma.tasks.base_task import BaseTask, TaskResult, StageResult
from automoma.tasks.factory import TaskFactory, create_task, create_task_from_exp
from automoma.tasks.open_task import OpenTask, ReachOpenTask
from automoma.tasks.pick_place_task import PickTask, PlaceTask, PickPlaceTask
from automoma.tasks.reach_task import ReachTask

__all__ = [
    # Base classes
    "BaseTask",
    "TaskResult",
    "StageResult",
    # Factory
    "TaskFactory",
    "create_task",
    "create_task_from_exp",
    # Tasks
    "OpenTask",
    "ReachOpenTask",
    "ReachTask",
    "PickTask",
    "PlaceTask",
    "PickPlaceTask",
]
