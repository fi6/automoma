from automoma.models.object import ObjectDescription
from automoma.models.robot import RobotDescription
from automoma.models.scene import SceneDescription
import numpy as np
from enum import Enum


class TaskType(Enum):
    PICKPLACE = "pickplace"
    ARTICULATE = "articulate"


class TaskDescription:
    task_type: TaskType
    robot: RobotDescription
    scene: SceneDescription
    object: ObjectDescription
    grasp_pose: np.ndarray
    object_scene_pose: np.ndarray
    object_scale: float | list[float] = 1.0

    def __init__(
        self,
        task_type: TaskType,
        robot: RobotDescription,
        scene: SceneDescription,
        object: ObjectDescription,
        grasp_pose: np.ndarray,
        object_scene_pose: np.ndarray,
        object_scale: float | list[float] = 1.0,
    ):
        self.task_type = task_type
        self.robot = robot
        self.scene = scene
        self.object = object
        self.grasp_pose = grasp_pose
        self.object_scene_pose = object_scene_pose
        self.object_scale = object_scale

    @classmethod
    def from_yaml(cls, task_yaml_path: str) -> "TaskDescription":
        raise NotImplementedError("from_yaml is not implemented yet.")
