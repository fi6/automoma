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

    def __init__(
        self,
        task_type: TaskType,
        robot: RobotDescription,
        scene: SceneDescription,
        object: ObjectDescription,
        grasp_pose: np.ndarray=None,
    ):
        self.task_type = task_type
        self.robot = robot
        self.scene = scene
        self.object = object
        self.grasp_pose = grasp_pose

        self.init_task()

    def init_task(self):
        if self.task_type == TaskType.PICKPLACE:
            self.init_task_pickplace()
        elif self.task_type == TaskType.ARTICULATE:
            self.init_task_articulate()

    def init_task_pickplace(self):
        self.start = {
            "pose": self.scene.get_object_pose(self.object),
        }
        self.goal = {
            "pose": [[0, 0, 0, 1, 0, 0, 0]],
        }
        
    def init_task_articulate(self):
        self.start = {
            "pose": self.scene.get_object_pose(self.object),
            "angle": 0.0,
        }
        self.goal = {
            # "angle": [0.80, 1.00, 1.25, 1.40, 1.57],  # radians
            "angle": [1.00, 1.20, 1.57],  # radians
        }
        
    def update_grasp(self, grasp_pose: np.ndarray):
        self.grasp_pose = grasp_pose

    @classmethod
    def from_yaml(cls, task_yaml_path: str) -> "TaskDescription":
        raise NotImplementedError("from_yaml is not implemented yet.")
