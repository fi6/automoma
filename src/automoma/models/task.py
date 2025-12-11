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
        grasp_pose: np.ndarray = None,
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
        angle = ao_grasp_task_generator("assets/grasp/7221")
        self.start = {
            "pose": self.scene.get_object_pose(self.object),
            "angle": 0.0,
        }
        self.goal = {
            "angle": angle
            # "angle": [0.80, 1.00, 1.25, 1.40, 1.57],  # radians
            # "angle_bound": [0.78, 1.57],  # radians
        }
        self.update_goal()

    def update_grasp(self, grasp_pose: np.ndarray):
        self.grasp_pose = grasp_pose
        self.update_goal()

    def update_goal(self, num_angles: int = 3):
        if self.goal.get("angle") is not None:
            return
        if self.task_type == TaskType.ARTICULATE:
            angle_bound = self.goal["angle_bound"]
            self.goal["angle"] = list(np.random.uniform(angle_bound[0], angle_bound[1], num_angles))
            print(f"Updated goal angles: {self.goal['angle']}")
        else:
            raise NotImplementedError("update_goal is only implemented for ARTICULATE tasks.")

    @classmethod
    def from_yaml(cls, task_yaml_path: str) -> "TaskDescription":
        raise NotImplementedError("from_yaml is not implemented yet.")


# def ao_grasp_task_generator_7221(object_grasp_folder: str):
#     import os

#     # object_grasp_folder: assets/grasp/7221
#     open_angles = []
#     for subdir in os.listdir(object_grasp_folder):
#         # subdir: assets/grasp/7221/0
#         object_init_state = os.path.join(object_grasp_folder, subdir, "init_state.npz")
#         # print(object_init_state)
#         if not os.path.exists(object_init_state):
#             continue
#         with np.load(object_init_state, allow_pickle=True) as data:
#             # print(data['data'].item()['object']['qpos'])
#             open_angles.append(data["data"].item()["object"]["qpos"][1])
#         # select open angles
#     open_angles = sorted(open_angles, reverse=True)
#     return [1.57, *[open_angles[x] for x in [1, 4]]]


def ao_grasp_task_generator(object_grasp_folder: str):
    import os

    # object_grasp_folder: assets/grasp/7221
    open_angles = []
    for subdir in os.listdir(object_grasp_folder):
        # subdir: assets/grasp/7221/0
        object_init_state = os.path.join(object_grasp_folder, subdir, "init_state.npz")
        # print(object_init_state)
        if not os.path.exists(object_init_state):
            continue
        with np.load(object_init_state, allow_pickle=True) as data:
            # print(data['data'].item()['object']['qpos'])
            open_angles.append(data["data"].item()["object"]["qpos"][1])
        # select open angles
    open_angles = sorted(open_angles, reverse=True)
    print(f"Available open angles: {open_angles}")
    return [*[open_angles[x] for x in [0, 5]]]
