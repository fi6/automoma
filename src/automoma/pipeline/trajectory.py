from automoma.models.task import TaskDescription, TaskType


class TrajectoryPipeline:
    akr_robot: None  # curobo robot instance

    def __init__(self, task: TaskDescription):
        self.task = task
        self.akr_robot = self.build_akr_robot()

    def build_akr_robot(self):
        self.task.robot
        self.task.object
        self.task.object_scale
        self.task.grasp_pose
        raise NotImplementedError("build_akr_robot is not implemented yet.")

    def generate_robot_init_state(self):
        self.task.scene
        self.task.object_scene_pose
        self.task.grasp_pose
        raise NotImplementedError("generate_robot_init_state is not implemented yet.")

    def generate_robot_goal_state(self):
        self.task.scene
        self.task.object_scene_pose
        self.task.grasp_pose
        if self.task.task_type == TaskType.PICKPLACE:
            pass
        elif self.task.task_type == TaskType.ARTICULATE:
            pass
        raise NotImplementedError("generate_robot_goal_state is not implemented yet.")

    def plan_akr_trajectory(self):
        # use this for both pickplace and articulate tasks
        self.akr_robot
        raise NotImplementedError("plan_akr_trajectory is not implemented yet.")

    def plan_trajectories(self):
        # plan approach trajectory(if needed)
        self.task.robot
        raise NotImplementedError("plan_trajectory is not implemented yet.")
