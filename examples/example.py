from automoma.models.object import ObjectDescription
from automoma.models.robot import RobotDescription
from automoma.models.task import TaskDescription, TaskType
from automoma.pipeline import GraspPipeline, ScenePipeline, TrajectoryPipeline


def main():
    object = ObjectDescription("7221")
    scene_pipeline = ScenePipeline()
    scene = scene_pipeline.generate_scene(object)
    grasp_pipeline = GraspPipeline()
    grasps = grasp_pipeline.generate_grasps(object, count=5)
    for grasp in grasps:
        task = TaskDescription(
            robot=RobotDescription("franka.yaml"),
            object=object,
            scene=scene,
            object_scene_pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            grasp_pose=grasp,
            task_type=TaskType.ARTICULATE,
        )
        trajectory_pipeline = TrajectoryPipeline(task)
        trajectory_pipeline.plan_trajector()
