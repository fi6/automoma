from automoma.models.object import ObjectDescription
from automoma.models.robot import RobotDescription
from automoma.models.task import TaskDescription, TaskType
from automoma.pipeline import GraspPipeline, ScenePipeline, TrajectoryPipeline


def main():
    objects = [ObjectDescription("7221"), ObjectDescription("11622")]
    scene_pipeline = ScenePipeline()
    scene, updated_objects = scene_pipeline.generate_scene(objects)
    grasp_pipeline = GraspPipeline()
    for object in updated_objects:
        grasps = grasp_pipeline.generate_grasps(object, count=5)
        for grasp in grasps:
            task = TaskDescription(
                robot=RobotDescription("franka.yaml"),
                object=object,
                scene=scene,
                grasp_pose=grasp,
                task_type=TaskType.ARTICULATE,
            )
            trajectory_pipeline = TrajectoryPipeline(task)
            trajectory_pipeline.plan_trajectory()
