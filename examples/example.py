from automoma.models.object import ObjectDescription
from automoma.models.robot import RobotDescription
from automoma.models.task import TaskDescription, TaskType
from automoma.pipeline import GraspPipeline, ScenePipeline, TrajectoryPipeline, InfinigenScenePipeline, AOGraspPipeline


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


def test():
    object = ObjectDescription(
            asset_type="Dishwasher",
            asset_id="11622",
            scale=0.6,
            urdf_path="assets/object/Dishwasher/11622/mobility.urdf",
        )
    objects = [object]
    scene_pipeline = InfinigenScenePipeline()
    # scene, valid_objects = scene_pipeline.generate_scene(objects, seed=100)
    scene_result = scene_pipeline.load_scene("/home/xinhai/Documents/automoma/third_party/infinigen/output/kitchen/v1_seed100_1756884596", objects)
    for object in scene_result.valid_objects:
        print(object.urdf_path)
        pipeline = AOGraspPipeline()
        grasps = pipeline.generate_grasps(object, 10)
        for grasp in grasps:
            print(grasp)
            task = TaskDescription(
                robot=RobotDescription("assets/robot/summit_franka/summit_franka.yml"),
                object=object,
                scene=scene_result.scene,
                grasp_pose=grasp,
                task_type=TaskType.ARTICULATE,
            )
            # trajectory_pipeline = TrajectoryPipeline(task)
            # trajectory_pipeline.plan_trajectory()
            
if __name__ == "__main__":
    test()