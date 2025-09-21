from automoma.models.object import ObjectDescription
from automoma.models.robot import RobotDescription
from automoma.models.task import TaskDescription, TaskType
from automoma.pipeline import GraspPipeline, ScenePipeline, TrajectoryPipeline, InfinigenScenePipeline, AOGraspPipeline, ReplayPipeline
from automoma.utils.transform import single_axis_self_rotation, matrix_to_pose
import numpy as np


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


def create_7221_object():
    """Create a 7221 microwave object."""
    object = ObjectDescription(
        asset_type="Microwave",
        asset_id="7221",
        scale=0.3562990018302636,
        urdf_path="assets/object/Microwave/7221/7221_0_scaling.urdf",
    )
    object.set_handle_link("link_0")
    return object


def load_scene(scene_path: str, objects: list):
    """Load scene from path."""
    
    from cuakr.utils.math import pose_multiply
    
    scene_pipeline = InfinigenScenePipeline()
    scene_result = scene_pipeline.load_scene(scene_path, objects)
    scene_pose = [0, 0, -0.12, 1, 0, 0, 0]
    scene_result.scene.set_pose(scene_pose)
    # set object poses
    for obj in scene_result.valid_objects:
        obj_pose = scene_result.scene.get_object_matrix(obj)
        obj_pose = single_axis_self_rotation(obj_pose, 'z', np.pi)
        obj_pose = matrix_to_pose(obj_pose)
        obj.set_pose(pose_multiply(scene_pose, obj_pose))
    # set scene pose
    return scene_result


def test():
    """Simple pipeline test with 7221 microwave."""
    print("=== AKR Pipeline Test ===")

    # Create object
    object = create_7221_object()

    # Load scene
    scene_path = "/home/xinhai/Documents/automoma/output/test/kitchen_0919/scene_4_seed_4"
    scene_result = load_scene(scene_path, [object])

    # Generate grasps
    pipeline = AOGraspPipeline()
    grasps = pipeline.generate_grasps(object, 20)
    
    # Create task
    task = TaskDescription(
        robot=RobotDescription("summit_franka", "assets/robot/summit_franka/summit_franka.yml"),
        object=object,
        scene=scene_result.scene,
        task_type=TaskType.ARTICULATE,
    )
    print("=== Task is created ===")
    
    trajectory_pipeline = TrajectoryPipeline(task)
    
    print("=== Trajectory pipeline is created ===")

    # Process each grasp
    for i, grasp in enumerate(grasps):
        print(f"\nProcessing grasp {i}")

        # Update task with current grasp
        task.update_grasp(grasp)
        print("=== Task updated with new grasp. ===")
        
        # Run pipeline
        trajectory_pipeline.load_akr_robot(f"assets/object/Microwave/7221/summit_franka_7221_0_grasp_{i:04d}.yml")
        print("=== AKR robot loaded. ===")

        trajectory_pipeline.plan_ik()
        print("=== IK planning completed. ===")

        trajectory_pipeline.plan_traj()
        print("=== Trajectory planning completed. ===")
        
        trajectory_pipeline.filter_traj()
        print("=== Trajectory filtering completed. ===")
        
        trajectory_pipeline.save_results(grasp_id=i)
        print("=== Results saved. ===")

        print(f"Completed grasp {i}")



if __name__ == "__main__":
    # Choose which demo to run
    test()  # Original pipeline test
    # test_with_replay()  # Test with replay
    # demo_replay_only()  # Replay only demo
