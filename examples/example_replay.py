import isaacsim
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width": 1920,
    "height": 1080
})

import omni.kit.actions.core

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
    scene_pose = [0, 0, -0.14, 1, 0, 0, 0]
    scene_result.scene.set_pose(scene_pose)
    # set object poses
    for obj in scene_result.valid_objects:
        obj_pose = scene_result.scene.get_object_matrix(obj)
        obj_pose = single_axis_self_rotation(obj_pose, 'z', np.pi)
        obj_pose = matrix_to_pose(obj_pose)
        obj.set_pose(pose_multiply(scene_pose, obj_pose))
    # set scene pose
    return scene_result


def demo_replay_only():
    """Demo replay functionality with existing results."""
    print("=== Replay Demo (using existing results) ===")
    
    # Create task for replay
    object = create_7221_object()
    scene_path = "/home/xinhai/Documents/automoma/output/test/kitchen_0919/scene_10_seed_10"
    scene_result = load_scene(scene_path, [object])
    
    task = TaskDescription(
        robot=RobotDescription("summit_franka", "assets/robot/summit_franka/summit_franka.yml"),
        object=object,
        scene=scene_result.scene,
        task_type=TaskType.ARTICULATE,
    )
    
    # Create replay pipeline directly
    replay_pipeline = ReplayPipeline(task, simulation_app)
    
    # Original object
    replay_pipeline.replayer.set_deactivate_prims("StaticCategoryFactory_Microwave_7221")
    
    # Ceiling for visualization
    # 1. "exterior" for the walls
    # 2. "ceiling" for the ceiling
    # 3. "Ceiling" for the light
    replay_pipeline.replayer.set_deactivate_prims("exterior")
    replay_pipeline.replayer.set_deactivate_prims("ceiling")
    replay_pipeline.replayer.set_deactivate_prims("Ceiling")

    action_registry = omni.kit.actions.core.get_action_registry()

    action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_rig")
    action.execute(lighting_mode=2)

    # action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_stage")
    # action.execute()

    # action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
    # action.execute()

    
    # Replay existing results for grasp 0
    grasp_id = 1
    
    print("Choose replay mode:")
    print("1. Replay IK solutions")
    print("2. Replay trajectories")  
    print("3. Replay filtered trajectories")
    print("4. Replay AKR trajectories")
    
    # For demo, just replay IK
    replay_pipeline.replay_ik(grasp_id=grasp_id)
    replay_pipeline.replay_traj(grasp_id=grasp_id)
    
    # Close when done
    replay_pipeline.close()


if __name__ == "__main__":
    # Choose which demo to run
    # test()  # Original pipeline test
    # test_with_replay()  # Test with replay
    demo_replay_only()  # Replay only demo
