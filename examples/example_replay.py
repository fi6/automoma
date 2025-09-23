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
    scene_pose = [0, 0, -0.13, 1, 0, 0, 0]
    scene_result.scene.set_pose(scene_pose)
    # set object poses
    for obj in scene_result.valid_objects:
        obj_pose = scene_result.scene.get_object_matrix(obj)
        obj_pose = single_axis_self_rotation(obj_pose, 'z', np.pi)
        obj_pose = matrix_to_pose(obj_pose)
        obj.set_pose(obj_pose)
    # set scene pose
    return scene_result


def demo_replay_only():
    """Demo replay functionality with existing results."""
    print("=== Replay Demo (using existing results) ===")
    
    # Create task for replay
    object = create_7221_object()
    scene_path = "/home/xinhai/Documents/automoma/output/infinigen_scene_10/scene_1_seed_1"
    scene_result = load_scene(scene_path, [object])
    
    task = TaskDescription(
        robot=RobotDescription("summit_franka_develop", "assets/robot/summit_franka/summit_franka.yml"),
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
    grasp_id = 2
    
    print("Choose replay mode:")
    print("1. Replay IK solutions")
    print("2. Replay trajectories")  
    print("3. Replay filtered trajectories")
    print("4. Replay AKR trajectories")
    print("5. Record trajectory data (NEW)")
    print("6. Evaluate policy (NEW - requires policy model)")
    
    # For demo, demonstrate different functionalities
    print("\n=== Running IK Replay ===")
    replay_pipeline.replay_ik(grasp_id=grasp_id)
    
    print("\n=== Running Trajectory Replay ===")
    replay_pipeline.replay_traj(grasp_id=grasp_id)
    
    print("\n=== Running Trajectory Recording (NEW) ===")
    # Record trajectory data for training/evaluation
    
    # for grasp_id in [0, 12, 14]:
    #     camera_results = replay_pipeline.replay_traj_record(
    #         grasp_id=grasp_id, 
    #         num_episodes=5  # Record 3 episodes for demo
    #     )
    # replay_pipeline.replayer.isaacsim_step(step=-1, render=True)  # Keep running until window is closed
    
    # print(f"Successfully recorded {len(camera_results)} episodes!")
    
    # # Show data structure
    # if camera_results:
    #     result = camera_results[0]
    #     print(f"Sample data structure:")
    #     print(f"  - Timesteps: {result.env_info.get('num_timesteps', 'Unknown')}")
    #     print(f"  - Robot: {result.env_info.get('robot_name', 'Unknown')}")
    #     print(f"  - Object: {result.env_info.get('object_id', 'Unknown')}")
    #     print(f"  - Has joint data: {bool(result.obs.get('joint', {}))}")
    #     print(f"  - Has RGB data: {bool(result.obs.get('rgb', {}))}")
    #     print(f"  - Has depth data: {bool(result.obs.get('depth', {}))}")
    
    # print("\n=== Trajectory Evaluation Demo (NEW) ===")
    
    # Close when done
    replay_pipeline.close()


if __name__ == "__main__":
    # Choose which demo to run
    # test()  # Original pipeline test
    # test_with_replay()  # Test with replay
    demo_replay_only()  # Replay only demo
