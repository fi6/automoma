"""
Example script demonstrating the new replay trajectory recording and evaluation functionality.
This script shows how to use the enhanced ReplayPipeline for data collection and policy evaluation.
"""

import isaacsim
from omni.isaac.kit import SimulationApp

# Initialize Isaac Sim
simulation_app = SimulationApp({
    "headless": False,
    "width": 1920,
    "height": 1080
})

import omni.kit.actions.core

from automoma.models.object import ObjectDescription
from automoma.models.robot import RobotDescription
from automoma.models.task import TaskDescription, TaskType
from automoma.pipeline import ReplayPipeline, InfinigenScenePipeline
from automoma.utils.transform import single_axis_self_rotation, matrix_to_pose
from automoma.utils.data_structures import CameraResult, TrajectoryEvaluationResult
import numpy as np
import torch
import os


def create_microwave_object():
    """Create a 7221 microwave object for testing."""
    object = ObjectDescription(
        asset_type="Microwave",
        asset_id="7221",
        scale=0.3562990018302636,
        urdf_path="assets/object/Microwave/7221/7221_0_scaling.urdf",
    )
    object.set_handle_link("link_0")
    return object


def load_test_scene(scene_path: str, objects: list):
    """Load scene for testing."""
    scene_pipeline = InfinigenScenePipeline()
    scene_result = scene_pipeline.load_scene(scene_path, objects)
    scene_pose = [0, 0, -0.13, 1, 0, 0, 0]
    scene_result.scene.set_pose(scene_pose)
    
    # Set object poses
    for obj in scene_result.valid_objects:
        obj_pose = scene_result.scene.get_object_matrix(obj)
        obj_pose = single_axis_self_rotation(obj_pose, 'z', np.pi)
        obj_pose = matrix_to_pose(obj_pose)
        obj.set_pose(obj_pose)
    
    return scene_result


def setup_test_environment():
    """Set up the test environment with lighting and disabled prims."""
    action_registry = omni.kit.actions.core.get_action_registry()
    
    # Set better lighting
    action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_rig")
    action.execute(lighting_mode=2)


def demo_trajectory_recording():
    """Demonstrate trajectory recording functionality."""
    print("=" * 60)
    print("DEMO: Trajectory Recording for Data Collection")
    print("=" * 60)
    
    # Create test task
    object = create_microwave_object()
    scene_path = "/home/xinhai/Documents/automoma/output/test/kitchen_0919/scene_4_seed_4"
    scene_result = load_test_scene(scene_path, [object])
    
    task = TaskDescription(
        robot=RobotDescription("summit_franka_develop", "assets/robot/summit_franka/summit_franka.yml"),
        object=object,
        scene=scene_result.scene,
        task_type=TaskType.ARTICULATE,
    )
    
    # Create replay pipeline
    replay_pipeline = ReplayPipeline(task, simulation_app)
    
    # Set up environment
    replay_pipeline.replayer.set_deactivate_prims("StaticCategoryFactory_Microwave_7221")
    replay_pipeline.replayer.set_deactivate_prims("exterior")
    replay_pipeline.replayer.set_deactivate_prims("ceiling")
    replay_pipeline.replayer.set_deactivate_prims("Ceiling")
    setup_test_environment()
    
    # Record trajectory data
    grasp_id = 0
    print(f"\nRecording trajectory data for grasp {grasp_id}...")
    
    try:
        camera_results = replay_pipeline.replay_traj_record(
            grasp_id=grasp_id,
            num_episodes=5  # Record 5 episodes for testing
        )
        
        print(f"\nSuccessfully recorded {len(camera_results)} episodes!")
        
        # Print summary of recorded data
        for i, result in enumerate(camera_results):
            print(f"Episode {i}:")
            print(f"  - Number of timesteps: {result.env_info.get('num_timesteps', 'Unknown')}")
            print(f"  - Scene ID: {result.env_info.get('scene_id', 'Unknown')}")
            print(f"  - Robot: {result.env_info.get('robot_name', 'Unknown')}")
            print(f"  - Object ID: {result.env_info.get('object_id', 'Unknown')}")
            
            # Check data structure
            has_joint_data = bool(result.obs.get("joint", {}))
            has_eef_data = bool(result.obs.get("eef", []))
            has_rgb_data = bool(result.obs.get("rgb", {}))
            has_depth_data = bool(result.obs.get("depth", {}))
            has_pc_data = bool(result.obs.get("point_cloud", []))
            
            print(f"  - Joint data: {'✓' if has_joint_data else '✗'}")
            print(f"  - End-effector data: {'✓' if has_eef_data else '✗'}")
            print(f"  - RGB data: {'✓' if has_rgb_data else '✗'}")
            print(f"  - Depth data: {'✓' if has_depth_data else '✗'}")
            print(f"  - Point cloud data: {'✓' if has_pc_data else '✗'}")
            
    except Exception as e:
        print(f"Recording failed: {e}")
        import traceback
        traceback.print_exc()
    
    return replay_pipeline


def demo_policy_evaluation(replay_pipeline):
    """Demonstrate policy evaluation functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Policy Evaluation")
    print("=" * 60)
    
    # Create a dummy policy model for demonstration
    class DummyPolicy:
        """Dummy policy for testing evaluation pipeline."""
        def __init__(self):
            self.name = "DummyPolicy"
            
        def predict(self, observation):
            # Return random action for demonstration
            return np.random.randn(7)  # Random 7-DOF action
    
    dummy_policy = DummyPolicy()
    
    # Run policy evaluation
    grasp_id = 0
    print(f"\nEvaluating policy on grasp {grasp_id}...")
    
    try:
        evaluation_results = replay_pipeline.replay_traj_evaluate(
            policy_model=dummy_policy,
            grasp_id=grasp_id,
            num_episodes=3  # Evaluate on 3 episodes for testing
        )
        
        print(f"\nSuccessfully evaluated {len(evaluation_results)} episodes!")
        
        # Print evaluation metrics
        total_position_error = 0
        total_rotation_error = 0
        success_count = 0
        
        for i, result in enumerate(evaluation_results):
            metrics = result.compute_metrics()
            print(f"Episode {i}:")
            print(f"  - Success: {'✓' if result.success else '✗'}")
            print(f"  - Steps: {result.num_steps}")
            print(f"  - Position error: {metrics['position_error']:.4f}")
            print(f"  - Rotation error: {metrics['rotation_error']:.4f}")
            
            total_position_error += metrics['position_error']
            total_rotation_error += metrics['rotation_error']
            if result.success:
                success_count += 1
        
        # Print overall statistics
        if len(evaluation_results) > 0:
            avg_pos_error = total_position_error / len(evaluation_results)
            avg_rot_error = total_rotation_error / len(evaluation_results)
            success_rate = success_count / len(evaluation_results)
            
            print(f"\nOverall Statistics:")
            print(f"  - Success Rate: {success_rate:.2%}")
            print(f"  - Average Position Error: {avg_pos_error:.4f}")
            print(f"  - Average Rotation Error: {avg_rot_error:.4f}")
            
            # Save evaluation results
            output_dir = replay_pipeline._get_output_directory(grasp_id)
            replay_pipeline.save_evaluation_results(evaluation_results, output_dir, grasp_id)
            
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


def demo_data_format_validation():
    """Demonstrate that the data format matches collect_data.py."""
    print("\n" + "=" * 60)
    print("DEMO: Data Format Validation")
    print("=" * 60)
    
    # Check if recorded data files exist
    test_output_dir = "/home/xinhai/Documents/automoma/output/summit_franka/scene_4_seed_4/7221/grasp_0000"
    camera_data_dir = os.path.join(test_output_dir, "camera_data")
    
    if os.path.exists(camera_data_dir):
        import h5py
        
        # List recorded files
        files = [f for f in os.listdir(camera_data_dir) if f.endswith('.hdf5')]
        print(f"Found {len(files)} recorded data files:")
        
        for file in files[:3]:  # Check first 3 files
            filepath = os.path.join(camera_data_dir, file)
            print(f"\nFile: {file}")
            
            try:
                with h5py.File(filepath, "r") as f:
                    print("  Structure:")
                    
                    # Check env_info
                    if "env_info" in f:
                        print("    env_info/")
                        for key in f["env_info"]:
                            print(f"      {key}: {f['env_info'][key][()]}")
                    
                    # Check obs structure
                    if "obs" in f:
                        print("    obs/")
                        for key in f["obs"]:
                            if isinstance(f["obs"][key], h5py.Group):
                                print(f"      {key}/")
                                for subkey in f["obs"][key]:
                                    shape = f["obs"][key][subkey].shape
                                    dtype = f["obs"][key][subkey].dtype
                                    print(f"        {subkey}: {shape} {dtype}")
                            else:
                                shape = f["obs"][key].shape
                                dtype = f["obs"][key].dtype
                                print(f"      {key}: {shape} {dtype}")
                                
            except Exception as e:
                print(f"  Error reading file: {e}")
    else:
        print("No camera data directory found. Run trajectory recording first.")


def main():
    """Main demonstration function."""
    print("Starting Enhanced Replay Pipeline Demo")
    print("This demo shows trajectory recording and policy evaluation capabilities.")
    
    try:
        # Demo 1: Trajectory Recording
        replay_pipeline = demo_trajectory_recording()
        
        # Demo 2: Policy Evaluation
        demo_policy_evaluation(replay_pipeline)
        
        # Demo 3: Data Format Validation
        demo_data_format_validation()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
        # Keep simulation running for manual inspection
        input("\nPress Enter to close the simulation...")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if 'replay_pipeline' in locals():
            replay_pipeline.close()


if __name__ == "__main__":
    main()