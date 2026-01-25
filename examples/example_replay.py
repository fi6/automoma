import isaacsim
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": 1920, "height": 1080})

import omni.kit.actions.core

from automoma.models.object import ObjectDescription
from automoma.models.robot import RobotDescription
from automoma.models.task import TaskDescription, TaskType
from automoma.pipeline import (
    GraspPipeline,
    ScenePipeline,
    TrajectoryPipeline,
    InfinigenScenePipeline,
    AOGraspPipeline,
    ReplayPipeline,
)
from automoma.utils.transform import single_axis_self_rotation, matrix_to_pose
import numpy as np
from typing import Dict, List, Any


# ============================================================================
# HYPERPARAMETERS - Modify these for different scenarios
# ============================================================================
OBJECT_ID = "7221"  # Options: 7221, 11622, 103634, 46197, 10944, 101773
SCENE_PATH = "assets/scene/infinigen/kitchen_1130/scene_0_seed_0"
OUTPUT_BASE_DIR = "output/collect_1205/traj"
ROBOT_NAME = "summit_franka"
GRASP_ID = 0
# ============================================================================


# Object configuration mapping with asset type, scale, and URDF paths
OBJECT_CONFIG_MAP = {
    "7221": {
        "asset_type": "Microwave",
        "asset_id": "7221",
        "scale": 0.3562990018302636,
        "urdf_path": "assets/object/Microwave/7221/7221_0_scaling.urdf",
        "handle_link": "link_0",
        "akr_path_template": "assets/object/Microwave/{object_id}/{robot_name}_{object_id}_0_grasp_{grasp_id:04d}.yml",
    },
    "11622": {
        "asset_type": "Dishwasher",
        "asset_id": "11622",
        "scale": 0.6446,
        "urdf_path": "assets/object/Dishwasher/11622/11622_0_scaling.urdf",
        "handle_link": "link_0",
        "akr_path_template": "assets/object/Dishwasher/{object_id}/{robot_name}_{object_id}_0_grasp_{grasp_id:04d}.yml",
    },
    "103634": {
        "asset_type": "TrashCan",
        "asset_id": "103634",
        "scale": 0.48385408192053975,
        "urdf_path": "assets/object/TrashCan/103634/103634_0_scaling.urdf",
        "handle_link": "link_0",
        "akr_path_template": "assets/object/TrashCan/{object_id}/{robot_name}_{object_id}_0_grasp_{grasp_id:04d}.yml",
    },
    "46197": {
        "asset_type": "StorageFurniture",
        "asset_id": "46197",
        "scale": 0.5113198146209817,
        "urdf_path": "assets/object/StorageFurniture/46197/46197_0_scaling.urdf",
        "handle_link": "link_0",
        "akr_path_template": "assets/object/StorageFurniture/{object_id}/{robot_name}_{object_id}_0_grasp_{grasp_id:04d}.yml",
    },
    "10944": {
        "asset_type": "Refrigerator",
        "asset_id": "10944",
        "scale": 0.900,
        "urdf_path": "assets/object/Refrigerator/10944/10944_0_scaling.urdf",
        "handle_link": "link_0",
        "akr_path_template": "assets/object/Refrigerator/{object_id}/{robot_name}_{object_id}_0_grasp_{grasp_id:04d}.yml",
    },
    "101773": {
        "asset_type": "Oven",
        "asset_id": "101773",
        "scale": 0.7231779244463762,
        "urdf_path": "assets/object/Oven/101773/101773_0_scaling.urdf",
        "handle_link": "link_0",
        "akr_path_template": "assets/object/Oven/{object_id}/{robot_name}_{object_id}_0_grasp_{grasp_id:04d}.yml",
    },
}

# Object prim naming mapping for deactivation
# Maps object_id to the prim name used in Isaac Sim scenes
OBJECT_PRIM_MAP = {
    "7221": "StaticCategoryFactory_Microwave_7221",
    "11622": "StaticCategoryFactory_Dishwasher_11622",
    "103634": "StaticCategoryFactory_TrashCan_103634",
    "46197": "StaticCategoryFactory_Cabinet_46197",
    "10944": "StaticCategoryFactory_Refrigerator_10944",
    "101773": "StaticCategoryFactory_Oven_101773",
}

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

def get_object_config(object_id: str) -> Dict[str, Any]:
    """Get the configuration for an object by ID."""
    if object_id not in OBJECT_CONFIG_MAP:
        raise ValueError(f"Unknown object ID: {object_id}. Available objects: {list(OBJECT_CONFIG_MAP.keys())}")
    return OBJECT_CONFIG_MAP[object_id]
def create_object(object_id: str) -> ObjectDescription:
    """Create an object description using the configuration map."""
    config = get_object_config(object_id)
    obj = ObjectDescription(
        asset_type=config["asset_type"],
        asset_id=config["asset_id"],
        scale=config["scale"],
        urdf_path=config["urdf_path"],
    )
    obj.set_handle_link(config["handle_link"])
    return obj


def load_scene(scene_path: str, objects: list):
    """Load scene from path."""

    from cuakr.utils.math import pose_multiply

    scene_pipeline = InfinigenScenePipeline()
    scene_result = scene_pipeline.load_scene(scene_path, objects)
    scene_pose = [0, 0, -0.13, 1, 0, 0, 0]
    scene_result.scene.set_pose(scene_pose)
    # set object poses
    for obj in scene_result.valid_objects:
        print("object_pose before:", scene_result.scene.get_object_pose(obj))
        obj_pose = scene_result.scene.get_object_matrix(obj)
        obj_pose = single_axis_self_rotation(obj_pose, "z", np.pi)
        obj_pose = matrix_to_pose(obj_pose)
        print("object_pose after:", obj_pose)
        obj.set_pose(obj_pose)
    # set scene pose
    return scene_result


def demo_replay_only(object_id, scene_path, output_base_dir, robot_name=ROBOT_NAME, grasp_id=GRASP_ID):
    """Demo replay functionality with existing results.
    
    Args:
        object_id: ID of the object to replay (e.g., "7221", "11622")
        scene_path: Path to the scene directory
        output_base_dir: Base directory for output results
        robot_name: Name of the robot to use (default: summit_franka)
        grasp_id: Grasp ID to replay (default: 0)
    """
    print("=== Replay Demo (using existing results) ===")
    print(f"Object ID: {object_id}")
    print(f"Scene Path: {scene_path}")
    print(f"Output Dir: {output_base_dir}")
    print(f"Robot: {robot_name}")
    print(f"Grasp ID: {grasp_id}")
    print()

    # Create task for replay
    object = create_object(object_id)
    scene_result = load_scene(scene_path, [object])

    task = TaskDescription(
        robot=RobotDescription(robot_name, f"assets/robot/{robot_name}/{robot_name}.yml"),
        object=object,
        scene=scene_result.scene,
        task_type=TaskType.ARTICULATE,
    )

    # Create replay pipeline directly
    replay_pipeline = ReplayPipeline(task, simulation_app, output_base_dir=output_base_dir)

    # Deactivate the object prim dynamically based on object_id
    if object_id in OBJECT_PRIM_MAP:
        object_prim = OBJECT_PRIM_MAP[object_id]
        print(f"Deactivating object prim: {object_prim}")
        replay_pipeline.replayer.set_deactivate_prims(object_prim)
    else:
        print(f"Warning: Object prim mapping not found for object_id {object_id}")

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

    print("Choose replay mode:")
    print("1. Replay IK solutions")
    replay_pipeline.replay_ik(grasp_id=grasp_id)
    print("2. Replay trajectories")
    # replay_pipeline.replay_traj(grasp_id=grasp_id)
    print("3. Replay filtered trajectories")
    print("4. Replay AKR trajectories")
    print("5. Record trajectory data (NEW)")
    print("6. Evaluate policy (NEW - requires policy model)")

    # For demo, demonstrate different functionalities
    print(f"\n=== Running Filtered Trajectory Replay (Grasp {grasp_id}) ===")
    replay_pipeline.replay_filtered_traj(grasp_id=grasp_id)
    

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


def demo_replay_fixed_base():
    """Demo replay functionality with fixed base robot.
    
    This function is kept for reference but uses the new hyperparameters.
    """
    print("=== Replay Demo (Fixed Base - using existing results) ===")
    
    # Use hyperparameters but hardcode for fixed base variant
    object_id = OBJECT_ID
    robot_name = "summit_franka_fixed_base"
    scene_path = SCENE_PATH
    output_base_dir = OUTPUT_BASE_DIR
    grasp_id = GRASP_ID

    # Create task for replay
    object = create_object(object_id)
    scene_result = load_scene(scene_path, [object])

    task = TaskDescription(
        robot=RobotDescription(robot_name, f"assets/robot/{robot_name}/{robot_name}.yml"),
        object=object,
        scene=scene_result.scene,
        task_type=TaskType.ARTICULATE,
    )

    # Create replay pipeline directly
    replay_pipeline = ReplayPipeline(task, simulation_app, output_base_dir=output_base_dir)

    # Deactivate the object prim dynamically based on object_id
    if object_id in OBJECT_PRIM_MAP:
        object_prim = OBJECT_PRIM_MAP[object_id]
        print(f"Deactivating object prim: {object_prim}")
        replay_pipeline.replayer.set_deactivate_prims(object_prim)

    # Ceiling for visualization
    replay_pipeline.replayer.set_deactivate_prims("exterior")
    replay_pipeline.replayer.set_deactivate_prims("ceiling")
    replay_pipeline.replayer.set_deactivate_prims("Ceiling")

    action_registry = omni.kit.actions.core.get_action_registry()
    action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_rig")
    action.execute(lighting_mode=2)

    print(f"\n=== Running IK Replay (Grasp {grasp_id}) ===")
    replay_pipeline.replay_ik(grasp_id=grasp_id)

    print(f"\n=== Running Filtered Trajectory Replay (Grasp {grasp_id}) ===")
    replay_pipeline.replay_filtered_traj(grasp_id=grasp_id)

    # Close when done
    replay_pipeline.close()


if __name__ == "__main__":
    # Use hyperparameters defined at the top of the file
    try:
        demo_replay_only(
            object_id=OBJECT_ID,
            scene_path=SCENE_PATH,
            output_base_dir=OUTPUT_BASE_DIR,
            robot_name=ROBOT_NAME,
            grasp_id=GRASP_ID
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
