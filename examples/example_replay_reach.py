#!/usr/bin/env python3
"""
Replayer for Two-Stage (Reach + Open) Motion Planning Results

This script replays and records camera data from two-stage planning results.
It supports replaying IK solutions and trajectories for both reach and open stages,
either separately or combined.

Usage Examples:
    # Replay reach stage IK
    python examples/example_replay_reach.py --scene-dir assets/scene/infinigen/kitchen_1130/scene_0_seed_0 \
        --plan-dir output/collect_1211/traj --robot-name summit_franka --object-id 7221 \
        --grasp-id 0 --mode replay_ik --stage reach

    # Replay open stage trajectories
    python examples/example_replay_reach.py --scene-dir assets/scene/infinigen/kitchen_1130/scene_0_seed_0 \
        --plan-dir output/collect_1211/traj --robot-name summit_franka --object-id 7221 \
        --grasp-id 0 --mode replay_traj --stage open

    # Replay combined (all stages) filtered trajectories
    python examples/example_replay_reach.py --scene-dir assets/scene/infinigen/kitchen_1130/scene_0_seed_0 \
        --plan-dir output/collect_1211/traj --robot-name summit_franka --object-id 7221 \
        --grasp-id 0 --mode replay_filtered_traj --stage all

    # Record camera data for reach stage (10 episodes)
    python examples/example_replay_reach.py --scene-dir assets/scene/infinigen/kitchen_1130/scene_0_seed_0 \
        --plan-dir output/collect_1211/traj --robot-name summit_franka --object-id 7221 \
        --grasp-id 0 --mode record --stage reach --num-episodes 10

    # Record combined trajectory (reach + open in sequence)
    python examples/example_replay_reach.py --scene-dir assets/scene/infinigen/kitchen_1130/scene_0_seed_0 \
        --plan-dir output/collect_1211/traj --robot-name summit_franka --object-id 7221 \
        --grasp-id 0 --mode record --stage all --num-episodes 5

Stages:
    - reach: Initial poses → start pose (robot-only)
    - open: Start pose → goal pose (with object articulation)
    - all: Combined reach + open in sequence

Modes:
    - replay_ik: Replay IK solutions
    - replay_traj: Replay trajectories
    - replay_filtered_traj: Replay filtered trajectories
    - record: Record camera data for training/evaluation
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Isaac Sim imports
import isaacsim
from omni.isaac.kit import SimulationApp

# Global variable to hold Isaac Sim app instance
simulation_app = None

def initialize_isaac_sim(headless: bool = False):
    """Initialize Isaac Sim application."""
    global simulation_app
    if simulation_app is None:
        simulation_app = SimulationApp({
            "headless": headless,
            "width": 1920,
            "height": 1080
        })
    return simulation_app

def import_modules():
    """Import required modules after Isaac Sim initialization."""
    from automoma.models.object import ObjectDescription
    from automoma.models.robot import RobotDescription
    from automoma.models.task import TaskDescription, TaskType
    from automoma.pipeline import InfinigenScenePipeline
    from automoma.utils.transform import single_axis_self_rotation, matrix_to_pose
    from automoma.utils.replayer import Replayer
    from cuakr.utils.math import pose_multiply
    import omni.kit.actions.core
    
    return (ObjectDescription, RobotDescription, TaskDescription, TaskType,
            InfinigenScenePipeline, Replayer, single_axis_self_rotation, 
            matrix_to_pose, pose_multiply, omni.kit.actions.core)

# ============================================================================
# Object Configuration Map
# ============================================================================
OBJECT_CONFIG_MAP = {
    "7221": {
        "asset_type": "Microwave",
        "asset_id": "7221",
        "scale": 0.3562990018302636,
        "urdf_path": "assets/object/Microwave/7221/7221_0_scaling.urdf",
        "handle_link": "link_0",
    },
    "11622": {
        "asset_type": "Dishwasher",
        "asset_id": "11622",
        "scale": 0.6446,
        "urdf_path": "assets/object/Dishwasher/11622/11622_0_scaling.urdf",
        "handle_link": "link_0",
    },
    "103634": {
        "asset_type": "TrashCan",
        "asset_id": "103634",
        "scale": 0.48385408192053975,
        "urdf_path": "assets/object/TrashCan/103634/103634_0_scaling.urdf",
        "handle_link": "link_0",
    },
    "46197": {
        "asset_type": "StorageFurniture",
        "asset_id": "46197",
        "scale": 0.5113198146209817,
        "urdf_path": "assets/object/StorageFurniture/46197/46197_0_scaling.urdf",
        "handle_link": "link_0",
    },
    "10944": {
        "asset_type": "Refrigerator",
        "asset_id": "10944",
        "scale": 0.900,
        "urdf_path": "assets/object/Refrigerator/10944/10944_0_scaling.urdf",
        "handle_link": "link_0",
    },
    "101773": {
        "asset_type": "Oven",
        "asset_id": "101773",
        "scale": 0.7231779244463762,
        "urdf_path": "assets/object/Oven/101773/101773_0_scaling.urdf",
        "handle_link": "link_0",
    },
}

OBJECT_PRIM_MAP = {
    "7221": "StaticCategoryFactory_Microwave_7221",
    "11622": "StaticCategoryFactory_Dishwasher_11622",
    "103634": "StaticCategoryFactory_TrashCan_103634",
    "46197": "StaticCategoryFactory_Cabinet_46197",
    "10944": "StaticCategoryFactory_Refrigerator_10944",
    "101773": "StaticCategoryFactory_Oven_101773",
}

ROBOT_CONFIG_MAP = {
    "summit_franka": "assets/robot/summit_franka/summit_franka.yml",
    "summit_franka_fixed_base": "assets/robot/summit_franka/summit_franka_fixed_base.yml",
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_object_config(object_id: str) -> Dict[str, Any]:
    """Get object configuration by ID."""
    if object_id not in OBJECT_CONFIG_MAP:
        raise ValueError(f"Unknown object ID: {object_id}")
    return OBJECT_CONFIG_MAP[object_id]

def create_object(object_id: str, ObjectDescription):
    """Create object description."""
    config = get_object_config(object_id)
    obj = ObjectDescription(
        asset_type=config["asset_type"],
        asset_id=config["asset_id"],
        scale=config["scale"],
        urdf_path=config["urdf_path"],
    )
    obj.set_handle_link(config["handle_link"])
    return obj

def load_scene(scene_path: str, objects: List, InfinigenScenePipeline, 
               single_axis_self_rotation, matrix_to_pose):
    """Load scene and set object poses."""
    scene_pipeline = InfinigenScenePipeline()
    scene_result = scene_pipeline.load_scene(scene_path, objects)
    
    # Set scene pose
    scene_pose = [0, 0, -0.13, 1, 0, 0, 0]
    scene_result.scene.set_pose(scene_pose)
    
    # Set object poses
    for obj in scene_result.valid_objects:
        obj_pose = scene_result.scene.get_object_matrix(obj)
        obj_pose = single_axis_self_rotation(obj_pose, 'z', np.pi)
        obj_pose = matrix_to_pose(obj_pose)
        obj.set_pose(obj_pose)
        print(f"Set pose for object {obj.asset_id}: {obj_pose}")
    
    return scene_result

def setup_visualization(action_registry):
    """Setup lighting and visualization."""
    action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_rig")
    action.execute(lighting_mode=2)

def deactivate_scene_prims(replayer, object_id: str):
    """Deactivate unnecessary scene prims."""
    # Common prims to deactivate
    common_prims = ["exterior", "ceiling", "Ceiling"]
    for prim_name in common_prims:
        replayer.set_deactivate_prims(prim_name)
    
    # Object-specific prim
    if object_id in OBJECT_PRIM_MAP:
        replayer.set_deactivate_prims(OBJECT_PRIM_MAP[object_id])

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_stage_data(plan_dir: str, robot_name: str, scene_name: str, 
                   object_id: str, grasp_id: int, stage: str, 
                   data_type: str = "traj_data") -> Dict[str, torch.Tensor]:
    """
    Load data for a specific stage.
    
    Args:
        plan_dir: Base planning directory
        robot_name: Robot name
        scene_name: Scene name
        object_id: Object ID
        grasp_id: Grasp ID
        stage: 'reach', 'open', or 'all'
        data_type: 'ik_data', 'traj_data', or 'filtered_traj_data'
    
    Returns:
        Dictionary containing the loaded data
    """
    # Construct path
    stage_dir = f"{stage}_stage"
    data_path = os.path.join(
        plan_dir, robot_name, scene_name, object_id, 
        f"grasp_{grasp_id:04d}", stage_dir, f"{data_type}.pt"
    )
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading {stage} stage data from: {data_path}")
    data = torch.load(data_path, weights_only=False)
    return data

def load_ik_data(plan_dir: str, robot_name: str, scene_name: str,
                object_id: str, grasp_id: int, stage: str) -> Dict[str, torch.Tensor]:
    """Load IK data for specified stage."""
    return load_stage_data(plan_dir, robot_name, scene_name, object_id, 
                          grasp_id, stage, "ik_data")

def load_traj_data(plan_dir: str, robot_name: str, scene_name: str,
                  object_id: str, grasp_id: int, stage: str,
                  filtered: bool = False) -> Dict[str, torch.Tensor]:
    """Load trajectory data for specified stage."""
    data_type = "filtered_traj_data" if filtered else "traj_data"
    return load_stage_data(plan_dir, robot_name, scene_name, object_id,
                          grasp_id, stage, data_type)

# ============================================================================
# Replay Functions
# ============================================================================

def replay_ik(replayer, plan_dir: str, robot_name: str, scene_name: str,
             object_id: str, grasp_id: int, stage: str):
    """
    Replay IK solutions for specified stage.
    
    Args:
        stage: 'reach', 'open', or 'all'
    """
    print(f"\n{'='*80}")
    print(f"Replaying IK for {stage.upper()} stage (Grasp {grasp_id})")
    print(f"{'='*80}\n")
    
    if stage == "all":
        # Replay both stages sequentially
        print("=== REACH STAGE IK ===")
        ik_data_reach = load_ik_data(plan_dir, robot_name, scene_name, 
                                     object_id, grasp_id, "reach")
        initial_iks = ik_data_reach["initial_iks"]
        print(f"Initial IKs: {initial_iks.shape}")
        replayer.replay_ik(initial_iks, initial_iks, robot_name)
        
        print("\n=== OPEN STAGE IK ===")
        ik_data_open = load_ik_data(plan_dir, robot_name, scene_name,
                                    object_id, grasp_id, "open")
        start_iks = ik_data_open["start_iks"]
        goal_iks = ik_data_open["goal_iks"]
        print(f"Start IKs: {start_iks.shape}, Goal IKs: {goal_iks.shape}")
        replayer.replay_ik(start_iks, goal_iks, robot_name)
    
    elif stage == "reach":
        ik_data = load_ik_data(plan_dir, robot_name, scene_name,
                              object_id, grasp_id, stage)
        initial_iks = ik_data["initial_iks"]
        print(f"Initial IKs: {initial_iks.shape}")
        # For reach stage, replay initial poses (no goal, just show poses)
        replayer.replay_ik(initial_iks, initial_iks, robot_name)
    
    elif stage == "open":
        ik_data = load_ik_data(plan_dir, robot_name, scene_name,
                              object_id, grasp_id, stage)
        start_iks = ik_data["start_iks"]
        goal_iks = ik_data["goal_iks"]
        print(f"Start IKs: {start_iks.shape}, Goal IKs: {goal_iks.shape}")
        replayer.replay_ik(start_iks, goal_iks, robot_name)

def replay_traj(replayer, plan_dir: str, robot_name: str, scene_name: str,
               object_id: str, grasp_id: int, stage: str, filtered: bool = False):
    """
    Replay trajectories for specified stage.
    
    Args:
        stage: 'reach', 'open', or 'all'
        filtered: Whether to use filtered trajectories
    """
    data_label = "filtered" if filtered else "unfiltered"
    print(f"\n{'='*80}")
    print(f"Replaying {data_label} trajectories for {stage.upper()} stage (Grasp {grasp_id})")
    print(f"{'='*80}\n")
    
    if stage == "all":
        # Replay both stages sequentially
        print("=== REACH STAGE TRAJECTORIES ===")
        traj_data_reach = load_traj_data(plan_dir, robot_name, scene_name,
                                        object_id, grasp_id, "reach", filtered)
        reach_start = traj_data_reach["reach_start_state"]
        reach_goal = traj_data_reach["reach_goal_state"]
        reach_traj = traj_data_reach["reach_traj"]
        reach_success = traj_data_reach["reach_success"]
        print(f"Reach trajectories: {reach_traj.shape}, Success: {reach_success.sum()}/{len(reach_success)}")
        replayer.replay_traj(reach_start, reach_goal, reach_traj, reach_success, robot_name)
        
        print("\n=== OPEN STAGE TRAJECTORIES ===")
        traj_data_open = load_traj_data(plan_dir, robot_name, scene_name,
                                       object_id, grasp_id, "open", filtered)
        open_start = traj_data_open["open_start_state"]
        open_goal = traj_data_open["open_goal_state"]
        open_traj = traj_data_open["open_traj"]
        open_success = traj_data_open["open_success"]
        print(f"Open trajectories: {open_traj.shape}, Success: {open_success.sum()}/{len(open_success)}")
        replayer.replay_traj_akr(open_start, open_goal, open_traj, open_success, robot_name)
    
    elif stage == "reach":
        traj_data = load_traj_data(plan_dir, robot_name, scene_name,
                                   object_id, grasp_id, stage, filtered)
        start_states = traj_data["reach_start_state"]
        goal_states = traj_data["reach_goal_state"]
        trajectories = traj_data["reach_traj"]
        success = traj_data["reach_success"]
        print(f"Trajectories: {trajectories.shape}, Success: {success.sum()}/{len(success)}")
        replayer.replay_traj(start_states, goal_states, trajectories, success, robot_name)
    
    elif stage == "open":
        traj_data = load_traj_data(plan_dir, robot_name, scene_name,
                                   object_id, grasp_id, stage, filtered)
        start_states = traj_data["open_start_state"]
        goal_states = traj_data["open_goal_state"]
        trajectories = traj_data["open_traj"]
        success = traj_data["open_success"]
        print(f"Trajectories: {trajectories.shape}, Success: {success.sum()}/{len(success)}")
        replayer.replay_traj_akr(start_states, goal_states, trajectories, success, robot_name)

def record_traj(replayer, plan_dir: str, robot_name: str, scene_name: str,
               object_id: str, grasp_id: int, stage: str, num_episodes: int = 10):
    """
    Record camera data for trajectories.
    
    Args:
        stage: 'reach', 'open', or 'all'
        num_episodes: Number of episodes to record
    """
    print(f"\n{'='*80}")
    print(f"Recording {num_episodes} episodes for {stage.upper()} stage (Grasp {grasp_id})")
    print(f"{'='*80}\n")
    
    # Output directory for camera data
    output_base = os.path.join(plan_dir, robot_name, scene_name, object_id,
                               f"grasp_{grasp_id:04d}", f"{stage}_stage", "camera_data")
    os.makedirs(output_base, exist_ok=True)
    
    if stage == "all":
        # Record combined trajectory (reach + open in sequence)
        print("=== RECORDING COMBINED (REACH + OPEN) TRAJECTORIES ===")
        
        # Load both stages
        traj_data_reach = load_traj_data(plan_dir, robot_name, scene_name,
                                        object_id, grasp_id, "reach", filtered=True)
        traj_data_open = load_traj_data(plan_dir, robot_name, scene_name,
                                       object_id, grasp_id, "open", filtered=True)
        
        reach_start = traj_data_reach["reach_start_state"]
        reach_goal = traj_data_reach["reach_goal_state"]
        reach_traj = traj_data_reach["reach_traj"]
        reach_success = traj_data_reach["reach_success"]
        
        open_start = traj_data_open["open_start_state"]
        open_goal = traj_data_open["open_goal_state"]
        open_traj = traj_data_open["open_traj"]
        open_success = traj_data_open["open_success"]
        
        # For combined recording, we need to match reach goal with open start
        # This requires careful alignment - for now, record them separately
        # TODO: Implement combined trajectory recording that sequences reach→open
        print("WARNING: Combined trajectory recording not yet implemented.")
        print("Recording reach and open stages separately...")
        
        # Randomly downsample reach stage
        print("\n--- Recording Reach Stage ---")
        reach_num_available = reach_traj.shape[0]
        if reach_num_available > num_episodes:
            # Random downsampling
            indices = torch.randperm(reach_num_available)[:num_episodes]
            reach_start = reach_start[indices]
            reach_goal = reach_goal[indices]
            reach_traj = reach_traj[indices]
            reach_success = reach_success[indices]
            print(f"Randomly downsampled from {reach_num_available} to {num_episodes} trajectories")
        else:
            print(f"Using all {reach_num_available} available trajectories")
        
        replayer.replay_traj_record(
            reach_start, reach_goal, reach_traj, reach_success, robot_name,
            output_dir=os.path.join(output_base, "reach"),
            scene_id=scene_name, object_id=object_id,
            angle_id="0", pose_id=str(grasp_id),
            num_episodes=reach_traj.shape[0]  # Use actual number after downsampling
        )
        
        # Randomly downsample open stage
        print("\n--- Recording Open Stage ---")
        open_num_available = open_traj.shape[0]
        if open_num_available > num_episodes:
            # Random downsampling
            indices = torch.randperm(open_num_available)[:num_episodes]
            open_start = open_start[indices]
            open_goal = open_goal[indices]
            open_traj = open_traj[indices]
            open_success = open_success[indices]
            print(f"Randomly downsampled from {open_num_available} to {num_episodes} trajectories")
        else:
            print(f"Using all {open_num_available} available trajectories")
        
        replayer.replay_traj_record(
            open_start, open_goal, open_traj, open_success, robot_name,
            output_dir=os.path.join(output_base, "open"),
            scene_id=scene_name, object_id=object_id,
            angle_id="0", pose_id=str(grasp_id),
            num_episodes=open_traj.shape[0]  # Use actual number after downsampling
        )
    
    elif stage == "reach":
        traj_data = load_traj_data(plan_dir, robot_name, scene_name,
                                   object_id, grasp_id, stage, filtered=True)
        start_states = traj_data["reach_start_state"]
        goal_states = traj_data["reach_goal_state"]
        trajectories = traj_data["reach_traj"]
        success = traj_data["reach_success"]
        
        num_available = trajectories.shape[0]
        print(f"Available trajectories: {num_available}")
        
        # Randomly downsample if needed
        if num_available > num_episodes:
            indices = torch.randperm(num_available)[:num_episodes]
            start_states = start_states[indices]
            goal_states = goal_states[indices]
            trajectories = trajectories[indices]
            success = success[indices]
            print(f"Randomly downsampled from {num_available} to {num_episodes} trajectories")
        else:
            print(f"Using all {num_available} available trajectories")
        
        replayer.replay_traj_record(
            start_states, goal_states, trajectories, success, robot_name,
            output_dir=output_base, scene_id=scene_name, object_id=object_id,
            angle_id="0", pose_id=str(grasp_id),
            num_episodes=trajectories.shape[0]  # Use actual number after downsampling
        )
    
    elif stage == "open":
        traj_data = load_traj_data(plan_dir, robot_name, scene_name,
                                   object_id, grasp_id, stage, filtered=True)
        start_states = traj_data["open_start_state"]
        goal_states = traj_data["open_goal_state"]
        trajectories = traj_data["open_traj"]
        success = traj_data["open_success"]
        
        num_available = trajectories.shape[0]
        print(f"Available trajectories: {num_available}")
        
        # Randomly downsample if needed
        if num_available > num_episodes:
            indices = torch.randperm(num_available)[:num_episodes]
            start_states = start_states[indices]
            goal_states = goal_states[indices]
            trajectories = trajectories[indices]
            success = success[indices]
            print(f"Randomly downsampled from {num_available} to {num_episodes} trajectories")
        else:
            print(f"Using all {num_available} available trajectories")
        
        replayer.replay_traj_record(
            start_states, goal_states, trajectories, success, robot_name,
            output_dir=output_base, scene_id=scene_name, object_id=object_id,
            angle_id="0", pose_id=str(grasp_id),
            num_episodes=trajectories.shape[0]  # Use actual number after downsampling
        )
    
    print(f"\nCamera data saved to: {output_base}")

# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Replay and record two-stage (reach + open) motion planning results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay reach stage IK
  python examples/example_replay_reach.py --scene-dir assets/scene/infinigen/kitchen_1130/scene_0_seed_0 \\
      --plan-dir output/collect_1211/traj --robot-name summit_franka --object-id 7221 \\
      --grasp-id 0 --mode replay_ik --stage reach

  # Replay open stage filtered trajectories  
  python examples/example_replay_reach.py --scene-dir assets/scene/infinigen/kitchen_1130/scene_0_seed_0 \\
      --plan-dir output/collect_1211/traj --robot-name summit_franka --object-id 7221 \\
      --grasp-id 0 --mode replay_filtered_traj --stage open

  # Record combined (all stages) camera data
  python examples/example_replay_reach.py --scene-dir assets/scene/infinigen/kitchen_1130/scene_0_seed_0 \\
      --plan-dir output/collect_1211/traj --robot-name summit_franka --object-id 7221 \\
      --grasp-id 0 --mode record --stage all --num-episodes 10
        """
    )
    
    # Required arguments
    parser.add_argument("--scene-dir", type=str, required=True,
                       help="Path to scene directory")
    parser.add_argument("--plan-dir", type=str, required=True,
                       help="Base directory containing planning results")
    parser.add_argument("--robot-name", type=str, required=True,
                       choices=list(ROBOT_CONFIG_MAP.keys()),
                       help="Robot name")
    parser.add_argument("--object-id", type=str, required=True,
                       choices=list(OBJECT_CONFIG_MAP.keys()),
                       help="Object ID")
    parser.add_argument("--grasp-id", type=int, required=True,
                       help="Grasp ID to replay")
    
    # Mode and stage
    parser.add_argument("--mode", type=str, required=True,
                       choices=["replay_ik", "replay_traj", "replay_filtered_traj", "record"],
                       help="Operation mode")
    parser.add_argument("--stage", type=str, required=True,
                       choices=["reach", "open", "all"],
                       help="Planning stage to replay/record")
    
    # Optional arguments
    parser.add_argument("--num-episodes", type=int, default=10,
                       help="Number of episodes to record (for record mode)")
    parser.add_argument("--headless", action="store_true",
                       help="Run in headless mode (no GUI)")
    
    args = parser.parse_args()
    
    # Extract scene name from scene directory
    scene_name = Path(args.scene_dir).name
    
    try:
        # Initialize Isaac Sim
        print("Initializing Isaac Sim...")
        global simulation_app
        simulation_app = initialize_isaac_sim(headless=args.headless)
        
        # Import modules
        print("Importing modules...")
        (ObjectDescription, RobotDescription, TaskDescription, TaskType,
         InfinigenScenePipeline, Replayer, single_axis_self_rotation,
         matrix_to_pose, pose_multiply, action_registry_module) = import_modules()
        
        # Create object and load scene
        print(f"Creating object: {args.object_id}")
        obj = create_object(args.object_id, ObjectDescription)
        
        print(f"Loading scene: {args.scene_dir}")
        scene_result = load_scene(args.scene_dir, [obj], InfinigenScenePipeline,
                                 single_axis_self_rotation, matrix_to_pose)
        
        # Create task
        robot_cfg_path = ROBOT_CONFIG_MAP[args.robot_name]
        robot = RobotDescription(args.robot_name, robot_cfg_path)
        task = TaskDescription(
            robot=robot,
            object=obj,
            scene=scene_result.scene,
            task_type=TaskType.ARTICULATE,
        )
        
        # Create replayer
        print("Creating replayer...")
        robot_cfg = robot.robot_cfg  # Use the robot_cfg from RobotDescription
        scene_cfg = {
            "path": scene_result.scene.scene_usd_path,
            "pose": scene_result.scene.pose,
        }
        object_cfg = {
            "path": obj.urdf_path,
            "asset_type": obj.asset_type,
            "asset_id": obj.asset_id,
            "pose": obj.pose,
            "joint_id": 0,
        }
        
        replayer = Replayer(simulation_app, robot_cfg, scene_cfg, object_cfg)
        
        # Setup visualization
        print("Setting up visualization...")
        action_registry = action_registry_module.get_action_registry()
        setup_visualization(action_registry)
        deactivate_scene_prims(replayer, args.object_id)
        
        # Execute requested mode
        print(f"\nExecuting mode: {args.mode} for stage: {args.stage}")
        
        if args.mode == "replay_ik":
            replay_ik(replayer, args.plan_dir, args.robot_name, scene_name,
                     args.object_id, args.grasp_id, args.stage)
        
        elif args.mode == "replay_traj":
            replay_traj(replayer, args.plan_dir, args.robot_name, scene_name,
                       args.object_id, args.grasp_id, args.stage, filtered=False)
        
        elif args.mode == "replay_filtered_traj":
            replay_traj(replayer, args.plan_dir, args.robot_name, scene_name,
                       args.object_id, args.grasp_id, args.stage, filtered=True)
        
        elif args.mode == "record":
            record_traj(replayer, args.plan_dir, args.robot_name, scene_name,
                       args.object_id, args.grasp_id, args.stage, args.num_episodes)
        
        # Keep simulation running
        print("\nReplay complete. Close the window to exit.")
        replayer.isaacsim_step(step=-1, render=True)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if simulation_app:
            simulation_app.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
