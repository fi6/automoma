#!/usr/bin/env python3
"""
Pipeline Collect Script

This script batch-collects camera data by replaying planned trajectories.
It stacks all filtered_traj_data.pt from all grasp folders under a scene+asset,
randomly selects NUM_EPISODES trajectories, replays them, and saves:
1. total_filtered_traj_data.pt (all stacked data)
2. selected_filtered_traj_data.pt (sampled data) 
3. camera_data/ (HDF5 episodes)

Usage:
    # Collect camera data for all scenes in a directory (with specific object)
    python pipeline_collect.py --scene-dir /path/to/scenes --plan-dir output --robot-name summit_franka --object_id 7221

    # Stats only (no collection)
    python pipeline_collect.py --plan-dir output --robot-name summit_franka --stats-only

Notes:
    - Requires that planning outputs already exist under: <plan_dir>/<robot_name>/<scene_name>/<object_id>/grasp_XXXX
    - Stacks all filtered_traj_data.pt files from all grasp folders
    - Saves all outputs under: <plan_dir>/<robot_name>/<scene_name>/<object_id>/
    - Produces stats JSON at: <plan_dir>/<robot_name>/camera_statistics.json
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure only one GPU is used

# Add the src directory to the path so we can import automoma
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Global variable to hold Isaac Sim app instance
simulation_app = None

def initialize_isaac_sim():
    """Initialize Isaac Sim when needed."""
    global simulation_app
    if simulation_app is None:
        # Lazy import and start Isaac Sim headless
        import isaacsim
        from omni.isaac.kit import SimulationApp
        simulation_app = SimulationApp({
            "headless": True,
            "width": 1920,
            "height": 1080,
        })
    return simulation_app

def import_automoma_modules():
    """Import automoma modules when needed."""
    from automoma.models.object import ObjectDescription
    from automoma.models.robot import RobotDescription
    from automoma.models.task import TaskDescription, TaskType
    from automoma.pipeline import InfinigenScenePipeline, ReplayPipeline
    from automoma.utils.transform import single_axis_self_rotation, matrix_to_pose
    from cuakr.utils.math import pose_multiply
    return ObjectDescription, RobotDescription, TaskDescription, TaskType, InfinigenScenePipeline, ReplayPipeline, single_axis_self_rotation, matrix_to_pose, pose_multiply


# =============================
# Hyperparameters (can be edited)
# =============================
NUM_EPISODES = 1000  # Number of episodes to randomly sample and record
RANDOM_SEED = 42  # For reproducible sampling
SELECT_TYPE = "RANDOM"  # "RANDOM" or "LARGEST" - how to select trajectories

# Optional scene cleanup toggles - common elements always deactivated
DEACTIVATE_PRIMS_COMMON = [
    "exterior",
    "ceiling", 
    "Ceiling",
]

def get_deactivate_prims_for_object(object_id: str) -> List[str]:
    """Get list of prims to deactivate for a specific object.
    
    Args:
        object_id: Object ID (e.g., "7221", "11622")
    
    Returns:
        List of prim names to deactivate, including common scene elements and object-specific factory
    """
    deactivate_list = DEACTIVATE_PRIMS_COMMON.copy()
    
    # Add object-specific factory prim to hide the duplicated static object
    if object_id in OBJECT_CONFIG_MAP:
        obj_config = OBJECT_CONFIG_MAP[object_id]
        asset_type = obj_config["asset_type"]
        if asset_type == "StorageFurniture":
            asset_type = "Cabinet"  # Adjust naming inconsistency
        factory_prim = f"StaticCategoryFactory_{asset_type}_{object_id}"
        deactivate_list.append(factory_prim)
    
    return deactivate_list

# Object configuration mapping with asset type, scale, and URDF paths (matching pipeline_plan.py)
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

# Robot configuration mapping - add more robots as needed
ROBOT_CONFIG_MAP = {
    "summit_franka": "assets/robot/summit_franka/summit_franka.yml",
    "summit_franka_fixed_base": "assets/robot/summit_franka/summit_franka_fixed_base.yml",
    # Add other robots here as needed
    # "other_robot": "assets/robot/other_robot/other_robot.yml",
}


def get_robot_config_path(robot_name: str) -> str:
    """Get the configuration path for a robot by name."""
    if robot_name not in ROBOT_CONFIG_MAP:
        raise ValueError(f"Unknown robot: {robot_name}. Available robots: {list(ROBOT_CONFIG_MAP.keys())}")
    return ROBOT_CONFIG_MAP[robot_name]


def get_object_config(object_id: str) -> Dict[str, Any]:
    """Get the configuration for an object by ID."""
    if object_id not in OBJECT_CONFIG_MAP:
        raise ValueError(f"Unknown object ID: {object_id}. Available objects: {list(OBJECT_CONFIG_MAP.keys())}")
    return OBJECT_CONFIG_MAP[object_id]


def create_object(object_id: str):
    """Create an object from the registry.
    
    Args:
        object_id: Object ID (e.g., "7221", "11622", "103634")
    
    Returns:
        ObjectDescription instance
    """
    ObjectDescription, _, _, _, _, _, _, _, _ = import_automoma_modules()
    obj_config = get_object_config(object_id)
    
    obj = ObjectDescription(
        asset_type=obj_config["asset_type"],
        asset_id=obj_config["asset_id"],
        scale=obj_config["scale"],
        urdf_path=obj_config["urdf_path"],
    )
    obj.set_handle_link(obj_config["handle_link"])
    return obj


def load_and_stack_trajectory_data(scene_asset_dir: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Load and stack all filtered_traj_data.pt files from grasp folders.
    
    Args:
        scene_asset_dir: Path to scene+asset directory containing grasp_XXXX folders
        
    Returns:
        Tuple of (stacked_start_states, stacked_goal_states, stacked_trajectories, stacked_success, grasp_sources)
        grasp_sources: list indicating which grasp each trajectory came from
    """
    scene_asset_path = Path(scene_asset_dir)
    
    # Check if total_filtered_traj_data.pt already exists
    total_traj_file = scene_asset_path / "total_filtered_traj_data.pt"
    if total_traj_file.exists():
        print(f"Loading existing total trajectory data from {total_traj_file}")
        try:
            data = torch.load(total_traj_file, weights_only=False)
            return (data["start_state"], data["goal_state"], data["traj"], 
                   data["success"], data["grasp_sources"])
        except Exception as e:
            print(f"Error loading existing file, regenerating: {e}")
    
    # Find all grasp folders
    grasp_folders = [d for d in scene_asset_path.iterdir() if d.is_dir() and d.name.startswith("grasp_")]
    grasp_folders.sort()
    
    if not grasp_folders:
        raise ValueError(f"No grasp folders found in {scene_asset_dir}")
    
    print(f"Found {len(grasp_folders)} grasp folders: {[f.name for f in grasp_folders]}")
    
    # Lists to collect data
    all_start_states = []
    all_goal_states = []
    all_trajectories = []
    all_success = []
    grasp_sources = []
    
    # Load data from each grasp folder
    for grasp_folder in grasp_folders:
        filtered_traj_path = grasp_folder / "filtered_traj_data.pt"
        # filtered_traj_path = grasp_folder / "filtered_traj_data_interpolated.pt"
        
        if not filtered_traj_path.exists():
            print(f"Warning: No filtered_traj_data.pt found in {grasp_folder.name}, skipping")
            continue
            
        try:
            traj_data = torch.load(filtered_traj_path, weights_only=False)
            start_states = traj_data["start_state"]
            goal_states = traj_data["goal_state"]
            trajectories = traj_data["traj"]
            success = traj_data["success"]
            
            # Only keep successful trajectories
            successful_mask = success.bool()
            if successful_mask.sum() == 0:
                print(f"Warning: No successful trajectories in {grasp_folder.name}, skipping")
                # Clean up memory
                del traj_data, start_states, goal_states, trajectories, success, successful_mask
                continue
            
            successful_start = start_states[successful_mask]
            successful_goal = goal_states[successful_mask]
            successful_traj = trajectories[successful_mask]
            successful_success = success[successful_mask]
            
            all_start_states.append(successful_start)
            all_goal_states.append(successful_goal)
            all_trajectories.append(successful_traj)
            all_success.append(successful_success)
            
            # Track which grasp each trajectory came from
            grasp_sources.extend([grasp_folder.name] * len(successful_start))
            
            print(f"Loaded {len(successful_start)} successful trajectories from {grasp_folder.name}")
            
            # Clean up memory for original data
            del traj_data, start_states, goal_states, trajectories, success, successful_mask
            del successful_start, successful_goal, successful_traj, successful_success
            
        except Exception as e:
            print(f"Error loading data from {grasp_folder.name}: {e}")
            continue
    
    if not all_start_states:
        raise ValueError(f"No successful trajectory data found in any grasp folder in {scene_asset_dir}")
    
    # Stack all data
    stacked_start_states = torch.cat(all_start_states, dim=0)
    stacked_goal_states = torch.cat(all_goal_states, dim=0)
    stacked_trajectories = torch.cat(all_trajectories, dim=0)
    stacked_success = torch.cat(all_success, dim=0)
    
    # Clean up intermediate lists to free memory
    del all_start_states, all_goal_states, all_trajectories, all_success
    
    print(f"Total stacked trajectories: {len(stacked_start_states)}")
    print(f"Success rate: {stacked_success.sum().item()}/{len(stacked_success)}")
    
    return stacked_start_states, stacked_goal_states, stacked_trajectories, stacked_success, grasp_sources


def randomly_sample_trajectories(start_states: torch.Tensor, goal_states: torch.Tensor, 
                                trajectories: torch.Tensor, success: torch.Tensor,
                                grasp_sources: List[str], num_episodes: int, 
                                random_seed: int = None, output_dir: str = None,
                                select_type: str = "RANDOM") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[int]]:
    """
    Sample num_episodes trajectories from the stacked data.
    
    Args:
        select_type: "RANDOM" for random sampling, "LARGEST" for largest opening angles
    
    Returns:
        Tuple of (sampled_start_states, sampled_goal_states, sampled_trajectories, sampled_success, sampled_grasp_sources, selected_indices)
    """
    # Check if selected_filtered_traj_data.pt already exists
    if output_dir:
        selected_traj_file = os.path.join(output_dir, "selected_filtered_traj_data.pt")
        if os.path.exists(selected_traj_file):
            print(f"Loading existing selected trajectory data from {selected_traj_file}")
            try:
                data = torch.load(selected_traj_file, weights_only=False)
                return (data["start_state"], data["goal_state"], data["traj"], 
                       data["success"], data["grasp_sources"], data["selected_indices"])
            except Exception as e:
                print(f"Error loading existing selected file, regenerating: {e}")
    
    total_trajectories = len(start_states)
    
    if num_episodes >= total_trajectories:
        print(f"Requested {num_episodes} episodes >= available {total_trajectories}, using all trajectories")
        selected_indices = list(range(total_trajectories))
    else:
        if select_type == "LARGEST":
            # Extract final opening angles from trajectories
            # trajectories shape: (N, Length, Joint)
            # Get the last joint angle from the last timestep
            final_angles = trajectories[:, -1, -1].cpu().numpy()
            
            # Print angle statistics for debug
            print(f"Opening angle statistics:")
            print(f"  Largest angle: {final_angles.max():.4f} rad ({np.degrees(final_angles.max()):.2f} deg)")
            print(f"  Smallest angle: {final_angles.min():.4f} rad ({np.degrees(final_angles.min()):.2f} deg)")
            print(f"  Mean angle: {final_angles.mean():.4f} rad ({np.degrees(final_angles.mean()):.2f} deg)")
            
            # Get indices of largest angles
            sorted_indices = np.argsort(final_angles)[::-1]  # Descending order
            selected_indices = sorted_indices[:num_episodes].tolist()
            selected_indices.sort()  # Sort for easier tracking
            
            print(f"Selected top {len(selected_indices)} trajectories with largest opening angles")
            print(f"  Selected angle range: [{final_angles[sorted_indices[num_episodes-1]]:.4f}, {final_angles[sorted_indices[0]]:.4f}] rad")
        else:  # RANDOM
            # Set random seed for reproducible sampling
            if random_seed is not None:
                random.seed(random_seed)
                torch.manual_seed(random_seed)
            
            selected_indices = random.sample(range(total_trajectories), num_episodes)
            selected_indices.sort()  # Sort for easier tracking
            print(f"Randomly selected {len(selected_indices)} trajectories")
        
    # Sample the data
    sampled_start_states = start_states[selected_indices]
    sampled_goal_states = goal_states[selected_indices]
    sampled_trajectories = trajectories[selected_indices]
    sampled_success = success[selected_indices]
    sampled_grasp_sources = [grasp_sources[i] for i in selected_indices]
    
    return sampled_start_states, sampled_goal_states, sampled_trajectories, sampled_success, sampled_grasp_sources, selected_indices


def save_trajectory_data(output_dir: str, start_states: torch.Tensor, goal_states: torch.Tensor,
                        trajectories: torch.Tensor, success: torch.Tensor, 
                        grasp_sources: List[str], filename: str) -> None:
    """Save trajectory data to file with metadata."""
    os.makedirs(output_dir, exist_ok=True)
    
    traj_data = {
        "start_state": start_states.cpu(),
        "goal_state": goal_states.cpu(),
        "traj": trajectories.cpu(),
        "success": success.cpu(),
        "grasp_sources": grasp_sources,  # Track which grasp each trajectory came from
        "num_trajectories": len(start_states),
    }
    
    output_path = os.path.join(output_dir, filename)
    torch.save(traj_data, output_path)
    print(f"Saved {len(start_states)} trajectories to {output_path}")
    
    # Clean up memory
    del traj_data



def load_scene(scene_path: str, objects: List):
    """Load scene and set object poses to match example/pipeline_plan behavior."""
    _, _, _, _, InfinigenScenePipeline, _, single_axis_self_rotation, matrix_to_pose, _ = import_automoma_modules()
    scene_pipeline = InfinigenScenePipeline()
    scene_result = scene_pipeline.load_scene(scene_path, objects)

    # Set a consistent scene base pose
    scene_pose = [0, 0, -0.13, 1, 0, 0, 0]
    scene_result.scene.set_pose(scene_pose)

    # Set object poses based on original example logic
    for obj in scene_result.valid_objects:
        obj_pose = scene_result.scene.get_object_matrix(obj)
        obj_pose = single_axis_self_rotation(obj_pose, 'z', np.pi)
        obj_pose = matrix_to_pose(obj_pose)
        # Set object pose directly (scene pose is already applied to scene)
        obj.set_pose(obj_pose)
        print(f"Set pose for object {obj.asset_id}: {obj_pose}")

    return scene_result


def collect_for_scene_asset(scene_path: str, scene_name: str, asset_id: str,
                           plan_dir: str, robot_name: str, object_id: str, num_episodes: int) -> Dict[str, Any]:
    """Collect camera data for a single scene+asset combination by stacking all grasp trajectories.
    
    Args:
        scene_path: Path to the scene directory
        scene_name: Name of the scene
        asset_id: Asset/object ID (e.g., "7221", "11622")
        plan_dir: Directory where planning results were saved
        robot_name: Name of the robot to use
        object_id: Object ID (same as asset_id for compatibility)
        num_episodes: Number of episodes to sample and record
    """
    stats = {
        "scene_name": scene_name,
        "asset_id": asset_id,
        "total_trajectories_found": 0,
        "episodes_recorded": 0,
        "output_dir": None,
    }

    try:
        # Initialize Isaac Sim and import modules
        global simulation_app
        simulation_app = initialize_isaac_sim()
        ObjectDescription, RobotDescription, TaskDescription, TaskType, InfinigenScenePipeline, ReplayPipeline, single_axis_self_rotation, matrix_to_pose, pose_multiply = import_automoma_modules()

        # Optional: lighting adjustments similar to example_replay
        import omni.kit.actions.core
        action_registry = omni.kit.actions.core.get_action_registry()
        action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_rig")
        action.execute(lighting_mode=2)

        # Construct scene+asset directory path
        scene_asset_dir = os.path.join(plan_dir, robot_name, scene_name, asset_id)
        if not os.path.exists(scene_asset_dir):
            print(f"Scene+asset directory not found: {scene_asset_dir}")
            return stats

        stats["output_dir"] = scene_asset_dir

        # Step 1: Load and stack all trajectory data from grasp folders
        print(f"Loading and stacking trajectory data from {scene_asset_dir}")
        try:
            stacked_start, stacked_goal, stacked_traj, stacked_success, grasp_sources = load_and_stack_trajectory_data(scene_asset_dir)
            stats["total_trajectories_found"] = len(stacked_start)
        except ValueError as e:
            print(f"Error loading trajectory data: {e}")
            return stats

        # Step 2: Save total stacked data (only if it doesn't exist)
        total_traj_file = os.path.join(scene_asset_dir, "total_filtered_traj_data.pt")
        if not os.path.exists(total_traj_file):
            save_trajectory_data(scene_asset_dir, stacked_start, stacked_goal, stacked_traj, 
                               stacked_success, grasp_sources, "total_filtered_traj_data.pt")

        # Step 3: Sample trajectories (check if already exists)
        print(f"Sampling {num_episodes} trajectories using {SELECT_TYPE} selection")
        sampled_start, sampled_goal, sampled_traj, sampled_success, sampled_sources, selected_indices = randomly_sample_trajectories(
            stacked_start, stacked_goal, stacked_traj, stacked_success, grasp_sources, num_episodes, RANDOM_SEED, scene_asset_dir, SELECT_TYPE)

        # Step 4: Save selected data with additional metadata (only if it doesn't exist)
        selected_output_path = os.path.join(scene_asset_dir, "selected_filtered_traj_data.pt")
        if not os.path.exists(selected_output_path):
            sampled_data_with_indices = {
                "start_state": sampled_start.cpu(),
                "goal_state": sampled_goal.cpu(),
                "traj": sampled_traj.cpu(),
                "success": sampled_success.cpu(),
                "grasp_sources": sampled_sources,
                "selected_indices": selected_indices,  # Track which original trajectories were selected
                "num_trajectories": len(sampled_start),
                "total_available": len(stacked_start),
                "random_seed": RANDOM_SEED,
            }
            torch.save(sampled_data_with_indices, selected_output_path)
            print(f"Saved selected {len(sampled_start)} trajectories to {selected_output_path}")
            del sampled_data_with_indices  # Clean up memory
        else:
            print(f"Selected trajectory data already exists at {selected_output_path}")
        
        # Clean up stacked data to free memory (keep only sampled data)
        del stacked_start, stacked_goal, stacked_traj, stacked_success, grasp_sources

        # Step 5: Set up scene and objects for replay
        obj = create_object(object_id)
        scene_result = load_scene(scene_path, [obj])

        # Create task and replay pipeline
        task = TaskDescription(
            robot=RobotDescription(robot_name, get_robot_config_path(robot_name)),
            object=obj,
            scene=scene_result.scene,
            task_type=TaskType.ARTICULATE,
        )

        replay_pipeline = ReplayPipeline(task, simulation_app=simulation_app, output_base_dir=plan_dir)

        # Optional: deactivate bulky prims to speed up and reduce occlusion
        # Get dynamic list of prims to deactivate based on object_id
        deactivate_prims = get_deactivate_prims_for_object(object_id)
        print(f"Deactivating prims: {deactivate_prims}")
        for prim_name in deactivate_prims:
            try:
                replay_pipeline.replayer.set_deactivate_prims(prim_name)
            except Exception:
                pass

        # Step 6: Record camera data using sampled trajectories
        print(f"Recording camera data for {len(sampled_start)} sampled trajectories")
        
        # Use the replayer's record functionality directly with sampled data
        replay_pipeline.replayer.replay_traj_record(
            start_states=sampled_start,
            goal_states=sampled_goal,
            trajs=sampled_traj,
            successes=sampled_success,
            robot_name=robot_name,
            output_dir=scene_asset_dir,
            scene_id=scene_name,
            object_id=asset_id,
            angle_id="0",
            pose_id="combined",  # Indicate this combines multiple grasps
            num_episodes=None  # Use all sampled trajectories
        )
        
        # Count episodes by checking the camera_data directory
        camera_data_dir = os.path.join(scene_asset_dir, "camera_data")
        if os.path.exists(camera_data_dir):
            episodes_recorded = len([f for f in os.listdir(camera_data_dir) if f.endswith('.hdf5')])
            stats["episodes_recorded"] = episodes_recorded
        else:
            stats["episodes_recorded"] = 0
        
        # Step 7: Save sampling metadata with detailed index tracking
        metadata_path = os.path.join(scene_asset_dir, "collection_metadata.json")
        if not os.path.exists(metadata_path):
            sampling_metadata = {
                "scene_name": scene_name,
                "asset_id": asset_id,
                "total_trajectories_available": stats["total_trajectories_found"],
                "trajectories_sampled": len(sampled_start),
                "episodes_recorded": stats["episodes_recorded"],
                "random_seed": RANDOM_SEED,
                "grasp_source_distribution": {src: sampled_sources.count(src) for src in set(sampled_sources)},
                "selected_indices": selected_indices,
                "selected_indices_count": len(selected_indices),
                "selected_indices_description": "Indices of trajectories selected from the total stacked pool for recording camera data",
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(sampling_metadata, f, indent=2)
            print(f"Saved collection metadata to {metadata_path}")
            
            # Also save selected indices separately for easy reference
            indices_path = os.path.join(scene_asset_dir, "selected_indices.json")
            indices_data = {
                "selected_indices": selected_indices,
                "total_count": len(selected_indices),
                "timestamp": datetime.now().isoformat(),
                "random_seed": RANDOM_SEED,
                "description": "List of indices selected from total stacked trajectories"
            }
            with open(indices_path, 'w') as f:
                json.dump(indices_data, f, indent=2)
            print(f"Saved selected indices to {indices_path}")
            
            del sampling_metadata, indices_data  # Clean up memory
        else:
            print(f"Collection metadata already exists at {metadata_path}")
        
        # Final memory cleanup
        del sampled_start, sampled_goal, sampled_traj, sampled_success, sampled_sources
        del selected_indices
        
        # Force garbage collection and GPU memory cleanup
        import gc
        gc.collect()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during collection for {scene_name}/{asset_id}: {e}")
        import traceback
        traceback.print_exc()

    return stats


def _extract_scene_number(scene_name: str) -> int:
    """Extract numeric part from scene name for proper sorting (e.g., 'scene_12_seed_12' -> 12)."""
    try:
        # Extract the number after 'scene_' and before '_seed'
        parts = scene_name.split('_')
        if len(parts) >= 2 and parts[0] == 'scene':
            return int(parts[1])
        return 0
    except (ValueError, IndexError):
        return 0


class StatisticCollect:
    """Class to manage camera collection statistics generation and updates."""
    
    def __init__(self, plan_dir: str, robot_name: str, object_id: str = None):
        """
        Initialize StatisticCollect.
        
        Args:
            plan_dir: Base plan directory
            robot_name: Robot name
            object_id: Optional object ID filter (if None, analyze all objects)
        """
        self.plan_dir = Path(plan_dir)
        self.robot_name = robot_name
        self.object_id = object_id
        self.base_dir = self.plan_dir / robot_name
        self.stats_file = self.base_dir / "camera_statistics.json"
        
        # Initialize statistics structure
        self.stats = {
            "plan_directory": str(self.base_dir),
            "object_id": object_id,
            "total_scenes": 0,
            "scenes_with_camera_data": 0,
            "total_scene_assets": 0,
            "scene_assets_with_camera_data": 0,
            "total_episodes": 0,
            "statistics_by_scene": {}
        }
    
    def analyze_scene_asset(self, scene_path: Path, asset_path: Path) -> Dict[str, Any]:
        """Analyze camera data for a single scene+asset combination."""
        camera_dir = asset_path / "camera_data"
        result = {
            "asset_id": asset_path.name,
            "has_camera_data": False,
            "episode_count": 0,
            "total_trajectories": 0,
            "selected_trajectories": 0,
        }
        
        # Check for camera_data directory
        if camera_dir.exists() and camera_dir.is_dir():
            h5_files = list(camera_dir.glob("*.hdf5"))
            if h5_files:
                result["has_camera_data"] = True
                result["episode_count"] = len(h5_files)
        
        # Check for trajectory data files
        total_traj_file = asset_path / "total_filtered_traj_data.pt"
        if total_traj_file.exists():
            try:
                data = torch.load(total_traj_file, weights_only=False)
                result["total_trajectories"] = data.get("num_trajectories", len(data.get("traj", [])))
            except Exception as e:
                print(f"Warning: Could not load {total_traj_file}: {e}")
        
        selected_traj_file = asset_path / "selected_filtered_traj_data.pt"
        if selected_traj_file.exists():
            try:
                data = torch.load(selected_traj_file, weights_only=False)
                result["selected_trajectories"] = data.get("num_trajectories", len(data.get("traj", [])))
            except Exception as e:
                print(f"Warning: Could not load {selected_traj_file}: {e}")
        
        return result
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Scan plan directory for camera_data outputs and generate comprehensive statistics."""
        if not self.base_dir.exists():
            print(f"Warning: Base directory {self.base_dir} does not exist")
            return self.stats
        
        # Find all scene directories
        scenes = [p for p in self.base_dir.iterdir() if p.is_dir() and p.name.startswith("scene_")]
        scenes.sort(key=lambda x: _extract_scene_number(x.name))
        self.stats["total_scenes"] = len(scenes)
        
        for scene_path in scenes:
            scene_name = scene_path.name
            scene_stats = {
                "scene_path": str(scene_path),
                "assets": {},
                "scene_total_episodes": 0,
            }
            
            # Find all asset directories within this scene
            if self.object_id:
                # Only analyze specified object_id
                asset_paths = [scene_path / self.object_id] if (scene_path / self.object_id).exists() else []
            else:
                # Analyze all object directories
                asset_paths = [p for p in scene_path.iterdir() if p.is_dir()]
            
            asset_paths.sort()
            self.stats["total_scene_assets"] += len(asset_paths)
            
            for asset_path in asset_paths:
                asset_stats = self.analyze_scene_asset(scene_path, asset_path)
                scene_stats["assets"][asset_path.name] = asset_stats
                
                if asset_stats["has_camera_data"]:
                    self.stats["scene_assets_with_camera_data"] += 1
                    scene_stats["scene_total_episodes"] += asset_stats["episode_count"]
                    self.stats["total_episodes"] += asset_stats["episode_count"]
            
            if scene_stats["scene_total_episodes"] > 0:
                self.stats["scenes_with_camera_data"] += 1
            
            self.stats["statistics_by_scene"][scene_name] = scene_stats
        
        return self.stats
    
    def save(self):
        """Save statistics to JSON file."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.stats_file, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"Camera statistics saved to {self.stats_file}")
    
    def print_summary(self):
        """Print a summary of the statistics."""
        print("\n" + "="*80)
        print("CAMERA COLLECTION STATISTICS SUMMARY")
        print("="*80)
        print(f"Plan Directory: {self.stats['plan_directory']}")
        if self.object_id:
            print(f"Object ID Filter: {self.object_id}")
        print(f"\nScenes:")
        print(f"  Total scenes: {self.stats['total_scenes']}")
        print(f"  Scenes with camera data: {self.stats['scenes_with_camera_data']}")
        print(f"\nScene+Asset Combinations:")
        print(f"  Total: {self.stats['total_scene_assets']}")
        print(f"  With camera data: {self.stats['scene_assets_with_camera_data']}")
        print(f"\nEpisodes:")
        print(f"  Total episodes recorded: {self.stats['total_episodes']}")
        print("="*80 + "\n")


def generate_camera_statistics(plan_dir: str, robot_name: str, object_id: str = None) -> Dict[str, Any]:
    """Scan plan_dir for camera_data outputs and summarize counts per scene/asset.
    
    Args:
        plan_dir: Directory where planning results were saved
        robot_name: Name of the robot
        object_id: If specified, only analyze this object_id; otherwise analyze all
    """
    stats_collector = StatisticCollect(plan_dir, robot_name, object_id)
    stats = stats_collector.generate_statistics()
    stats_collector.print_summary()
    return stats


def _legacy_generate_camera_statistics(plan_dir: str, robot_name: str, object_id: str = None) -> Dict[str, Any]:
    """Legacy implementation - kept for reference."""
    base_dir = os.path.join(plan_dir, robot_name)
    stats: Dict[str, Any] = {
        "plan_directory": base_dir,
        "object_id": object_id,
        "total_scenes": 0,
        "scenes_with_camera_data": 0,
        "total_scene_assets": 0,
        "scene_assets_with_camera_data": 0,
        "total_episodes": 0,
        "statistics_by_scene": {}
    }

    base = Path(base_dir)
    if not base.exists():
        return stats

    # iterate scenes
    scenes = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("scene_")]
    scenes.sort(key=lambda x: _extract_scene_number(x.name))  # Sort numerically by scene number
    stats["total_scenes"] = len(scenes)

    for scene_path in scenes:
        scene_entry = {
            "scene_name": scene_path.name,
            "asset_folders": {},
            "total_episodes": 0,
            "has_camera_data": False,
        }

        # asset id folders under the scene (e.g., 7221)
        if object_id:
            # Only check the specified object_id
            asset_dirs_to_check = [scene_path / object_id] if (scene_path / object_id).exists() else []
        else:
            # Check all asset directories
            asset_dirs_to_check = [d for d in scene_path.iterdir() if d.is_dir()]
        
        for asset_dir in sorted(asset_dirs_to_check):
            asset_entry = {
                "asset_id": asset_dir.name,
                "total_episodes": 0,
                "has_camera_data": False,
                "has_total_traj_data": False,
                "has_selected_traj_data": False,
                "collection_metadata": None,
            }
            
            # Check for new trajectory data files
            total_traj_file = asset_dir / "total_filtered_traj_data.pt"
            selected_traj_file = asset_dir / "selected_filtered_traj_data.pt"
            metadata_file = asset_dir / "collection_metadata.json"
            
            asset_entry["has_total_traj_data"] = total_traj_file.exists()
            asset_entry["has_selected_traj_data"] = selected_traj_file.exists()
            
            # Load metadata if available
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        asset_entry["collection_metadata"] = json.load(f)
                except Exception:
                    pass
            
            # Check camera data directory
            cam_dir = asset_dir / "camera_data"
            if cam_dir.exists():
                # count hdf5 files
                ep_count = len([f for f in cam_dir.iterdir() if f.is_file() and f.suffix == ".hdf5"]) 
                asset_entry["total_episodes"] = ep_count
                if ep_count > 0:
                    asset_entry["has_camera_data"] = True
                    scene_entry["has_camera_data"] = True
                    
            scene_entry["asset_folders"][asset_dir.name] = asset_entry
            scene_entry["total_episodes"] += asset_entry["total_episodes"]
            
            stats["total_scene_assets"] += 1
            if asset_entry["has_camera_data"]:
                stats["scene_assets_with_camera_data"] += 1

        if scene_entry["has_camera_data"]:
            stats["scenes_with_camera_data"] += 1

        stats["statistics_by_scene"][scene_path.name] = scene_entry
        stats["total_episodes"] += scene_entry["total_episodes"]

    return stats


def run_collection_for_directory(scene_dir: str, plan_dir: str, robot_name: str, object_id: str, num_episodes: int) -> None:
    """Run camera data collection for scenes that have been planned.
    
    Args:
        scene_dir: Directory containing scene subdirectories
        plan_dir: Directory where planning results were saved
        robot_name: Name of the robot to use
        object_id: Object ID to collect for (e.g., "7221", "11622")
        num_episodes: Number of episodes to sample and record
    """
    scene_path = Path(scene_dir)
    plan_path = Path(plan_dir) / robot_name

    if not scene_path.exists():
        print(f"###################### Error: Scene directory {scene_dir} does not exist ######################")
        return
    
    if not plan_path.exists():
        print(f"###################### Error: Plan directory {plan_path} does not exist ######################")
        return

    # Find scene+asset combinations that have filtered_traj_data.pt files
    scene_asset_combinations = []
    
    # Get all scene directories and sort them numerically by scene number
    scene_dirs = [d for d in plan_path.iterdir() if d.is_dir() and d.name.startswith('scene_')]
    scene_dirs.sort(key=lambda x: _extract_scene_number(x.name))
    
    for scene_dir_item in scene_dirs:
            # Check the specific object_id directory under this scene
            asset_dir = scene_dir_item / object_id
            if asset_dir.is_dir():
                # Check if this asset has any grasp folders with filtered_traj_data.pt files
                has_traj_data = False
                grasp_count = 0
                for grasp_dir in asset_dir.iterdir():
                    if grasp_dir.is_dir() and grasp_dir.name.startswith('grasp_'):
                        traj_file = grasp_dir / "filtered_traj_data.pt"
                        if traj_file.exists():
                            has_traj_data = True
                            grasp_count += 1
                
                if has_traj_data:
                    # Find corresponding scene in scene_dir
                    scene_full_path = scene_path / scene_dir_item.name
                    if scene_full_path.exists() and scene_full_path.is_dir():
                        scene_asset_combinations.append({
                            'scene_path': scene_full_path,
                            'scene_name': scene_dir_item.name,
                            'asset_id': object_id,
                            'grasp_count': grasp_count
                        })
                    else:
                        print(f"###################### Warning: Planned scene {scene_dir_item.name} not found in {scene_dir} ######################")

    if not scene_asset_combinations:
        print(f"###################### No scene+asset combinations found for object_id {object_id} with filtered_traj_data.pt in {plan_path} ######################")
        return

    print("######################")
    print(f"#### Found {len(scene_asset_combinations)} scene+asset combinations with trajectory data for object_id {object_id} ####")
    for combo in scene_asset_combinations:
        print(f"#### - {combo['scene_name']}/{combo['asset_id']} ({combo['grasp_count']} grasps)")
    print("######################")

    # Process each scene+asset combination
    for combo in scene_asset_combinations:
        scene_name = combo['scene_name']
        asset_id = combo['asset_id']
        print("######################")
        print(f"#### Collecting Camera Data: {scene_name}/{asset_id} ####")
        print("######################")
        
        stat = collect_for_scene_asset(
            scene_path=str(combo['scene_path']),
            scene_name=scene_name,
            asset_id=asset_id,
            plan_dir=plan_dir,
            robot_name=robot_name,
            object_id=object_id,
            num_episodes=num_episodes
        )
        
        # Optionally, write per-scene-asset quick stats
        quick_stats_path = os.path.join(plan_dir, robot_name, scene_name, asset_id, "collection_quick_stats.json")
        try:
            with open(quick_stats_path, 'w') as f:
                json.dump(stat, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed writing quick stats for {scene_name}/{asset_id}: {e}")

    # After all scene+asset combinations, generate global stats
    print("######################")
    print("#### CAMERA COLLECTION COMPLETE ####")
    print("#### Generating camera statistics... ####")
    print("######################")

    stats_collector = StatisticCollect(plan_dir, robot_name, object_id)
    stats_collector.generate_statistics()
    stats_collector.print_summary()
    stats_collector.save()


def main():
    parser = argparse.ArgumentParser(description="Collect camera data by replaying stacked trajectory data from all grasps per scene+asset")
    parser.add_argument("--scene_dir", type=str, default="output/test_collect/infinigen_scene_100", help="Directory containing scene subdirectories (e.g., /path/to/kitchen)")
    parser.add_argument("--plan_dir", type=str, default="output/test_collect/traj", help="Directory where planning results were saved")
    parser.add_argument("--robot_name", type=str, default="summit_franka", help="Robot name used in planning outputs")
    parser.add_argument("--object_id", type=str, default="7221", help=f"Object ID to collect for. Available: {', '.join(OBJECT_CONFIG_MAP.keys())}")
    parser.add_argument("--stats_only", action="store_true", help="Only generate camera statistics from plan-dir")
    parser.add_argument("--num_episodes", type=int, default=None, help="Override number of episodes to record (default: NUM_EPISODES constant)")
    parser.add_argument("--select_type", type=str, default=None, choices=["RANDOM", "LARGEST"], help="Override trajectory selection type (default: SELECT_TYPE constant)")

    args = parser.parse_args()

    # Validate object_id
    if args.object_id not in OBJECT_CONFIG_MAP:
        print(f"Error: Unknown object_id '{args.object_id}'")
        print(f"Available object IDs: {', '.join(OBJECT_CONFIG_MAP.keys())}")
        return

    # Update global SELECT_TYPE if overridden
    global SELECT_TYPE
    if args.select_type is not None:
        SELECT_TYPE = args.select_type
    
    # Resolve effective hyperparameters
    effective_num = args.num_episodes if args.num_episodes is not None else NUM_EPISODES
    effective_select_type = SELECT_TYPE

    if args.stats_only:
        print("###################### Generating camera statistics only ######################")
        print(f"###################### Plan directory: {args.plan_dir} ######################")
        print(f"###################### Robot name: {args.robot_name} ######################")
        print(f"###################### Object ID: {args.object_id} ######################")
        stats = generate_camera_statistics(args.plan_dir, args.robot_name, args.object_id)
        stats_collector = StatisticCollect(args.plan_dir, args.robot_name, args.object_id)
        stats_collector.save()
        return

    if not args.scene_dir:
        print("###################### Error: --scene-dir is required when not using --stats-only ######################")
        return

    print("###################### Starting camera collection with stacked trajectory approach ######################")
    print(f"###################### Scene directory: {args.scene_dir} ######################")
    print(f"###################### Plan directory: {args.plan_dir} ######################")
    print(f"###################### Robot name: {args.robot_name} ######################")
    print(f"###################### Object ID: {args.object_id} ######################")
    print(f"###################### Episodes to sample and record: {effective_num} ######################")
    print(f"###################### Selection type: {effective_select_type} ######################")
    print(f"###################### Random seed: {RANDOM_SEED} ######################")

    run_collection_for_directory(args.scene_dir, args.plan_dir, args.robot_name, args.object_id, effective_num)


if __name__ == "__main__":
    main()
