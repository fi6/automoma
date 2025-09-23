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
    # Collect camera data for all scenes in a directory
    python pipeline_collect.py --scene-dir /path/to/scenes --plan-dir output --robot-name summit_franka

    # Stats only (no collection)
    python pipeline_collect.py --plan-dir output --robot-name summit_franka --stats-only

Notes:
    - Requires that planning outputs already exist under: <plan_dir>/<robot_name>/<scene_name>/<asset_id>/grasp_XXXX
    - Stacks all filtered_traj_data.pt files from all grasp folders
    - Saves all outputs under: <plan_dir>/<robot_name>/<scene_name>/<asset_id>/
    - Produces stats JSON at: <plan_dir>/<robot_name>/camera_statistics.json
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Ensure only one GPU is used

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

# Optional scene cleanup toggles
DEACTIVATE_PRIMS = [
    "exterior",
    "ceiling", 
    "Ceiling",
    # Original object factory prim used in example_replay to hide the duplicated static object
    "StaticCategoryFactory_Microwave_7221",
]


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
    
    print(f"Total stacked trajectories: {len(stacked_start_states)}")
    print(f"Success rate: {stacked_success.sum().item()}/{len(stacked_success)}")
    
    return stacked_start_states, stacked_goal_states, stacked_trajectories, stacked_success, grasp_sources


def randomly_sample_trajectories(start_states: torch.Tensor, goal_states: torch.Tensor, 
                                trajectories: torch.Tensor, success: torch.Tensor,
                                grasp_sources: List[str], num_episodes: int, 
                                random_seed: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[int]]:
    """
    Randomly sample num_episodes trajectories from the stacked data.
    
    Returns:
        Tuple of (sampled_start_states, sampled_goal_states, sampled_trajectories, sampled_success, sampled_grasp_sources, selected_indices)
    """
    total_trajectories = len(start_states)
    
    if num_episodes >= total_trajectories:
        print(f"Requested {num_episodes} episodes >= available {total_trajectories}, using all trajectories")
        selected_indices = list(range(total_trajectories))
    else:
        # Set random seed for reproducible sampling
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        selected_indices = random.sample(range(total_trajectories), num_episodes)
        selected_indices.sort()  # Sort for easier tracking
        
    print(f"Selected {len(selected_indices)} trajectories for recording")
    
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


def create_7221_object():
    """Create a 7221 microwave object (same as in other scripts)."""
    ObjectDescription, _, _, _, _, _, _, _, _ = import_automoma_modules()
    obj = ObjectDescription(
        asset_type="Microwave",
        asset_id="7221",
        scale=0.3562990018302636,
        urdf_path="assets/object/Microwave/7221/7221_0_scaling.urdf",
    )
    obj.set_handle_link("link_0")
    return obj


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

    return scene_result


def collect_for_scene_asset(scene_path: str, scene_name: str, asset_id: str,
                           plan_dir: str, robot_name: str, num_episodes: int) -> Dict[str, Any]:
    """Collect camera data for a single scene+asset combination by stacking all grasp trajectories."""
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

        # Step 2: Save total stacked data
        save_trajectory_data(scene_asset_dir, stacked_start, stacked_goal, stacked_traj, 
                           stacked_success, grasp_sources, "total_filtered_traj_data.pt")

        # Step 3: Randomly sample trajectories
        print(f"Randomly sampling {num_episodes} trajectories")
        sampled_start, sampled_goal, sampled_traj, sampled_success, sampled_sources, selected_indices = randomly_sample_trajectories(
            stacked_start, stacked_goal, stacked_traj, stacked_success, grasp_sources, num_episodes, RANDOM_SEED)

        # Step 4: Save selected data with additional metadata
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
        selected_output_path = os.path.join(scene_asset_dir, "selected_filtered_traj_data.pt")
        torch.save(sampled_data_with_indices, selected_output_path)
        print(f"Saved selected {len(sampled_start)} trajectories to {selected_output_path}")

        # Step 5: Set up scene and objects for replay
        obj = create_7221_object()
        scene_result = load_scene(scene_path, [obj])

        # Create task and replay pipeline
        task = TaskDescription(
            robot=RobotDescription(robot_name, "assets/robot/summit_franka/summit_franka.yml"),
            object=obj,
            scene=scene_result.scene,
            task_type=TaskType.ARTICULATE,
        )

        replay_pipeline = ReplayPipeline(task, simulation_app=simulation_app, output_base_dir=plan_dir)

        # Optional: deactivate bulky prims to speed up and reduce occlusion
        for prim_name in DEACTIVATE_PRIMS:
            try:
                replay_pipeline.replayer.set_deactivate_prims(prim_name)
            except Exception:
                pass

        # Step 6: Record camera data using sampled trajectories
        print(f"Recording camera data for {len(sampled_start)} sampled trajectories")
        
        # Use the replayer's record functionality directly with sampled data
        camera_results = replay_pipeline.replayer.replay_traj_record(
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
        
        stats["episodes_recorded"] = len(camera_results)
        
        # Step 7: Save sampling metadata
        sampling_metadata = {
            "scene_name": scene_name,
            "asset_id": asset_id,
            "total_trajectories_available": stats["total_trajectories_found"],
            "trajectories_sampled": len(sampled_start),
            "episodes_recorded": stats["episodes_recorded"],
            "random_seed": RANDOM_SEED,
            "grasp_source_distribution": {src: sampled_sources.count(src) for src in set(sampled_sources)},
            "selected_indices": selected_indices,
        }
        
        metadata_path = os.path.join(scene_asset_dir, "collection_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(sampling_metadata, f, indent=2)
        print(f"Saved collection metadata to {metadata_path}")

    except Exception as e:
        print(f"Error during collection for {scene_name}/{asset_id}: {e}")
        import traceback
        traceback.print_exc()

    return stats


def generate_camera_statistics(plan_dir: str, robot_name: str) -> Dict[str, Any]:
    """Scan plan_dir for camera_data outputs and summarize counts per scene/asset."""
    base_dir = os.path.join(plan_dir, robot_name)
    stats: Dict[str, Any] = {
        "plan_directory": base_dir,
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
    scenes.sort()
    stats["total_scenes"] = len(scenes)

    for scene_path in scenes:
        scene_entry = {
            "scene_name": scene_path.name,
            "asset_folders": {},
            "total_episodes": 0,
            "has_camera_data": False,
        }

        # asset id folders under the scene (e.g., 7221)
        for asset_dir in sorted([d for d in scene_path.iterdir() if d.is_dir()]):
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


def run_collection_for_directory(scene_dir: str, plan_dir: str, robot_name: str, num_episodes: int) -> None:
    """Run camera data collection for scenes that have been planned."""
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
    for scene_dir_item in plan_path.iterdir():
        if scene_dir_item.is_dir() and scene_dir_item.name.startswith('scene_'):
            # Check asset directories under this scene
            for asset_dir in scene_dir_item.iterdir():
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
                                'asset_id': asset_dir.name,
                                'grasp_count': grasp_count
                            })
                        else:
                            print(f"###################### Warning: Planned scene {scene_dir_item.name} not found in {scene_dir} ######################")

    if not scene_asset_combinations:
        print(f"###################### No scene+asset combinations with filtered_traj_data.pt found in {plan_path} ######################")
        return

    print("######################")
    print(f"#### Found {len(scene_asset_combinations)} scene+asset combinations with trajectory data ####")
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

    stats = generate_camera_statistics(plan_dir, robot_name)
    stats_file = os.path.join(plan_dir, robot_name, "camera_statistics.json")
    try:
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"###################### Camera statistics saved to {stats_file} ######################")
    except Exception as e:
        print(f"Warning: Failed writing camera statistics: {e}")


def main():
    parser = argparse.ArgumentParser(description="Collect camera data by replaying stacked trajectory data from all grasps per scene+asset")
    parser.add_argument("--scene_dir", type=str, default="output/test_collect/infinigen_scene_100", help="Directory containing scene subdirectories (e.g., /path/to/kitchen)")
    parser.add_argument("--plan_dir", type=str, default="output/test_collect/traj", help="Directory where planning results were saved")
    parser.add_argument("--robot_name", type=str, default="summit_franka", help="Robot name used in planning outputs")
    parser.add_argument("--stats_only", action="store_true", help="Only generate camera statistics from plan-dir")
    parser.add_argument("--num_episodes", type=int, default=None, help="Override number of episodes to record (default: NUM_EPISODES constant)")

    args = parser.parse_args()

    # Resolve effective hyperparameters
    effective_num = args.num_episodes if args.num_episodes is not None else NUM_EPISODES

    if args.stats_only:
        print("###################### Generating camera statistics only ######################")
        print(f"###################### Plan directory: {args.plan_dir} ######################")
        stats = generate_camera_statistics(args.plan_dir, args.robot_name)
        stats_file = os.path.join(args.plan_dir, args.robot_name, "camera_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"###################### Camera statistics saved to {stats_file} ######################")
        return

    if not args.scene_dir:
        print("###################### Error: --scene-dir is required when not using --stats-only ######################")
        return

    print("###################### Starting camera collection with stacked trajectory approach ######################")
    print(f"###################### Scene directory: {args.scene_dir} ######################")
    print(f"###################### Plan directory: {args.plan_dir} ######################")
    print(f"###################### Robot name: {args.robot_name} ######################")
    print(f"###################### Episodes to sample and record: {effective_num} ######################")
    print(f"###################### Random seed: {RANDOM_SEED} ######################")

    run_collection_for_directory(args.scene_dir, args.plan_dir, args.robot_name, effective_num)


if __name__ == "__main__":
    main()
