#!/usr/bin/env python3
"""
Pipeline Collect Script

This script batch-collects camera data by replaying planned trajectories.
It mirrors pipeline_plan for iteration over scenes, and example_replay for
Isaac Sim setup and replay flow, then writes per-scene camera stats.

Usage:
    # Collect camera data for all scenes in a directory
    python pipeline_collect.py --scene-dir /path/to/scenes --plan-dir output --robot-name summit_franka

    # Stats only (no collection)
    python pipeline_collect.py --plan-dir output --robot-name summit_franka --stats-only

Notes:
    - Requires that planning outputs already exist under: <plan_dir>/<robot_name>/<scene_name>/<asset_id>/grasp_XXXX
    - Saves HDF5 episodes under: grasp_XXXX/camera_data/episodeXXXXXX.hdf5
    - Produces stats JSON at: <plan_dir>/<robot_name>/camera_statistics.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import torch

# Lazy import and start Isaac Sim headless for this scene
import isaacsim
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({
    "headless": True,
    "width": 1920,
    "height": 1080,
})

# Add the src directory to the path so we can import automoma
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import numpy as np

from automoma.models.object import ObjectDescription
from automoma.models.robot import RobotDescription
from automoma.models.task import TaskDescription, TaskType
from automoma.pipeline import InfinigenScenePipeline, ReplayPipeline
from automoma.utils.transform import single_axis_self_rotation, matrix_to_pose
from cuakr.utils.math import pose_multiply


# =============================
# Hyperparameters (can be edited)
# =============================
NUM_EPISODES = 300  # Alias, do not edit unless needed
# GRASP_IDS = [0, 12, 14]
GRASP_IDS = [0]

# Optional scene cleanup toggles
DEACTIVATE_PRIMS = [
    "exterior",
    "ceiling",
    "Ceiling",
    # Original object factory prim used in example_replay to hide the duplicated static object
    "StaticCategoryFactory_Microwave_7221",
]


def create_7221_object() -> ObjectDescription:
    """Create a 7221 microwave object (same as in other scripts)."""
    obj = ObjectDescription(
        asset_type="Microwave",
        asset_id="7221",
        scale=0.3562990018302636,
        urdf_path="assets/object/Microwave/7221/7221_0_scaling.urdf",
    )
    obj.set_handle_link("link_0")
    return obj


def load_scene(scene_path: str, objects: List[ObjectDescription]):
    """Load scene and set object poses to match example/pipeline_plan behavior."""
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


def collect_for_scene(scene_path: str, scene_name: str, plan_dir: str, robot_name: str,
                      num_episodes: int, grasp_ids: List[int]) -> Dict[str, Any]:
    """Collect camera data for a single scene across provided grasp IDs."""
    stats = {
        "scene_name": scene_name,
        "grasp_count": 0,
        "total_episodes_requested": 0,
        "total_episodes_recorded": 0,
        "grasp_stats": {}
    }

    # Optional: lighting adjustments similar to example_replay
    import omni.kit.actions.core
    action_registry = omni.kit.actions.core.get_action_registry()
    action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_rig")
    action.execute(lighting_mode=2)

    try:
        # Prepare object and scene
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

        # Iterate grasps
        stats["grasp_count"] = len(grasp_ids)
        for gid in grasp_ids:
            per_grasp = {
                "grasp_id": gid,
                "episodes_requested": num_episodes,
                "episodes_recorded": 0,
                "output_dir": None,
            }

            # Run recording
            results = replay_pipeline.replay_traj_record(grasp_id=gid, num_episodes=num_episodes)
            per_grasp["episodes_recorded"] = len(results)

            # Derive output directory path consistent with ReplayPipeline
            scene_parent = Path(scene_path).name  # expects scene_xxx
            per_grasp["output_dir"] = os.path.join(plan_dir, robot_name, scene_parent, obj.asset_id, f"grasp_{gid:04d}")

            stats["total_episodes_requested"] += num_episodes
            stats["total_episodes_recorded"] += per_grasp["episodes_recorded"]
            stats["grasp_stats"][f"grasp_{gid:04d}"] = per_grasp

        # Cleanly close the pipeline
        # try:
        #     replay_pipeline.close()
        # except Exception:
        #     pass

    finally:
        pass
    #     # Ensure Isaac Sim shuts down
    #     try:
    #         simulation_app.close()
    #     except Exception:
    #         pass

    return stats


def generate_camera_statistics(plan_dir: str, robot_name: str) -> Dict[str, Any]:
    """Scan plan_dir for camera_data outputs and summarize counts per scene/grasp."""
    base_dir = os.path.join(plan_dir, robot_name)
    stats: Dict[str, Any] = {
        "plan_directory": base_dir,
        "total_scenes": 0,
        "scenes_with_camera_data": 0,
        "total_grasps": 0,
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
            "total_grasps": 0,
            "total_episodes": 0,
            "has_camera_data": False,
        }

        # asset id folders under the scene (e.g., 7221)
        for asset_dir in sorted([d for d in scene_path.iterdir() if d.is_dir()]):
            asset_entry = {
                "asset_id": asset_dir.name,
                "grasps": {},
                "total_grasps": 0,
                "total_episodes": 0,
            }
            # grasp_XXXX folders
            grasp_dirs = [g for g in asset_dir.iterdir() if g.is_dir() and g.name.startswith("grasp_")]
            grasp_dirs.sort()
            for gdir in grasp_dirs:
                cam_dir = gdir / "camera_data"
                ep_count = 0
                if cam_dir.exists():
                    # count hdf5 files
                    ep_count = len([f for f in cam_dir.iterdir() if f.is_file() and f.suffix == ".hdf5"]) 
                asset_entry["grasps"][gdir.name] = {"episodes": ep_count, "path": str(cam_dir)}
                asset_entry["total_grasps"] += 1
                asset_entry["total_episodes"] += ep_count
            scene_entry["asset_folders"][asset_dir.name] = asset_entry
            scene_entry["total_grasps"] += asset_entry["total_grasps"]
            scene_entry["total_episodes"] += asset_entry["total_episodes"]

        if scene_entry["total_episodes"] > 0:
            scene_entry["has_camera_data"] = True
            stats["scenes_with_camera_data"] += 1

        stats["statistics_by_scene"][scene_path.name] = scene_entry
        stats["total_grasps"] += scene_entry["total_grasps"]
        stats["total_episodes"] += scene_entry["total_episodes"]

    return stats


def run_collection_for_directory(scene_dir: str, plan_dir: str, robot_name: str,
                                 num_episodes: int, grasp_ids: List[int]) -> None:
    """Run camera data collection for scenes that have been planned."""
    scene_path = Path(scene_dir)
    plan_path = Path(plan_dir) / robot_name

    if not scene_path.exists():
        print(f"###################### Error: Scene directory {scene_dir} does not exist ######################")
        return
    
    if not plan_path.exists():
        print(f"###################### Error: Plan directory {plan_path} does not exist ######################")
        return

    # Find scenes that actually have filtered_traj_data.pt files (i.e., actually planned)
    planned_scenes = []
    for scene_dir_item in plan_path.iterdir():
        if scene_dir_item.is_dir() and scene_dir_item.name.startswith('scene_'):
            # Check if this scene has any filtered_traj_data.pt files
            has_traj_data = False
            for asset_dir in scene_dir_item.iterdir():
                if asset_dir.is_dir():
                    for grasp_dir in asset_dir.iterdir():
                        if grasp_dir.is_dir() and grasp_dir.name.startswith('grasp_'):
                            traj_file = grasp_dir / "filtered_traj_data.pt"
                            if traj_file.exists():
                                has_traj_data = True
                                break
                    if has_traj_data:
                        break
            
            if has_traj_data:
                # Find corresponding scene in scene_dir
                scene_full_path = scene_path / scene_dir_item.name
                if scene_full_path.exists() and scene_full_path.is_dir():
                    planned_scenes.append(scene_full_path)
                else:
                    print(f"###################### Warning: Planned scene {scene_dir_item.name} not found in {scene_dir} ######################")

    if not planned_scenes:
        print(f"###################### No scenes with filtered_traj_data.pt found in {plan_path} ######################")
        return

    print("######################")
    print(f"#### Found {len(planned_scenes)} scenes with trajectory data to collect ####")
    for scene in planned_scenes:
        print(f"#### - {scene.name}")
    print("######################")

    # Process scenes
    for scene_subdir in planned_scenes:
        scene_name = scene_subdir.name
        print("######################")
        print(f"#### Collecting Camera Data: {scene_name} ####")
        print("######################")
        stat = collect_for_scene(str(scene_subdir), scene_name, plan_dir, robot_name, num_episodes, grasp_ids)
        # Optionally, write per-scene quick stats
        quick_stats_path = os.path.join(plan_dir, robot_name, scene_name, "camera_quick_stats.json")
        try:
            with open(quick_stats_path, 'w') as f:
                json.dump(stat, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed writing quick stats for {scene_name}: {e}")

    # After all scenes, generate global stats
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
    parser = argparse.ArgumentParser(description="Collect camera data by replaying planned trajectories across scenes")
    parser.add_argument("--scene_dir", type=str, default="output/infinigen_scene_10", help="Directory containing scene subdirectories (e.g., /path/to/kitchen)")
    parser.add_argument("--plan_dir", type=str, default="output", help="Directory where planning results were saved")
    parser.add_argument("--robot_name", type=str, default="summit_franka", help="Robot name used in planning outputs")
    parser.add_argument("--stats_only", action="store_true", help="Only generate camera statistics from plan-dir")
    parser.add_argument("--num_episodes", type=int, default=None, help="Override number of episodes to record per grasp")
    parser.add_argument("--grasp_ids", type=int, nargs='*', default=None, help="Override list of grasp IDs to record")

    args = parser.parse_args()

    # Resolve effective hyperparameters
    effective_num = args.num_episodes if args.num_episodes is not None else NUM_EPISODES
    effective_grasps = args.grasp_ids if args.grasp_ids is not None else GRASP_IDS

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

    print("###################### Starting camera collection ######################")
    print(f"###################### Scene directory: {args.scene_dir} ######################")
    print(f"###################### Plan directory: {args.plan_dir} ######################")
    print(f"###################### Robot name: {args.robot_name} ######################")
    print(f"###################### Episodes per grasp: {effective_num} ######################")
    print(f"###################### Grasp IDs: {effective_grasps} ######################")

    run_collection_for_directory(args.scene_dir, args.plan_dir, args.robot_name, effective_num, effective_grasps)


if __name__ == "__main__":
    main()
