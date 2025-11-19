#!/usr/bin/env python3
"""
Pipeline Plan Script

This script provides two main functionalities:
1. Run planning pipeline: Given a scene directory with scene subdirectories, run the planning
   pipeline on each scene and save results to a plan directory
2. Generate statistics: Analyze results from a plan directory and generate statistics

Usage:
    # Run pipeline with separate scene and plan directories
    python pipeline_plan.py --scene-dir /path/to/scenes --plan-dir /path/to/output

    # Generate statistics only from plan directory
    python pipeline_plan.py --plan-dir /path/to/output --stats-only
    
    # Run with clustering statistics recording (for ablation study)
    python pipeline_plan.py --scene-dir /path/to/scenes --plan-dir /path/to/output --record-clustering-stats
    
    # Run IK-only mode (skip trajectory planning, useful for IK statistics collection)
    python pipeline_plan.py --scene-dir /path/to/scenes --plan-dir /path/to/output --ik-only --record-clustering-stats

Examples:
    python scripts/pipeline_plan.py --scene_dir /home/xinhai/automoma/output/infinigen_scene_100 --plan_dir output/summit_franka
    python pipeline_plan.py --plan-dir output/summit_franka --stats-only
    python pipeline_plan.py --scene_dir output/infinigen_scene_100 --plan_dir output/traj --ik-only --record-clustering-stats
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Any
import pandas as pd

# Add the src directory to the path so we can import automoma
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from automoma.models.object import ObjectDescription
from automoma.models.robot import RobotDescription
from automoma.models.task import TaskDescription, TaskType
from automoma.pipeline import ScenePipeline, TrajectoryPipeline, InfinigenScenePipeline, AOGraspPipeline
from automoma.utils.transform import single_axis_self_rotation, matrix_to_pose
from cuakr.utils.math import pose_multiply

OBJECT_ID = "7221"

# GRASP_IDS = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 18]
# GRASP_IDS = [4, 5, 6, 9, 12, 13, 18]

# GRASP_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
GRASP_IDS = [0, 1, 2, 4, 5, 9, 11, 12, 13]
SCENE_IDS = [f"scene_{i}_seed_{i}" for i in range(0, 1)]
# SCENE_IDS = [f"scene_{i}_seed_{i}" for i in range(0, 33) if i not in [5, 27]]  # scenes 0 to 31
# SCENE_IDS = [f"scene_{i}_seed_{i+101}" for i in range(0, 10)]  # scenes 0 to 31



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


def create_7221_object():
    """Create a 7221 microwave object (same as in example)."""
    object = ObjectDescription(
        asset_type="Microwave",
        asset_id="7221",
        scale=0.3562990018302636,
        urdf_path="assets/object/Microwave/7221/7221_0_scaling.urdf",
    )
    object.set_handle_link("link_0")
    return object

def create_11622_object():
    """Create a 11622 dishwasher object."""
    object = ObjectDescription(
        asset_type="Dishwasher",
        asset_id="11622",
        scale=0.6446,
        urdf_path="assets/object/Dishwasher/11622/11622_0_scaling.urdf",
    )
    object.set_handle_link("link_0")
    return object

def load_scene(scene_path: str, objects: list):
    """Load scene from path (same as in example)."""
    scene_pipeline = InfinigenScenePipeline()
    scene_result = scene_pipeline.load_scene(scene_path, objects)
    scene_pose = [0, 0, -0.13, 1, 0, 0, 0]
    scene_result.scene.set_pose(scene_pose)

    # set object poses
    for obj in scene_result.valid_objects:
        obj_pose = scene_result.scene.get_object_matrix(obj)
        obj_pose = single_axis_self_rotation(obj_pose, "z", np.pi)
        obj_pose = matrix_to_pose(obj_pose)
        obj.set_pose(pose_multiply(scene_pose, obj_pose))

    return scene_result


def run_pipeline_for_scene(scene_path: str, scene_name: str, plan_dir: str, robot_name: str, record_clustering_stats: bool = False, ik_only: bool = False):
    """Run the complete pipeline for a single scene.
    
    Args:
        scene_path: Path to the scene directory
        scene_name: Name of the scene
        plan_dir: Directory to save planning results
        robot_name: Name of the robot to use
        record_clustering_stats: Whether to record clustering statistics
        ik_only: If True, only run IK planning without trajectory generation
    """
    print("######################")
    print(f"#### Processing Scene: {scene_name} ####")
    if ik_only:
        print("#### Mode: IK-only (no trajectory planning) ####")
    print("######################")

    try:
        # Create object
        print(f"###################### Creating {OBJECT_ID} object ######################")
        if OBJECT_ID == "7221":
            object = create_7221_object()
        elif OBJECT_ID == "11622":
            object = create_11622_object()
        else:
            raise ValueError(f"Unknown object ID: {OBJECT_ID}")

        # Load scene
        print(f"###################### Loading scene from {scene_path} ######################")
        scene_result = load_scene(scene_path, [object])
        print("###################### Scene loaded successfully ######################")

        # Generate grasps
        print("###################### Generating grasps ######################")
        pipeline = AOGraspPipeline()
        grasps = pipeline.generate_grasps(object, 20)
        print(f"###################### Generated {len(grasps)} grasps ######################")

        # Create task with custom output directory
        print("###################### Creating task description ######################")
        task = TaskDescription(
            robot=RobotDescription(robot_name, get_robot_config_path(robot_name)),
            object=object,
            scene=scene_result.scene,
            task_type=TaskType.ARTICULATE,
        )
        
        # TODO: for summit_franka_fixed_base, the angle needs to be multiplied by 4
        if robot_name == "summit_franka_fixed_base":
            task.goal['angle'] *= 5
        print("###################### Task created successfully ######################")

        # Create trajectory pipeline with custom output directory
        trajectory_pipeline = TrajectoryPipeline(task, output_base_dir=plan_dir, record_clustering_stats=record_clustering_stats)
        print(f"###################### Trajectory pipeline created with output dir: {plan_dir} ######################")

        # Process each grasp
        total_grasps = len(grasps)
        for grasp_id, grasp in grasps.items():
            print(f"######################")
            print(f"#### Processing Grasp {grasp_id}/{total_grasps} ####")
            print(f"######################")
            if grasp_id not in GRASP_IDS:
                print(f"Skipping grasp {grasp_id} as it's not in the specified GRASP_IDS")
                continue

            # Update task with current grasp
            task.update_grasp(grasp)
            print("###################### Task updated with new grasp ######################")

            if trajectory_pipeline.check_results_exist(grasp_id):
                print(
                    f"###################### Results already exist for grasp {grasp_id}, skipping... ######################"
                )
                continue

            try:
                # Run pipeline steps
                if OBJECT_ID == "7221":
                    trajectory_pipeline.load_akr_robot(
                        f"assets/object/Microwave/{OBJECT_ID}/{robot_name}_{OBJECT_ID}_0_grasp_{grasp_id:04d}.yml"
                    )
                elif OBJECT_ID == "11622":
                    trajectory_pipeline.load_akr_robot(
                        f"assets/object/Dishwasher/{OBJECT_ID}/{robot_name}_{OBJECT_ID}_0_grasp_{grasp_id:04d}.yml"
                    )
                print("###################### AKR robot loaded ######################")

                trajectory_pipeline.plan_ik()
                print("###################### IK planning completed ######################")

                # Skip trajectory planning if ik_only mode
                if ik_only:
                    trajectory_pipeline.save_results(grasp_id=grasp_id)
                    print("###################### IK results saved (ik-only mode) ######################")
                    print(
                        f"###################### Completed grasp {grasp_id}/{total_grasps} successfully (IK-only) ######################"
                    )
                    continue

                trajectory_pipeline.plan_traj(batch_size=20)
                print("###################### Trajectory planning completed ######################")

                trajectory_pipeline.filter_traj()
                print("###################### Trajectory filtering completed ######################")

                trajectory_pipeline.save_results(grasp_id=grasp_id)
                print("###################### Results saved ######################")

                print(
                    f"###################### Completed grasp {grasp_id}/{total_grasps} successfully ######################"
                )

            except Exception as e:
                print(f"###################### Error processing grasp {grasp_id}: {e} ######################")
                continue

        print("######################")
        print(f"#### Completed Scene: {scene_name} ####")
        print("######################")

        return True

    except Exception as e:
        print(f"###################### Error processing scene {scene_name}: {e} ######################")
        return False


import re
from pathlib import Path


def run_pipelines_for_directory(scene_dir: str, plan_dir: str, robot_name: str, record_clustering_stats: bool = False, ik_only: bool = False):
    """Run pipelines for all scene subdirectories in a directory.
    
    Args:
        scene_dir: Directory containing scene subdirectories
        plan_dir: Directory to save planning results
        robot_name: Name of the robot to use
        record_clustering_stats: Whether to record clustering statistics
        ik_only: If True, only run IK planning without trajectory generation
    """
    scene_path = Path(scene_dir)

    if not scene_path.exists():
        print(f"###################### Error: Scene directory {scene_dir} does not exist ######################")
        return

    # Create plan directory if it doesn't exist
    plan_path = Path(plan_dir)
    plan_path.mkdir(parents=True, exist_ok=True)
    print(f"###################### Plan directory: {plan_dir} ######################")

    scene_dirs = []
    if scene_path.name.startswith("scene_"):
        scene_dirs.append(scene_path)
    else:
        # Find all subdirectories that look like scenes
        for item in scene_path.iterdir():
            if item.is_dir() and item.name.startswith("scene_"):
                scene_dirs.append(item)

    # Sort numerically by scene number
    def extract_scene_number(path):
        match = re.search(r"scene_(\d+)", path.name)
        return int(match.group(1)) if match else -1

    scene_dirs.sort(key=extract_scene_number)

    if not scene_dirs:
        print(f"###################### No scene directories found in {scene_dir} ######################")
        return

    print("######################")
    print(f"#### Found {len(scene_dirs)} scenes to process ####")
    print("######################")
    successful_scenes = 0
    failed_scenes = 0

    for scene_subdir in scene_dirs:
        scene_name = scene_subdir.name
        if SCENE_IDS != None and scene_name not in SCENE_IDS:
            print(f"Skipping scene {scene_name} as it's not in the specified SCENE_IDS")
            continue
        scene_full_path = str(scene_subdir)

        success = run_pipeline_for_scene(scene_full_path, scene_name, plan_dir, robot_name, record_clustering_stats, ik_only)
        if success:
            successful_scenes += 1
        else:
            failed_scenes += 1

    print("######################")
    print("#### PIPELINE COMPLETE ####")
    print(f"#### Successful scenes: {successful_scenes} ####")
    print(f"#### Failed scenes: {failed_scenes} ####")
    print("#### Generating statistics... ####")
    print("######################")

    # Generate statistics
    stats = generate_statistics(plan_dir, robot_name)

    # Save statistics in plan directory
    stats_file = os.path.join(plan_dir, "pipeline_statistics.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"###################### Statistics saved to {stats_file} ######################")
    
    # Generate clustering statistics if recording
    if record_clustering_stats:
        print("#### Generating clustering statistics... ####")
        clustering_stats_df = collect_clustering_statistics(plan_dir, robot_name)
        if clustering_stats_df is not None:
            stats_output_dir = Path("output/statistics")
            stats_output_dir.mkdir(parents=True, exist_ok=True)
            csv_file = stats_output_dir / f"clustering_stats_{robot_name}.csv"
            clustering_stats_df.to_csv(csv_file, index=False)
            print(f"###################### Clustering statistics saved to {csv_file} ######################")
        else:
            print("###################### No clustering statistics found ######################")


def generate_statistics(plan_dir: str, robot_name: str) -> Dict[str, Any]:
    """
    Generate statistics by reading result files from the plan directory.
    This is a standalone function that analyzes existing results.
    """
    print("###################### Analyzing results for statistics ######################")
    plan_dir = os.path.join(plan_dir, robot_name)

    plan_path = Path(plan_dir)
    stats = {
        "plan_directory": str(plan_path),
        "total_scenes": 0,
        "scenes_with_results": 0,
        "total_grasps": 0,
        "statistics_by_scene": {},
        "overall_statistics": {
            "total_ik_results": 0,
            "total_start_iks": 0,
            "total_goal_iks": 0,
            "total_traj_results": 0,
            "total_traj_samples": 0,
            "traj_success_count": 0,
            "total_filtered_results": 0,
            "total_filtered_samples": 0,
            "filtered_success_count": 0,
            "average_traj_success_rate": 0.0,
            "average_filtered_success_rate": 0.0,
        },
    }

    if not plan_path.exists():
        print(f"###################### Error: Plan directory {plan_dir} does not exist ######################")
        return stats

    # Find scene directories in plan directory
    scene_dirs = []
    for item in plan_path.iterdir():
        if item.is_dir() and item.name.startswith("scene_"):
            scene_dirs.append(item.name)

    scene_dirs.sort()
    stats["total_scenes"] = len(scene_dirs)

    for scene_name in scene_dirs:
        scene_output_dir = os.path.join(plan_dir, scene_name, OBJECT_ID)

        scene_stats = {
            "scene_name": scene_name,
            "has_results": False,
            "total_grasps": 0,
            "successful_grasps": 0,
            "grasp_results": {},
        }

        if os.path.exists(scene_output_dir):
            scene_stats["has_results"] = True
            stats["scenes_with_results"] += 1

            # Find all grasp directories
            grasp_dirs = []
            for item in os.listdir(scene_output_dir):
                if item.startswith("grasp_") and os.path.isdir(os.path.join(scene_output_dir, item)):
                    grasp_dirs.append(item)

            grasp_dirs.sort()
            scene_stats["total_grasps"] = len(grasp_dirs)
            stats["total_grasps"] += len(grasp_dirs)

            # Analyze each grasp
            for grasp_dir in grasp_dirs:
                grasp_path = os.path.join(scene_output_dir, grasp_dir)
                grasp_stats = analyze_grasp_results(grasp_path)
                scene_stats["grasp_results"][grasp_dir] = grasp_stats

                # Update overall statistics for IK
                if grasp_stats["ik_exists"]:
                    stats["overall_statistics"]["total_ik_results"] += 1
                    stats["overall_statistics"]["total_start_iks"] += grasp_stats["start_iks_count"]
                    stats["overall_statistics"]["total_goal_iks"] += grasp_stats["goal_iks_count"]

                # Update overall statistics for trajectories
                if grasp_stats["traj_exists"]:
                    stats["overall_statistics"]["total_traj_results"] += 1
                    stats["overall_statistics"]["total_traj_samples"] += grasp_stats["traj_total_count"]
                    stats["overall_statistics"]["traj_success_count"] += grasp_stats["traj_success_count"]

                # Update overall statistics for filtered
                if grasp_stats["filtered_exists"]:
                    stats["overall_statistics"]["total_filtered_results"] += 1
                    stats["overall_statistics"]["total_filtered_samples"] += grasp_stats["filtered_total_count"]
                    stats["overall_statistics"]["filtered_success_count"] += grasp_stats["filtered_success_count"]
                    if grasp_stats["filtered_success_count"] > 0:
                        scene_stats["successful_grasps"] += 1

        stats["statistics_by_scene"][scene_name] = scene_stats

    # Calculate averages
    if stats["overall_statistics"]["total_traj_samples"] > 0:
        stats["overall_statistics"]["average_traj_success_rate"] = (
            stats["overall_statistics"]["traj_success_count"] / stats["overall_statistics"]["total_traj_samples"]
        )

    if stats["overall_statistics"]["total_filtered_samples"] > 0:
        stats["overall_statistics"]["average_filtered_success_rate"] = (
            stats["overall_statistics"]["filtered_success_count"]
            / stats["overall_statistics"]["total_filtered_samples"]
        )

    return stats


def analyze_grasp_results(grasp_path: str) -> Dict[str, Any]:
    """Analyze result files for a single grasp."""
    result = {
        "grasp_path": grasp_path,
        "ik_exists": False,
        "start_iks_count": 0,
        "goal_iks_count": 0,
        "traj_exists": False,
        "traj_total_count": 0,
        "traj_success_count": 0,
        "filtered_exists": False,
        "filtered_total_count": 0,
        "filtered_success_count": 0,
    }

    # Check IK results
    ik_file = os.path.join(grasp_path, "ik_data.pt")
    if os.path.exists(ik_file):
        result["ik_exists"] = True
        try:
            ik_data = torch.load(ik_file, weights_only=False)
            if "start_iks" in ik_data:
                result["start_iks_count"] = ik_data["start_iks"].shape[0]
            if "goal_iks" in ik_data:
                result["goal_iks_count"] = ik_data["goal_iks"].shape[0]
        except Exception as e:
            print(f"Warning: Could not load IK data from {ik_file}: {e}")

    # Check trajectory results
    traj_file = os.path.join(grasp_path, "traj_data.pt")
    if os.path.exists(traj_file):
        result["traj_exists"] = True
        try:
            traj_data = torch.load(traj_file, weights_only=False)
            if "success" in traj_data:
                result["traj_total_count"] = traj_data["success"].shape[0]
                result["traj_success_count"] = traj_data["success"].count_nonzero().item()
        except Exception as e:
            print(f"Warning: Could not load trajectory data from {traj_file}: {e}")

    # Check filtered results
    filtered_file = os.path.join(grasp_path, "filtered_traj_data.pt")
    if os.path.exists(filtered_file):
        result["filtered_exists"] = True
        try:
            filtered_data = torch.load(filtered_file, weights_only=False)
            if "success" in filtered_data:
                result["filtered_total_count"] = filtered_data["success"].shape[0]
                result["filtered_success_count"] = filtered_data["success"].count_nonzero().item()
        except Exception as e:
            print(f"Warning: Could not load filtered data from {filtered_file}: {e}")

    return result


def collect_clustering_statistics(plan_dir: str, robot_name: str) -> pd.DataFrame:
    """
    Collect clustering statistics from all IK data files.
    
    Returns a DataFrame with columns:
    - scene_name
    - object_id
    - grasp_id
    - ik_type (start/goal)
    - before_clustering
    - initial_ik_count (after kmeans preprocessing)
    - ap_input_count
    - ap_unique_labels
    - clustering_method (affinity_propagation/kmeans_fallback/none)
    - final_ik_count
    - clustering_params (as JSON string)
    """
    plan_dir = os.path.join(plan_dir, robot_name)
    plan_path = Path(plan_dir)
    
    if not plan_path.exists():
        print(f"Plan directory {plan_dir} does not exist")
        return None
    
    records = []
    
    # Find all scene directories
    for scene_dir in plan_path.iterdir():
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene_"):
            continue
            
        scene_name = scene_dir.name
        object_dir = scene_dir / OBJECT_ID
        
        if not object_dir.exists():
            continue
        
        # Find all grasp directories
        for grasp_dir in object_dir.iterdir():
            if not grasp_dir.is_dir() or not grasp_dir.name.startswith("grasp_"):
                continue
            
            grasp_id = grasp_dir.name
            ik_file = grasp_dir / "ik_data.pt"
            
            if not ik_file.exists():
                continue
            
            try:
                ik_data = torch.load(str(ik_file), weights_only=False)
                
                # Check if clustering stats exist
                if "clustering_stats" not in ik_data:
                    continue
                
                clustering_stats = ik_data["clustering_stats"]
                clustering_params = clustering_stats.get("clustering_params", {})
                
                # Extract start IK stats
                if "start_ik" in clustering_stats:
                    start_stats = clustering_stats["start_ik"]
                    record = {
                        "scene_name": scene_name,
                        "object_id": OBJECT_ID,
                        "grasp_id": grasp_id,
                        "ik_type": "start",
                        "before_clustering": start_stats.get("before_clustering", 0),
                        "initial_ik_count": start_stats.get("initial_ik_count", 0),
                        "kmeans_applied": start_stats.get("kmeans_applied", False),
                        "kmeans_clusters": start_stats.get("kmeans_clusters", 0),
                        "ap_input_count": start_stats.get("ap_input_count", 0),
                        "ap_unique_labels": start_stats.get("ap_unique_labels", 0),
                        "clustering_method": start_stats.get("clustering_method", "unknown"),
                        "kmeans_fallback_clusters": start_stats.get("kmeans_fallback_clusters", 0),
                        "final_ik_count": start_stats.get("final_ik_count", 0),
                        "ap_fallback_clusters": clustering_params.get("ap_fallback_clusters", 0),
                        "ap_clusters_upperbound": clustering_params.get("ap_clusters_upperbound", 0),
                        "ap_clusters_lowerbound": clustering_params.get("ap_clusters_lowerbound", 0),
                    }
                    records.append(record)
                
                # Extract goal IK stats
                if "goal_ik" in clustering_stats:
                    goal_stats = clustering_stats["goal_ik"]
                    record = {
                        "scene_name": scene_name,
                        "object_id": OBJECT_ID,
                        "grasp_id": grasp_id,
                        "ik_type": "goal",
                        "before_clustering": goal_stats.get("before_clustering", 0),
                        "initial_ik_count": goal_stats.get("initial_ik_count", 0),
                        "kmeans_applied": goal_stats.get("kmeans_applied", False),
                        "kmeans_clusters": goal_stats.get("kmeans_clusters", 0),
                        "ap_input_count": goal_stats.get("ap_input_count", 0),
                        "ap_unique_labels": goal_stats.get("ap_unique_labels", 0),
                        "clustering_method": goal_stats.get("clustering_method", "unknown"),
                        "kmeans_fallback_clusters": goal_stats.get("kmeans_fallback_clusters", 0),
                        "final_ik_count": goal_stats.get("final_ik_count", 0),
                        "ap_fallback_clusters": clustering_params.get("ap_fallback_clusters", 0),
                        "ap_clusters_upperbound": clustering_params.get("ap_clusters_upperbound", 0),
                        "ap_clusters_lowerbound": clustering_params.get("ap_clusters_lowerbound", 0),
                    }
                    records.append(record)
                    
            except Exception as e:
                print(f"Warning: Could not load clustering stats from {ik_file}: {e}")
                continue
    
    if not records:
        return None
    
    df = pd.DataFrame(records)
    return df


def analyze_grasp_results(grasp_path: str) -> Dict[str, Any]:
    """Analyze result files for a single grasp."""
    result = {
        "grasp_path": grasp_path,
        "ik_exists": False,
        "start_iks_count": 0,
        "goal_iks_count": 0,
        "traj_exists": False,
        "traj_total_count": 0,
        "traj_success_count": 0,
        "filtered_exists": False,
        "filtered_total_count": 0,
        "filtered_success_count": 0,
    }

    # Check IK results
    ik_file = os.path.join(grasp_path, "ik_data.pt")
    if os.path.exists(ik_file):
        result["ik_exists"] = True
        try:
            ik_data = torch.load(ik_file, weights_only=False)
            # Count start and goal IKs
            if "start_iks" in ik_data:
                result["start_iks_count"] = ik_data["start_iks"].shape[0]
            if "goal_iks" in ik_data:
                result["goal_iks_count"] = ik_data["goal_iks"].shape[0]
        except Exception as e:
            print(f"Warning: Could not load IK data from {ik_file}: {e}")

    # Check trajectory results
    traj_file = os.path.join(grasp_path, "traj_data.pt")
    if os.path.exists(traj_file):
        result["traj_exists"] = True
        try:
            traj_data = torch.load(traj_file, weights_only=False)
            if "success" in traj_data:
                result["traj_total_count"] = traj_data["success"].shape[0]
                result["traj_success_count"] = traj_data["success"].count_nonzero().item()
        except Exception as e:
            print(f"Warning: Could not load trajectory data from {traj_file}: {e}")

    # Check filtered results
    filtered_file = os.path.join(grasp_path, "filtered_traj_data.pt")
    if os.path.exists(filtered_file):
        result["filtered_exists"] = True
        try:
            filtered_data = torch.load(filtered_file, weights_only=False)
            if "success" in filtered_data:
                result["filtered_total_count"] = filtered_data["success"].shape[0]
                result["filtered_success_count"] = filtered_data["success"].count_nonzero().item()
        except Exception as e:
            print(f"Warning: Could not load filtered data from {filtered_file}: {e}")

    return result


def main():
    """Main function to run the pipeline planner."""
    parser = argparse.ArgumentParser(
        description="Run pipeline planning for all scenes in a directory and generate statistics"
    )
    parser.add_argument(
        "--scene_dir", type=str, help="Directory containing scene subdirectories (e.g., /path/to/kitchen_0919)"
    )
    parser.add_argument(
        "--plan_dir", type=str, default="output", help="Directory to save planning results (e.g., output/summit_franka)"
    )
    parser.add_argument(
        "--stats_only", action="store_true", help="Only generate statistics from existing results in plan-dir"
    )
    parser.add_argument("--robot_name", type=str, default="summit_franka", help="Name of the robot to use for planning")
    parser.add_argument(
        "--record-clustering-stats", 
        action="store_true", 
        help="Record detailed clustering statistics for ablation study (creates CSV in output/statistics/)"
    )
    parser.add_argument(
        "--ik-only",
        action="store_true",
        help="Only run IK planning without trajectory generation (useful for collecting IK statistics)"
    )

    args = parser.parse_args()

    if args.stats_only:
        print("###################### Generating statistics only ######################")
        print(f"###################### Plan directory: {args.plan_dir} ######################")
        stats = generate_statistics(args.plan_dir, args.robot_name)
        stats_file = os.path.join(args.plan_dir, args.robot_name, "pipeline_statistics.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"###################### Statistics saved to {stats_file} ######################")
        
        # Also collect clustering stats if requested
        if args.record_clustering_stats:
            print("#### Collecting clustering statistics from existing data... ####")
            clustering_stats_df = collect_clustering_statistics(args.plan_dir, args.robot_name)
            if clustering_stats_df is not None:
                stats_output_dir = Path("output/statistics")
                stats_output_dir.mkdir(parents=True, exist_ok=True)
                csv_file = stats_output_dir / f"clustering_stats_{args.robot_name}.csv"
                clustering_stats_df.to_csv(csv_file, index=False)
                print(f"###################### Clustering statistics saved to {csv_file} ######################")
                
                # Print summary
                print("\n#### Clustering Statistics Summary ####")
                print(f"Total records: {len(clustering_stats_df)}")
                print(f"\nClustering method distribution:")
                print(clustering_stats_df['clustering_method'].value_counts())
                print(f"\nAverage final IK count: {clustering_stats_df['final_ik_count'].mean():.2f}")
                print(f"Average AP unique labels: {clustering_stats_df['ap_unique_labels'].mean():.2f}")
            else:
                print("###################### No clustering statistics found ######################")
    else:
        if not args.scene_dir:
            print(
                "###################### Error: --scene-dir is required when not using --stats-only ######################"
            )
            return
        print("###################### Starting pipeline planning ######################")
        print(f"###################### Scene directory: {args.scene_dir} ######################")
        print(f"###################### Plan directory: {args.plan_dir} ######################")
        if args.record_clustering_stats:
            print("###################### Recording clustering statistics for ablation study ######################")
        if args.ik_only:
            print("###################### IK-only mode: trajectory planning will be skipped ######################")
        run_pipelines_for_directory(args.scene_dir, args.plan_dir, args.robot_name, args.record_clustering_stats, args.ik_only)


if __name__ == "__main__":
    main()
