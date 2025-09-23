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

Examples:
    python pipeline_plan.py --scene-dir /home/xinhai/Documents/automoma/output/test/kitchen_0919 --plan-dir output/summit_franka
    python pipeline_plan.py --plan-dir output/summit_franka --stats-only
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Any

# Add the src directory to the path so we can import automoma
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from automoma.models.object import ObjectDescription
from automoma.models.robot import RobotDescription
from automoma.models.task import TaskDescription, TaskType
from automoma.pipeline import ScenePipeline, TrajectoryPipeline, InfinigenScenePipeline, AOGraspPipeline
from automoma.utils.transform import single_axis_self_rotation, matrix_to_pose
from cuakr.utils.math import pose_multiply


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


def load_scene(scene_path: str, objects: list):
    """Load scene from path (same as in example)."""
    scene_pipeline = InfinigenScenePipeline()
    scene_result = scene_pipeline.load_scene(scene_path, objects)
    scene_pose = [0, 0, -0.13, 1, 0, 0, 0]
    scene_result.scene.set_pose(scene_pose)
    
    # set object poses
    for obj in scene_result.valid_objects:
        obj_pose = scene_result.scene.get_object_matrix(obj)
        obj_pose = single_axis_self_rotation(obj_pose, 'z', np.pi)
        obj_pose = matrix_to_pose(obj_pose)
        obj.set_pose(pose_multiply(scene_pose, obj_pose))
    
    return scene_result


def run_pipeline_for_scene(scene_path: str, scene_name: str, plan_dir: str, robot_name: str):
    """Run the complete pipeline for a single scene."""
    print("######################")
    print(f"#### Processing Scene: {scene_name} ####")
    print("######################")
    
    try:
        # Create object
        print("###################### Creating 7221 microwave object ######################")
        object = create_7221_object()
        
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
            robot=RobotDescription(robot_name,"assets/robot/summit_franka/summit_franka.yml"),
            object=object,
            scene=scene_result.scene,
            task_type=TaskType.ARTICULATE,
        )
        print("###################### Task created successfully ######################")
        
        # Create trajectory pipeline with custom output directory
        trajectory_pipeline = TrajectoryPipeline(task, output_base_dir=plan_dir)
        print(f"###################### Trajectory pipeline created with output dir: {plan_dir} ######################")

        # Process each grasp
        total_grasps = len(grasps)
        for grasp_id, grasp in grasps.items():
            print(f"######################")
            print(f"#### Processing Grasp {grasp_id}/{total_grasps} ####")
            print(f"######################")
            # if grasp_id % 5 == 0:
            #     continue

            # Update task with current grasp
            task.update_grasp(grasp)
            print("###################### Task updated with new grasp ######################")
            
            try:
                # Run pipeline steps
                trajectory_pipeline.load_akr_robot(f"assets/object/Microwave/7221/summit_franka_7221_0_grasp_{grasp_id:04d}.yml")
                print("###################### AKR robot loaded ######################")

                trajectory_pipeline.plan_ik()
                print("###################### IK planning completed ######################")

                trajectory_pipeline.plan_traj(batch_size=10)
                print("###################### Trajectory planning completed ######################")
                
                trajectory_pipeline.filter_traj()
                print("###################### Trajectory filtering completed ######################")
                
                trajectory_pipeline.save_results(grasp_id=grasp_id)
                print("###################### Results saved ######################")

                print(f"###################### Completed grasp {grasp_id}/{total_grasps} successfully ######################")
                
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


def run_pipelines_for_directory(scene_dir: str, plan_dir: str, robot_name: str):
    """Run pipelines for all scene subdirectories in a directory."""
    scene_path = Path(scene_dir)
    
    if not scene_path.exists():
        print(f"###################### Error: Scene directory {scene_dir} does not exist ######################")
        return
    
    # Create plan directory if it doesn't exist
    plan_path = Path(plan_dir)
    plan_path.mkdir(parents=True, exist_ok=True)
    print(f"###################### Plan directory: {plan_dir} ######################")
    
    scene_dirs = []
    if scene_path.name.startswith('scene_'):
        scene_dirs.append(scene_path)
    else:
    # Find all subdirectories that look like scenes
        for item in scene_path.iterdir():
            if item.is_dir() and item.name.startswith('scene_'):
                scene_dirs.append(item)
        
    scene_dirs.sort()  # Process in order
    
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
        scene_full_path = str(scene_subdir)

        success = run_pipeline_for_scene(scene_full_path, scene_name, plan_dir, robot_name)
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
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"###################### Statistics saved to {stats_file} ######################")


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
            "total_traj_results": 0,
            "total_filtered_results": 0,
            "ik_success_count": 0,
            "traj_success_count": 0,
            "filtered_success_count": 0,
            "average_ik_success_rate": 0.0,
            "average_traj_success_rate": 0.0,
            "average_filtered_success_rate": 0.0
        }
    }
    
    if not plan_path.exists():
        print(f"###################### Error: Plan directory {plan_dir} does not exist ######################")
        return stats
    
    # Find scene directories in plan directory
    scene_dirs = []
    for item in plan_path.iterdir():
        if item.is_dir() and item.name.startswith('scene_'):
            scene_dirs.append(item.name)
    
    scene_dirs.sort()
    stats["total_scenes"] = len(scene_dirs)
    
    for scene_name in scene_dirs:
        scene_output_dir = os.path.join(plan_dir, scene_name, "7221")
        
        scene_stats = {
            "scene_name": scene_name,
            "has_results": False,
            "total_grasps": 0,
            "successful_grasps": 0,
            "grasp_results": {}
        }
        
        if os.path.exists(scene_output_dir):
            scene_stats["has_results"] = True
            stats["scenes_with_results"] += 1
            
            # Find all grasp directories
            grasp_dirs = []
            for item in os.listdir(scene_output_dir):
                if item.startswith('grasp_') and os.path.isdir(os.path.join(scene_output_dir, item)):
                    grasp_dirs.append(item)
            
            grasp_dirs.sort()
            scene_stats["total_grasps"] = len(grasp_dirs)
            stats["total_grasps"] += len(grasp_dirs)
            
            # Analyze each grasp
            for grasp_dir in grasp_dirs:
                grasp_path = os.path.join(scene_output_dir, grasp_dir)
                grasp_stats = analyze_grasp_results(grasp_path)
                scene_stats["grasp_results"][grasp_dir] = grasp_stats
                
                # Update overall statistics
                if grasp_stats["ik_exists"]:
                    stats["overall_statistics"]["total_ik_results"] += 1
                    if grasp_stats["ik_success_count"] > 0:
                        stats["overall_statistics"]["ik_success_count"] += 1
                        
                if grasp_stats["traj_exists"]:
                    stats["overall_statistics"]["total_traj_results"] += 1
                    if grasp_stats["traj_success_count"] > 0:
                        stats["overall_statistics"]["traj_success_count"] += 1
                        
                if grasp_stats["filtered_exists"]:
                    stats["overall_statistics"]["total_filtered_results"] += 1
                    if grasp_stats["filtered_success_count"] > 0:
                        stats["overall_statistics"]["filtered_success_count"] += 1
                        scene_stats["successful_grasps"] += 1
        
        stats["statistics_by_scene"][scene_name] = scene_stats
    
    # Calculate averages
    if stats["overall_statistics"]["total_ik_results"] > 0:
        stats["overall_statistics"]["average_ik_success_rate"] = (
            stats["overall_statistics"]["ik_success_count"] / 
            stats["overall_statistics"]["total_ik_results"]
        )
        
    if stats["overall_statistics"]["total_traj_results"] > 0:
        stats["overall_statistics"]["average_traj_success_rate"] = (
            stats["overall_statistics"]["traj_success_count"] / 
            stats["overall_statistics"]["total_traj_results"]
        )
        
    if stats["overall_statistics"]["total_filtered_results"] > 0:
        stats["overall_statistics"]["average_filtered_success_rate"] = (
            stats["overall_statistics"]["filtered_success_count"] / 
            stats["overall_statistics"]["total_filtered_results"]
        )
    
    return stats


def analyze_grasp_results(grasp_path: str) -> Dict[str, Any]:
    """Analyze result files for a single grasp."""
    result = {
        "grasp_path": grasp_path,
        "ik_exists": False,
        "traj_exists": False, 
        "filtered_exists": False,
        "ik_total_count": 0,
        "ik_success_count": 0,
        "traj_total_count": 0,
        "traj_success_count": 0,
        "filtered_total_count": 0,
        "filtered_success_count": 0
    }
    
    # Check IK results
    ik_file = os.path.join(grasp_path, "ik_data.pt")
    if os.path.exists(ik_file):
        result["ik_exists"] = True
        try:
            ik_data = torch.load(ik_file, weights_only=False)
            if "start_ik" in ik_data:
                result["ik_total_count"] = len(ik_data["start_ik"])
                # Count successful IK solutions (those that exist and are valid)
                result["ik_success_count"] = sum(1 for ik in ik_data["start_ik"] if ik is not None)
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
        "--scene_dir", 
        type=str,
        help="Directory containing scene subdirectories (e.g., /path/to/kitchen_0919)"
    )
    parser.add_argument(
        "--plan_dir", 
        type=str,
        default="output",
        help="Directory to save planning results (e.g., output/summit_franka)"
    )
    parser.add_argument(
        "--stats_only",
        action="store_true", 
        help="Only generate statistics from existing results in plan-dir"
    )
    parser.add_argument(
        "--robot_name",
        type=str,
        default="summit_franka",
        help="Name of the robot to use for planning"
    )
    
    args = parser.parse_args()
    
    if args.stats_only:
        print("###################### Generating statistics only ######################")
        print(f"###################### Plan directory: {args.plan_dir} ######################")
        stats = generate_statistics(args.plan_dir, args.robot_name)
        stats_file = os.path.join(args.plan_dir, args.robot_name, "pipeline_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"###################### Statistics saved to {stats_file} ######################")
    else:
        if not args.scene_dir:
            print("###################### Error: --scene-dir is required when not using --stats-only ######################")
            return
        print("###################### Starting pipeline planning ######################")
        print(f"###################### Scene directory: {args.scene_dir} ######################")
        print(f"###################### Plan directory: {args.plan_dir} ######################")
        run_pipelines_for_directory(args.scene_dir, args.plan_dir, args.robot_name)


if __name__ == "__main__":
    main()