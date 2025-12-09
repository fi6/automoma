#!/usr/bin/env python3
"""
Pipeline Plan Script

This script provides two main functionalities:
1. Run planning pipeline: Given a scene directory with scene subdirectories, run the planning
   pipeline on each scene and save results to a plan directory
2. Generate statistics: Analyze results from a plan directory and generate statistics
   - Single-object statistics: Statistics for one object across scenes
   - Joint statistics: Combined statistics for all objects and scenes

Usage:
    # Run pipeline with separate scene and plan directories (default object: 7221)
    python pipeline_plan.py --scene-dir /path/to/scenes --plan-dir /path/to/output

    # Run pipeline with a specific object
    python pipeline_plan.py --scene-dir /path/to/scenes --plan-dir /path/to/output --object_id 11622

    # Generate statistics only from plan directory (single object)
    python pipeline_plan.py --plan-dir /path/to/output --stats-only --object_id 11622

    # Generate joint statistics across all objects and scenes
    python pipeline_plan.py --plan-dir /path/to/output --stats-only --joint_stats

    # Generate joint statistics for specific objects
    python pipeline_plan.py --plan-dir /path/to/output --stats-only --joint_stats --object_ids 7221 11622 46197

    # Run with clustering statistics recording (for ablation study)
    python pipeline_plan.py --scene-dir /path/to/scenes --plan-dir /path/to/output --object_id 7221 --record-clustering-stats

    # Run IK-only mode (skip trajectory planning, useful for IK statistics collection)
    python pipeline_plan.py --scene-dir /path/to/scenes --plan-dir /path/to/output --object_id 103634 --ik-only --record-clustering-stats

Examples:
    python scripts/pipeline_plan.py --scene_dir /home/xinhai/automoma/output/infinigen_scene_100 --plan_dir output/summit_franka --object_id 7221
    python pipeline_plan.py --plan-dir output/summit_franka --stats-only --object_id 7221
    python pipeline_plan.py --plan-dir output/collect_1205/traj --stats-only --joint_stats
    python pipeline_plan.py --scene_dir output/infinigen_scene_100 --plan_dir output/traj --object_id 11622 --ik-only --record-clustering-stats

Supported objects:
    - 7221: Microwave
    - 11622: Dishwasher
    - 103634: TrashCan
    - 46197: StorageFurniture (Cabinet)
    - 10944: Refrigerator
    - 101773: Oven
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

# Default object ID (can be overridden via --object_id argument)
OBJECT_ID = "7221"

# GRASP_IDS = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 18]
# GRASP_IDS = [4, 5, 6, 9, 12, 13, 18]

GRASP_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# GRASP_IDS = [0, 1, 2, 4, 5, 9, 11, 12, 13]
SCENE_IDS = [f"scene_{i}_seed_{i}" for i in range(0, 1)]
# SCENE_IDS = [f"scene_{i}_seed_{i}" for i in range(33, 70) if i not in [36, 39]]
# SCENE_IDS = [f"scene_{i}_seed_{i}" for i in range(0, 33) if i not in [5, 27]]  # scenes 0 to 31
# SCENE_IDS = [f"scene_{i}_seed_{i+101}" for i in range(0, 10)]  # scenes 0 to 31


# Robot configuration mapping - add more robots as needed
ROBOT_CONFIG_MAP = {
    "summit_franka": "assets/robot/summit_franka/summit_franka.yml",
    "summit_franka_fixed_base": "assets/robot/summit_franka/summit_franka_fixed_base.yml",
    # Add other robots here as needed
    # "other_robot": "assets/robot/other_robot/other_robot.yml",
}

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


class StatisticsPlan:
    """Class to manage statistics generation and updates for pipeline results."""
    
    def __init__(self, plan_dir: str, robot_name: str = None):
        """
        Initialize StatisticsPlan.
        
        Args:
            plan_dir: Base plan directory
            robot_name: Optional robot name filter (if None, process all robots)
        """
        self.plan_dir = Path(plan_dir)
        self.robot_name = robot_name
        self.stats_file = self.plan_dir / "pipeline_statistics.json"
        
        # Initialize or load existing statistics
        if self.stats_file.exists():
            with open(self.stats_file, 'r') as f:
                self.stats = json.load(f)
        else:
            self.stats = {
                "detail_data": {},
                "object_data": {},
                "scene_data": {}
            }
    
    def analyze_grasp(self, grasp_path: str) -> Dict[str, Any]:
        """Analyze result files for a single grasp."""
        result = {
            "ik_pairs": (0, 0),
            "traj_count": 0,
            "traj_filter1_count": 0,
            "traj_filter2_count": 0,
        }

        # Check IK results
        ik_file = os.path.join(grasp_path, "ik_data.pt")
        if os.path.exists(ik_file):
            try:
                ik_data = torch.load(ik_file, weights_only=False)
                start_count = ik_data["start_iks"].shape[0] if "start_iks" in ik_data else 0
                goal_count = ik_data["goal_iks"].shape[0] if "goal_iks" in ik_data else 0
                result["ik_pairs"] = (start_count, goal_count)
            except Exception as e:
                print(f"Warning: Could not load IK data from {ik_file}: {e}")

        # Check trajectory results (traj_count = success count from traj_data)
        traj_file = os.path.join(grasp_path, "traj_data.pt")
        if os.path.exists(traj_file):
            try:
                traj_data = torch.load(traj_file, weights_only=False)
                if "success" in traj_data:
                    result["traj_count"] = traj_data["success"].count_nonzero().item()
            except Exception as e:
                print(f"Warning: Could not load trajectory data from {traj_file}: {e}")

        # Check filtered results
        filtered_file = os.path.join(grasp_path, "filtered_traj_data.pt")
        if os.path.exists(filtered_file):
            try:
                filtered_data = torch.load(filtered_file, weights_only=False)
                if "success" in filtered_data:
                    result["traj_filter1_count"] = filtered_data["success"].shape[0]
                    result["traj_filter2_count"] = filtered_data["success"].count_nonzero().item()
            except Exception as e:
                print(f"Warning: Could not load filtered data from {filtered_file}: {e}")

        return result
    
    def update_grasp_stats(self, robot: str, scene_name: str, object_id: str, grasp_name: str):
        """Update statistics for a single grasp."""
        grasp_path = self.plan_dir / robot / scene_name / object_id / grasp_name
        
        if not grasp_path.exists():
            print(f"Warning: Grasp path does not exist: {grasp_path}")
            return
        
        # Analyze grasp
        grasp_stats = self.analyze_grasp(str(grasp_path))
        
        # Initialize nested dictionaries if needed
        if robot not in self.stats["detail_data"]:
            self.stats["detail_data"][robot] = {}
        if scene_name not in self.stats["detail_data"][robot]:
            self.stats["detail_data"][robot][scene_name] = {}
        if object_id not in self.stats["detail_data"][robot][scene_name]:
            self.stats["detail_data"][robot][scene_name][object_id] = {}
        
        # Update detail_data
        self.stats["detail_data"][robot][scene_name][object_id][grasp_name] = grasp_stats
        
        # Update object_data aggregation
        if object_id not in self.stats["object_data"]:
            self.stats["object_data"][object_id] = {}
        if robot not in self.stats["object_data"][object_id]:
            self.stats["object_data"][object_id][robot] = {}
        
        # Recalculate scene total for object_data
        scene_total = sum(
            grasp["traj_filter2_count"]
            for grasp in self.stats["detail_data"][robot][scene_name][object_id].values()
        )
        self.stats["object_data"][object_id][robot][scene_name] = scene_total
        
        # Update scene_data aggregation
        if scene_name not in self.stats["scene_data"]:
            self.stats["scene_data"][scene_name] = {}
        if robot not in self.stats["scene_data"][scene_name]:
            self.stats["scene_data"][scene_name][robot] = {}
        
        self.stats["scene_data"][scene_name][robot][object_id] = scene_total
    
    def generate_full_statistics(self):
        """Generate complete statistics by scanning the entire plan directory."""
        print("###################### Analyzing results for statistics ######################")
        
        if not self.plan_dir.exists():
            print(f"###################### Error: Plan directory {self.plan_dir} does not exist ######################")
            return
        
        # Reset statistics
        self.stats = {
            "detail_data": {},
            "object_data": {},
            "scene_data": {}
        }
        
        # Find all robot directories
        robot_dirs = []
        if self.robot_name:
            robot_path = self.plan_dir / self.robot_name
            if robot_path.exists() and robot_path.is_dir():
                robot_dirs = [self.robot_name]
        else:
            for item in self.plan_dir.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    robot_dirs.append(item.name)
        
        robot_dirs.sort()
        
        for robot in robot_dirs:
            robot_path = self.plan_dir / robot
            if not robot_path.exists():
                continue
                
            print(f"#### Processing robot: {robot} ####")
            self.stats["detail_data"][robot] = {}
            
            # Find all scene directories
            scene_dirs = []
            for item in robot_path.iterdir():
                if item.is_dir() and item.name.startswith("scene_"):
                    scene_dirs.append(item.name)
            
            scene_dirs.sort()
            
            for scene_name in scene_dirs:
                scene_path = robot_path / scene_name
                self.stats["detail_data"][robot][scene_name] = {}
                
                # Find all object directories
                object_dirs = []
                for item in scene_path.iterdir():
                    if item.is_dir() and item.name in OBJECT_CONFIG_MAP:
                        object_dirs.append(item.name)
                
                object_dirs.sort()
                
                for object_id in object_dirs:
                    object_path = scene_path / object_id
                    self.stats["detail_data"][robot][scene_name][object_id] = {}
                    
                    # Initialize object_data and scene_data if needed
                    if object_id not in self.stats["object_data"]:
                        self.stats["object_data"][object_id] = {}
                    if robot not in self.stats["object_data"][object_id]:
                        self.stats["object_data"][object_id][robot] = {}
                    
                    if scene_name not in self.stats["scene_data"]:
                        self.stats["scene_data"][scene_name] = {}
                    if robot not in self.stats["scene_data"][scene_name]:
                        self.stats["scene_data"][scene_name][robot] = {}
                    
                    # Find all grasp directories
                    grasp_dirs = []
                    for item in object_path.iterdir():
                        if item.is_dir() and item.name.startswith("grasp_"):
                            grasp_dirs.append(item.name)
                    
                    grasp_dirs.sort()
                    
                    scene_total_traj_filter2 = 0
                    
                    for grasp_name in grasp_dirs:
                        grasp_path = object_path / grasp_name
                        grasp_stats = self.analyze_grasp(str(grasp_path))
                        
                        self.stats["detail_data"][robot][scene_name][object_id][grasp_name] = grasp_stats
                        scene_total_traj_filter2 += grasp_stats["traj_filter2_count"]
                    
                    # Update object_data aggregation
                    self.stats["object_data"][object_id][robot][scene_name] = scene_total_traj_filter2
                    
                    # Update scene_data aggregation
                    self.stats["scene_data"][scene_name][robot][object_id] = scene_total_traj_filter2
    
    def save(self):
        """Save statistics to JSON file."""
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        with open(self.stats_file, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"###################### Statistics saved to {self.stats_file} ######################")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get the current statistics dictionary."""
        return self.stats


def run_pipeline_for_scene(
    scene_path: str,
    scene_name: str,
    plan_dir: str,
    robot_name: str,
    object_id: str,
    record_clustering_stats: bool = False,
    ik_only: bool = False,
    stats_plan: StatisticsPlan = None,
):
    """Run the complete pipeline for a single scene.

    Args:
        scene_path: Path to the scene directory
        scene_name: Name of the scene
        plan_dir: Directory to save planning results
        robot_name: Name of the robot to use
        object_id: ID of the object to manipulate
        record_clustering_stats: Whether to record clustering statistics
        ik_only: If True, only run IK planning without trajectory generation
    """
    print("######################")
    print(f"#### Processing Scene: {scene_name} ####")
    print(f"#### Object ID: {object_id} ####")
    if ik_only:
        print("#### Mode: IK-only (no trajectory planning) ####")
    print("######################")

    try:
        # Create object
        print(f"###################### Creating {object_id} object ######################")
        object = create_object(object_id)
        config = get_object_config(object_id)

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
        # if robot_name == "summit_franka_fixed_base":
        #     task.goal["angle"] *= 4
        if robot_name == "summit_franka":
            task.goal["angle"] *= 4
        print("###################### Task created successfully ######################")

        # Create trajectory pipeline with custom output directory
        trajectory_pipeline = TrajectoryPipeline(
            task, output_base_dir=plan_dir, record_clustering_stats=record_clustering_stats
        )
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
                akr_path = config["akr_path_template"].format(
                    object_id=object_id,
                    robot_name=robot_name,
                    grasp_id=grasp_id
                )
                trajectory_pipeline.load_akr_robot(akr_path)
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

                batch_size = 20
                trajectory_pipeline.plan_traj(batch_size=batch_size)
                print("###################### Trajectory planning completed ######################")

                trajectory_pipeline.filter_traj()
                print("###################### Trajectory filtering completed ######################")

                trajectory_pipeline.save_results(grasp_id=grasp_id)
                print("###################### Results saved ######################")

                # Update statistics if stats_plan is provided
                if stats_plan:
                    grasp_name = f"grasp_{grasp_id:04d}"
                    stats_plan.update_grasp_stats(robot_name, scene_name, object_id, grasp_name)
                    stats_plan.save()

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


def run_pipelines_for_directory(
    scene_dir: str,
    plan_dir: str,
    robot_name: str,
    object_id: str,
    record_clustering_stats: bool = False,
    ik_only: bool = False,
):
    """Run pipelines for all scene subdirectories in a directory.

    Args:
        scene_dir: Directory containing scene subdirectories
        plan_dir: Directory to save planning results
        robot_name: Name of the robot to use
        object_id: ID of the object to manipulate
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

    # Create StatisticsPlan instance
    stats_plan = StatisticsPlan(plan_dir, robot_name)

    for scene_subdir in scene_dirs:
        scene_name = scene_subdir.name
        if SCENE_IDS != None and scene_name not in SCENE_IDS:
            print(f"Skipping scene {scene_name} as it's not in the specified SCENE_IDS")
            continue
        scene_full_path = str(scene_subdir)

        success = run_pipeline_for_scene(
            scene_full_path, scene_name, plan_dir, robot_name, object_id, record_clustering_stats, ik_only, stats_plan
        )
        if success:
            successful_scenes += 1
        else:
            failed_scenes += 1

    print("######################")
    print("#### PIPELINE COMPLETE ####")
    print(f"#### Successful scenes: {successful_scenes} ####")
    print(f"#### Failed scenes: {failed_scenes} ####")
    print("#### Generating final statistics... ####")
    print("######################")

    # Generate final complete statistics
    stats_plan.generate_full_statistics()
    stats_plan.save()

    # Generate clustering statistics if recording
    if record_clustering_stats:
        print("#### Generating clustering statistics... ####")
        clustering_stats_df = collect_clustering_statistics(plan_dir, robot_name, object_id)
        if clustering_stats_df is not None:
            stats_output_dir = Path("output/statistics")
            stats_output_dir.mkdir(parents=True, exist_ok=True)
            csv_file = stats_output_dir / f"clustering_stats_{robot_name}_{object_id}.csv"
            clustering_stats_df.to_csv(csv_file, index=False)
            print(f"###################### Clustering statistics saved to {csv_file} ######################")
        else:
            print("###################### No clustering statistics found ######################")



def collect_clustering_statistics(plan_dir: str, robot_name: str, object_id: str) -> pd.DataFrame:
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
        object_dir = scene_dir / object_id

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
                        "object_id": object_id,
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
                        "object_id": object_id,
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


def main():
    """Main function to run the pipeline planner."""
    parser = argparse.ArgumentParser(
        description="Run pipeline planning for all scenes in a directory and generate statistics"
    )
    parser.add_argument(
        "--scene-dir", "--scene_dir", type=str, help="Directory containing scene subdirectories (e.g., /path/to/kitchen_0919)"
    )
    parser.add_argument(
        "--plan-dir", "--plan_dir", type=str, default="output", help="Directory to save planning results (e.g., output/summit_franka)"
    )
    parser.add_argument(
        "--object-id", "--object_id",
        type=str,
        default="7221",
        help=f"ID of the object to manipulate. Available objects: {', '.join(OBJECT_CONFIG_MAP.keys())}",
    )
    parser.add_argument(
        "--stats-only", "--stats_only", action="store_true", help="Only generate statistics from existing results in plan-dir"
    )
    parser.add_argument(
        "--joint-stats", "--joint_stats", action="store_true", help="Generate joint statistics across all objects and scenes"
    )
    parser.add_argument(
        "--object-ids", "--object_ids",
        type=str,
        nargs="+",
        help="List of object IDs for joint statistics (if not provided, auto-detect all objects)",
    )
    parser.add_argument("--robot_name", type=str, default="summit_franka", help="Name of the robot to use for planning")
    parser.add_argument(
        "--record-clustering-stats",
        action="store_true",
        help="Record detailed clustering statistics for ablation study (creates CSV in output/statistics/)",
    )
    parser.add_argument(
        "--ik-only",
        action="store_true",
        help="Only run IK planning without trajectory generation (useful for collecting IK statistics)",
    )

    args = parser.parse_args()

    # Validate object_id
    if not args.joint_stats and args.object_id not in OBJECT_CONFIG_MAP:
        print(f"###################### Error: Unknown object_id: {args.object_id} ######################")
        print(f"###################### Available objects: {', '.join(OBJECT_CONFIG_MAP.keys())} ######################")
        return

    if args.stats_only:
        # Generate statistics
        print("###################### Generating statistics only ######################")
        print(f"###################### Plan directory: {args.plan_dir} ######################")
        
        # Use StatisticsPlan class to generate statistics
        stats_plan = StatisticsPlan(args.plan_dir, args.robot_name if not args.joint_stats else None)
        stats_plan.generate_full_statistics()
        stats_plan.save()

        # Also collect clustering stats if requested
        if args.record_clustering_stats and args.object_id:
            print("#### Collecting clustering statistics from existing data... ####")
            clustering_stats_df = collect_clustering_statistics(args.plan_dir, args.robot_name, args.object_id)
            if clustering_stats_df is not None:
                stats_output_dir = Path("output/statistics")
                stats_output_dir.mkdir(parents=True, exist_ok=True)
                csv_file = stats_output_dir / f"clustering_stats_{args.robot_name}_{args.object_id}.csv"
                clustering_stats_df.to_csv(csv_file, index=False)
                print(f"###################### Clustering statistics saved to {csv_file} ######################")

                # Print summary
                print("\n#### Clustering Statistics Summary ####")
                print(f"Total records: {len(clustering_stats_df)}")
                print(f"\nClustering method distribution:")
                print(clustering_stats_df["clustering_method"].value_counts())
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
        print(f"###################### Object ID: {args.object_id} ######################")
        if args.record_clustering_stats:
            print("###################### Recording clustering statistics for ablation study ######################")
        if args.ik_only:
            print("###################### IK-only mode: trajectory planning will be skipped ######################")
        run_pipelines_for_directory(
            args.scene_dir, args.plan_dir, args.robot_name, args.object_id, args.record_clustering_stats, args.ik_only
        )


if __name__ == "__main__":
    main()
