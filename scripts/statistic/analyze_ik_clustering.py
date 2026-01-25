"""
Analyze IK clustering effectiveness by comparing raw vs clustered IK solutions.

This script demonstrates that clustered IK solutions effectively cover the 
original IK distribution by:
1. Generating raw IK without clustering (using large cluster parameter trick)
2. Generating clustered IK with default parameters
3. Visualizing coverage with t-SNE/UMAP
4. Computing statistics to show distribution preservation

Author: Statistical Analysis Suite for AutoMoMA
Date: 2025-10-16
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json

from config import *
from automoma.models.object import ObjectDescription
from automoma.models.robot import RobotDescription
from automoma.models.task import TaskDescription, TaskType
from automoma.pipeline import InfinigenScenePipeline, AOGraspPipeline
from automoma.utils.transform import single_axis_self_rotation, matrix_to_pose
from cuakr.planner.planner import AKRPlanner
from cuakr.utils.math import pose_multiply
from utils import (setup_plotting_style, compute_tsne, compute_umap, 
                  compute_joint_statistics, plot_2d_embedding,
                  plot_joint_distributions, save_figure)

from scipy.stats import wasserstein_distance, entropy


def compute_distribution_metrics(raw_ik: np.ndarray, clustered_ik: np.ndarray, 
                                joint_names: list) -> dict:
    """
    Compute metrics to evaluate clustering quality in joint space.
    
    Metrics include:
    - Wasserstein distance: measure of distribution discrepancy across all 10 robot joints
    - Coverage: percentage of raw range covered by clustered per joint
    - Mean/Std comparisons per joint
    
    Args:
        raw_ik: Raw IK solutions [N, 11] (last dim is object angle, excluded from metrics)
        clustered_ik: Clustered IK solutions [M, 11]
        joint_names: List of joint names (12 total, first 10 are robot joints)
        
    Returns:
        Dictionary with metrics per joint and aggregate
    """
    metrics = {
        'per_joint': {},
        'aggregate': {}
    }
    
    coverage_values = []
    
    # Compute per-joint metrics
    for i, joint_name in enumerate(joint_names):
        raw_vals = raw_ik[:, i]
        clustered_vals = clustered_ik[:, i]
        
        # Coverage: what fraction of raw range is covered by clustered
        raw_min, raw_max = np.min(raw_vals), np.max(raw_vals)
        clustered_min, clustered_max = np.min(clustered_vals), np.max(clustered_vals)
        
        if raw_max > raw_min:
            coverage = 100 * (min(raw_max, clustered_max) - max(raw_min, clustered_min)) / (raw_max - raw_min)
            coverage = max(0, coverage)
        else:
            coverage = 100
        coverage_values.append(coverage)
        
        # Mean difference
        raw_mean = np.mean(raw_vals)
        clustered_mean = np.mean(clustered_vals)
        mean_diff = abs(raw_mean - clustered_mean)
        
        # Std dev comparison
        raw_std = np.std(raw_vals)
        clustered_std = np.std(clustered_vals)
        std_ratio = clustered_std / (raw_std + 1e-8)
        
        metrics['per_joint'][joint_name] = {
            'coverage_percent': float(coverage),
            'mean_difference': float(mean_diff),
            'std_ratio': float(std_ratio),
            'raw_mean': float(raw_mean),
            'clustered_mean': float(clustered_mean),
            'raw_std': float(raw_std),
            'clustered_std': float(clustered_std)
        }
    
    # Compute Wasserstein distance across ALL 10 robot joints (exclude object angle dim 10)
    # This measures the overall distribution discrepancy in robot joint space
    raw_robot_joints = raw_ik[:, :-1]  # First 10 dims are robot joints
    clustered_robot_joints = clustered_ik[:, :-1]
    
    # Flatten to 1D for overall Wasserstein distance
    raw_flat = raw_robot_joints.flatten()
    clustered_flat = clustered_robot_joints.flatten()
    
    overall_wasserstein = wasserstein_distance(raw_flat, clustered_flat)
    
    # Aggregate metrics
    metrics['aggregate'] = {
        'wasserstein_distance_robot_joints': float(overall_wasserstein),  # MAIN METRIC
        'mean_coverage_percent': float(np.mean(coverage_values)),
        'min_coverage_percent': float(np.min(coverage_values)),
        'solution_reduction_percent': float(100 * (1 - len(clustered_ik) / len(raw_ik))),
        'solution_reduction_ratio': float(len(raw_ik) / len(clustered_ik))
    }
    
    return metrics


# Cache helper functions
def get_cache_path(scene_name: str, grasp_id: int, ik_type: str) -> Path:
    """Get path to cached IK data file.
    
    Args:
        scene_name: Scene name
        grasp_id: Grasp ID
        ik_type: 'raw' or 'clustered'
    """
    cache_dir = IK_CLUSTERING['cache_dir']
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{scene_name}_grasp_{grasp_id:04d}_{ik_type}_ik.pt"


def save_ik_to_cache(scene_name: str, grasp_id: int, ik_type: str, 
                     start_ik: torch.Tensor, goal_ik: torch.Tensor):
    """Save IK data to cache."""
    if not IK_CLUSTERING['enable_cache']:
        return
    
    cache_path = get_cache_path(scene_name, grasp_id, ik_type)
    torch.save({
        'start_ik': start_ik,
        'goal_ik': goal_ik,
        'scene_name': scene_name,
        'grasp_id': grasp_id,
        'ik_type': ik_type
    }, cache_path)
    print(f"  ✓ Cached {ik_type} IK to {cache_path.name}")


def load_ik_from_cache(scene_name: str, grasp_id: int, ik_type: str):
    """Load IK data from cache if available.
    
    Returns:
        Tuple of (start_ik, goal_ik) or None if not cached
    """
    if not IK_CLUSTERING['enable_cache']:
        return None
    
    cache_path = get_cache_path(scene_name, grasp_id, ik_type)
    if not cache_path.exists():
        return None
    
    try:
        data = torch.load(cache_path, map_location='cpu', weights_only=False)
        print(f"  ✓ Loaded {ik_type} IK from cache: {cache_path.name}")
        return data['start_ik'], data['goal_ik']
    except Exception as e:
        print(f"  ⚠️  Failed to load cache: {e}")
        return None


def create_object():
    """Create the microwave object."""
    object = ObjectDescription(
        asset_type="Microwave",
        asset_id="7221",
        scale=0.3562990018302636,
        urdf_path="assets/object/Microwave/7221/7221_0_scaling.urdf",
    )
    object.set_handle_link("link_0")
    return object


def load_scene_and_setup(scene_path: str):
    """Load scene and setup object."""
    scene_pipeline = InfinigenScenePipeline()
    object = create_object()
    scene_result = scene_pipeline.load_scene(scene_path, [object])
    
    scene_pose = [0, 0, -0.13, 1, 0, 0, 0]
    scene_result.scene.set_pose(scene_pose)
    
    # Set object pose
    for obj in scene_result.valid_objects:
        obj_pose = scene_result.scene.get_object_matrix(obj)
        obj_pose = single_axis_self_rotation(obj_pose, 'z', np.pi)
        obj_pose = matrix_to_pose(obj_pose)
        obj.set_pose(pose_multiply(scene_pose, obj_pose))
    
    return scene_result, object


def plan_ik_with_clustering(task: TaskDescription, grasp_pose: np.ndarray, 
                           use_clustering: bool = True):
    """
    Plan IK with or without clustering.
    
    Args:
        task: Task description
        grasp_pose: Grasp pose to use
        use_clustering: If False, bypass clustering by setting large cluster values
    """
    # Setup planner
    scene_cfg = {
        "path": task.scene.scene_usd_path,
        "pose": task.scene.pose,
    }
    object_cfg = {
        "path": task.object.urdf_path,
        "asset_type": task.object.asset_type,
        "asset_id": task.object.asset_id,
    }
    robot_cfg = task.robot.robot_cfg
    
    object_cfg = AKRPlanner.load_object_from_metadata(task.scene.metadata_path, object_cfg)
    planner = AKRPlanner(scene_cfg, object_cfg, robot_cfg)
    
    # Set clustering parameters
    if use_clustering:
        clustering_params = {
            'kmeans_clusters': IK_CLUSTERING['default_kmeans_clusters'],
            'ap_fallback_clusters': IK_CLUSTERING['default_ap_fallback_clusters']
        }
        print(f"  Using clustering: kmeans={clustering_params['kmeans_clusters']}, "
              f"ap_fallback={clustering_params['ap_fallback_clusters']}")
    else:
        # Trick: set very large values to bypass clustering
        no_cluster_val = IK_CLUSTERING['no_clustering_value']
        clustering_params = {
            'kmeans_clusters': no_cluster_val,
            'ap_fallback_clusters': no_cluster_val
        }
        print(f"  Bypassing clustering with large values: {no_cluster_val}")
    
    # Plan IK
    ik_result = planner.plan_ik(
        grasp_pose=grasp_pose,
        start_angle=0.0,
        goal_angle=1.57,
        robot_cfg=robot_cfg,
        clustering_params=clustering_params,
        handle_link="link_0"
    )
    
    return ik_result


def _visualize_clustering_coverage(raw_iks, clustered_iks, scene_name, grasp_id, ik_type, output_dir):
    """
    Visualize how clustered IK covers raw IK distribution.
    
    Strategy: Plot raw IK first (small dots), then clustered on top (larger diamonds)
    to show that clustered solutions cover the raw distribution.
    """
    all_data = np.vstack([raw_iks, clustered_iks])
    n_raw = len(raw_iks)
    n_clustered = len(clustered_iks)
    
    print(f"  Computing t-SNE for {ik_type} IKs ({len(all_data)} points)...")
    tsne_result = compute_tsne(all_data, perplexity=min(30, len(all_data)//3))
    
    raw_tsne = tsne_result[:n_raw]
    clustered_tsne = tsne_result[n_raw:]
    
    # Create visualization showing coverage
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Scatter plot with raw (small) and clustered (large, on top)
    ax = axes[0]
    ax.scatter(raw_tsne[:, 0], raw_tsne[:, 1],
              c=VIZ_CONFIG['color_raw'], alpha=VIZ_CONFIG['alpha_filtered_layer'], 
              s=VIZ_CONFIG['marker_size_small'], 
              label=f'Raw ({n_raw})')
    ax.scatter(clustered_tsne[:, 0], clustered_tsne[:, 1],
              c=VIZ_CONFIG['color_clustered'], alpha=VIZ_CONFIG['alpha_clustered_layer'], 
              s=VIZ_CONFIG['marker_size_large'],
              edgecolors='black', linewidths=VIZ_CONFIG['linewidth_thin'], marker='D', 
              label=f'Clustered ({n_clustered})')
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title(f'IK Clustering Coverage ({ik_type.capitalize()})\n{scene_name}, Grasp {grasp_id:04d}', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=VIZ_CONFIG['alpha_grid'])
    
    # Add reduction statistics
    reduction = 100 * (1 - n_clustered / n_raw)
    text_str = f'Reduction: {reduction:.1f}%\n({n_raw} → {n_clustered} solutions)'
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=VIZ_CONFIG['alpha_infobox']))
    
    # Right: Density contours showing coverage
    ax = axes[1]
    
    from scipy.stats import gaussian_kde
    
    # Raw IK density (base layer)
    if len(raw_tsne) > 10:
        kde_raw = gaussian_kde(raw_tsne.T)
        x_min, x_max = tsne_result[:, 0].min(), tsne_result[:, 0].max()
        y_min, y_max = tsne_result[:, 1].min(), tsne_result[:, 1].max()
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        z_raw = np.reshape(kde_raw(positions).T, xx.shape)
        ax.contourf(xx, yy, z_raw, levels=10, cmap='Blues', 
                   alpha=VIZ_CONFIG['alpha_density_fill'])
        ax.contour(xx, yy, z_raw, levels=VIZ_CONFIG['contour_levels'], 
                  colors=VIZ_CONFIG['color_raw'],
                  alpha=VIZ_CONFIG['alpha_density_line'], 
                  linewidths=VIZ_CONFIG['linewidth_thin'])
    
    # Clustered IK points on top
    ax.scatter(clustered_tsne[:, 0], clustered_tsne[:, 1],
              c=VIZ_CONFIG['color_clustered'], s=VIZ_CONFIG['marker_size_large'], 
              alpha=VIZ_CONFIG['alpha_clustered_points'],
              edgecolors='black', linewidths=VIZ_CONFIG['linewidth_medium'], marker='D', 
              label='Clustered')
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title(f'Density Coverage ({ik_type.capitalize()})\n{scene_name}, Grasp {grasp_id:04d}', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=VIZ_CONFIG['alpha_grid'])
    
    plt.tight_layout()
    save_figure(fig, output_dir / f"{scene_name}_grasp_{grasp_id:04d}_{ik_type}_ik_coverage")
    plt.close()
    
    print(f"  ✓ Saved {ik_type} IK coverage visualization")


def analyze_ik_clustering_single_grasp(scene_name: str, scene_path: str, 
                                       grasp_id: int, output_dir: Path):
    """Analyze IK clustering for a single grasp."""
    print(f"\n{'='*80}")
    print(f"Analyzing IK Clustering: {scene_name}, Grasp {grasp_id}")
    print(f"{'='*80}")
    
    # Try to load from cache first
    print("\n1. Checking cache...")
    raw_cached = load_ik_from_cache(scene_name, grasp_id, 'raw')
    clustered_cached = load_ik_from_cache(scene_name, grasp_id, 'clustered')
    
    if raw_cached is not None and clustered_cached is not None:
        print("  ✓ Using cached IK data (skipping planner)")
        raw_start_ik, raw_goal_ik = raw_cached
        clustered_start_ik, clustered_goal_ik = clustered_cached
    else:
        print("  ⚠️  Cache miss, running planner...")
        
        # Load scene and setup
        print("\n2. Loading scene...")
        scene_result, object = load_scene_and_setup(scene_path)
        
        # Generate grasp
        print("\n3. Generating grasp...")
        grasp_pipeline = AOGraspPipeline()
        grasps = grasp_pipeline.generate_grasps(object, 20)
        
        if grasp_id not in grasps:
            print(f"Grasp {grasp_id} not found!")
            return None
        
        grasp_pose = grasps[grasp_id]
        
        # Create task
        robot = RobotDescription("summit_franka", "assets/robot/summit_franka/summit_franka.yml")
        task = TaskDescription(
            robot=robot,
            object=object,
            scene=scene_result.scene,
            task_type=TaskType.ARTICULATE,
        )
        task.update_grasp(grasp_pose)
        
        # Plan IK without clustering (raw)
        print("\n4. Planning IK WITHOUT clustering (raw)...")
        raw_ik_result = plan_ik_with_clustering(task, grasp_pose, use_clustering=False)
        raw_start_ik = raw_ik_result.start_ik
        raw_goal_ik = raw_ik_result.goal_ik
        print(f"  Raw start IKs: {raw_start_ik.shape[0]}")
        print(f"  Raw goal IKs: {raw_goal_ik.shape[0]}")
        
        # Plan IK with clustering
        print("\n5. Planning IK WITH clustering...")
        clustered_ik_result = plan_ik_with_clustering(task, grasp_pose, use_clustering=True)
        clustered_start_ik = clustered_ik_result.start_ik
        clustered_goal_ik = clustered_ik_result.goal_ik
        print(f"  Clustered start IKs: {clustered_start_ik.shape[0]}")
        print(f"  Clustered goal IKs: {clustered_ik_result.goal_ik.shape[0]}")
        
        # Cache the results
        save_ik_to_cache(scene_name, grasp_id, 'raw', raw_start_ik, raw_goal_ik)
        save_ik_to_cache(scene_name, grasp_id, 'clustered', clustered_start_ik, clustered_goal_ik)
    
    # Convert to numpy
    raw_start = raw_start_ik.cpu().numpy()
    raw_goal = raw_goal_ik.cpu().numpy()
    clustered_start = clustered_start_ik.cpu().numpy()
    clustered_goal = clustered_goal_ik.cpu().numpy()
    
    # Compute statistics
    print("\n6. Computing statistics...")
    raw_start_stats = compute_joint_statistics(raw_start, JOINT_NAMES)
    clustered_start_stats = compute_joint_statistics(clustered_start, JOINT_NAMES)
    raw_goal_stats = compute_joint_statistics(raw_goal, JOINT_NAMES)
    clustered_goal_stats = compute_joint_statistics(clustered_goal, JOINT_NAMES)
    
    # Visualization: Better visualization showing clustered covering raw
    print("\n7. Generating coverage visualizations...")
    
    # Visualize START IKs with better coverage display
    _visualize_clustering_coverage(
        raw_start, clustered_start, scene_name, grasp_id, "start", output_dir
    )
    
    # Visualize GOAL IKs
    _visualize_clustering_coverage(
        raw_goal, clustered_goal, scene_name, grasp_id, "goal", output_dir
    )
    
    # Plot joint distributions comparison - single curve view for START IKs
    print("\n8. Generating joint distribution comparisons...")
    
    # Create single curve figure for start IKs showing all joints
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Prepare data for curve visualization
    x_positions = np.arange(len(JOINT_NAMES))
    
    # Calculate mean values for raw and clustered
    raw_means = np.mean(raw_start, axis=0)
    clustered_means = np.mean(clustered_start, axis=0)
    
    # Calculate ranges (max - min)
    raw_ranges = np.max(raw_start, axis=0) - np.min(raw_start, axis=0)
    clustered_ranges = np.max(clustered_start, axis=0) - np.min(clustered_start, axis=0)
    
    # Plot curves
    line_width = 3
    marker_size = 100
    
    # Raw IK curve
    ax.plot(x_positions, raw_means, 'o-', color=VIZ_CONFIG['color_raw'], 
           linewidth=line_width, markersize=10, label='Raw IK Mean',
           alpha=0.8)
    
    # Clustered IK curve
    ax.plot(x_positions, clustered_means, 's-', color=VIZ_CONFIG['color_clustered'], 
           linewidth=line_width, markersize=10, label='Clustered IK Mean',
           alpha=0.8)
    
    # Add fill between for range comparison
    ax.fill_between(x_positions, 
                     np.min(raw_start, axis=0), 
                     np.max(raw_start, axis=0),
                     alpha=0.15, color=VIZ_CONFIG['color_raw'], label='Raw IK Range')
    
    ax.fill_between(x_positions,
                     np.min(clustered_start, axis=0),
                     np.max(clustered_start, axis=0),
                     alpha=0.2, color=VIZ_CONFIG['color_clustered'], label='Clustered IK Range')
    
    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=11)
    
    ax.set_ylabel('Joint Value (rad)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Joint Name', fontsize=13, fontweight='bold')
    ax.set_title(f'Start IK Coverage Comparison: Raw vs Clustered\n{scene_name}, Grasp {grasp_id:04d}', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, axis='both')
    ax.legend(loc='best', fontsize=12, frameon=True)
    
    # Compute distribution metrics for start IKs
    start_metrics = compute_distribution_metrics(raw_start, clustered_start, JOINT_NAMES)
    
    # Add statistics box with prominent Wasserstein distance
    reduction = 100 * (1 - len(clustered_start) / len(raw_start))
    
    # Main metric: Wasserstein distance on 10 robot joints
    ws_dist = start_metrics['aggregate']['wasserstein_distance_robot_joints']
    
    stats_text = (f'Wasserstein Distance (10 Robot Joints): {ws_dist:.6f}\n'
                 f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
                 f'Solution Reduction: {reduction:.1f}% ({len(raw_start)}→{len(clustered_start)})\n'
                 f'Mean Coverage: {start_metrics["aggregate"]["mean_coverage_percent"]:.1f}%\n'
                 f'Min Coverage: {start_metrics["aggregate"]["min_coverage_percent"]:.1f}%')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=13, family='monospace', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, pad=0.8))
    
    plt.tight_layout()
    save_figure(fig, output_dir / f"{scene_name}_grasp_{grasp_id:04d}_start_coverage_comparison")
    plt.close()
    
    print(f"  ✓ Saved start IK coverage comparison (single curve view)")
    print(f"     Wasserstein Distance (10 Robot Joints): {start_metrics['aggregate']['wasserstein_distance_robot_joints']:.6f}")
    print(f"     Mean Coverage: {start_metrics['aggregate']['mean_coverage_percent']:.1f}%")
    
    # Create single curve figure for goal IKs showing all joints
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Prepare data for curve visualization
    x_positions = np.arange(len(JOINT_NAMES))
    
    # Calculate mean values for raw and clustered
    raw_means = np.mean(raw_goal, axis=0)
    clustered_means = np.mean(clustered_goal, axis=0)
    
    # Calculate ranges (max - min)
    raw_ranges = np.max(raw_goal, axis=0) - np.min(raw_goal, axis=0)
    clustered_ranges = np.max(clustered_goal, axis=0) - np.min(clustered_goal, axis=0)
    
    # Plot curves
    line_width = 3
    marker_size = 100
    
    # Raw IK curve
    ax.plot(x_positions, raw_means, 'o-', color=VIZ_CONFIG['color_raw'], 
           linewidth=line_width, markersize=10, label='Raw IK Mean',
           alpha=0.8)
    
    # Clustered IK curve
    ax.plot(x_positions, clustered_means, 's-', color=VIZ_CONFIG['color_clustered'], 
           linewidth=line_width, markersize=10, label='Clustered IK Mean',
           alpha=0.8)
    
    # Add fill between for range comparison
    ax.fill_between(x_positions, 
                     np.min(raw_goal, axis=0), 
                     np.max(raw_goal, axis=0),
                     alpha=0.15, color=VIZ_CONFIG['color_raw'], label='Raw IK Range')
    
    ax.fill_between(x_positions,
                     np.min(clustered_goal, axis=0),
                     np.max(clustered_goal, axis=0),
                     alpha=0.2, color=VIZ_CONFIG['color_clustered'], label='Clustered IK Range')
    
    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=11)
    
    ax.set_ylabel('Joint Value (rad)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Joint Name', fontsize=13, fontweight='bold')
    ax.set_title(f'Goal IK Coverage Comparison: Raw vs Clustered\n{scene_name}, Grasp {grasp_id:04d}', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, axis='both')
    ax.legend(loc='best', fontsize=12, frameon=True)
    
    # Compute distribution metrics for goal IKs
    goal_metrics = compute_distribution_metrics(raw_goal, clustered_goal, JOINT_NAMES)
    
    # Add statistics box with prominent Wasserstein distance
    reduction = 100 * (1 - len(clustered_goal) / len(raw_goal))
    
    # Main metric: Wasserstein distance on 10 robot joints
    ws_dist = goal_metrics['aggregate']['wasserstein_distance_robot_joints']
    
    stats_text = (f'Wasserstein Distance (10 Robot Joints): {ws_dist:.6f}\n'
                 f'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
                 f'Solution Reduction: {reduction:.1f}% ({len(raw_goal)}→{len(clustered_goal)})\n'
                 f'Mean Coverage: {goal_metrics["aggregate"]["mean_coverage_percent"]:.1f}%\n'
                 f'Min Coverage: {goal_metrics["aggregate"]["min_coverage_percent"]:.1f}%')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=11, family='monospace', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, pad=0.8)) 
    plt.tight_layout()
    save_figure(fig, output_dir / f"{scene_name}_grasp_{grasp_id:04d}_goal_coverage_comparison")
    plt.close()
    
    print(f"  ✓ Saved goal IK coverage comparison (single curve view)")
    print(f"     Wasserstein Distance (10 Robot Joints): {goal_metrics['aggregate']['wasserstein_distance_robot_joints']:.6f}")
    print(f"     Mean Coverage: {goal_metrics['aggregate']['mean_coverage_percent']:.1f}%")
    
    # Object angle comparison (11th dimension)
    obj_idx = TRAJ_ANALYSIS['object_joint_index']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(raw_start[:, obj_idx], bins=VIZ_CONFIG['histogram_bins'], 
                alpha=VIZ_CONFIG['alpha_histogram'], label='Raw', 
                color=VIZ_CONFIG['color_raw'], edgecolor='black')
    axes[0].hist(clustered_start[:, obj_idx], bins=VIZ_CONFIG['histogram_bins'], 
                alpha=VIZ_CONFIG['alpha_histogram'], label='Clustered',
                color=VIZ_CONFIG['color_clustered'], edgecolor='black')
    axes[0].set_xlabel('Object Angle (rad)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Start Object Angle - {scene_name}, Grasp {grasp_id:04d}')
    axes[0].legend()
    axes[0].grid(True, alpha=VIZ_CONFIG['alpha_grid'])
    
    axes[1].hist(raw_goal[:, obj_idx], bins=VIZ_CONFIG['histogram_bins'], 
                alpha=VIZ_CONFIG['alpha_histogram'], label='Raw',
                color=VIZ_CONFIG['color_raw'], edgecolor='black')
    axes[1].hist(clustered_goal[:, obj_idx], bins=VIZ_CONFIG['histogram_bins'], 
                alpha=VIZ_CONFIG['alpha_histogram'], label='Clustered',
                color=VIZ_CONFIG['color_clustered'], edgecolor='black')
    axes[1].set_xlabel('Object Angle (rad)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Goal Object Angle - {scene_name}, Grasp {grasp_id:04d}')
    axes[1].legend()
    axes[1].grid(True, alpha=VIZ_CONFIG['alpha_grid'])
    
    plt.tight_layout()
    save_figure(fig, output_dir / f"{scene_name}_grasp_{grasp_id:04d}_object_angle_comparison")
    plt.close()
    
    # Save metrics to JSON
    metrics_data = {
        'scene_name': scene_name,
        'grasp_id': grasp_id,
        'start_ik': {
            'metrics': start_metrics,
            'counts': {
                'raw': len(raw_start),
                'clustered': len(clustered_start)
            }
        },
        'goal_ik': {
            'metrics': goal_metrics,
            'counts': {
                'raw': len(raw_goal),
                'clustered': len(clustered_goal)
            }
        }
    }
    
    metrics_path = output_dir / f"{scene_name}_grasp_{grasp_id:04d}_clustering_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  ✓ Saved clustering metrics to {metrics_path.name}")
    
    # Return statistics for aggregation
    return {
        'scene_name': scene_name,
        'grasp_id': grasp_id,
        'raw_start_count': len(raw_start),
        'clustered_start_count': len(clustered_start),
        'raw_goal_count': len(raw_goal),
        'clustered_goal_count': len(clustered_goal),
        'start_reduction_percent': 100 * (1 - len(clustered_start) / len(raw_start)),
        'goal_reduction_percent': 100 * (1 - len(clustered_goal) / len(raw_goal)),
        'start_metrics': start_metrics,
        'goal_metrics': goal_metrics,
        'raw_start_stats': {k: v for k, v in raw_start_stats.items()},
        'clustered_start_stats': {k: v for k, v in clustered_start_stats.items()},
        'raw_goal_stats': {k: v for k, v in raw_goal_stats.items()},
        'clustered_goal_stats': {k: v for k, v in clustered_goal_stats.items()},
    }


def main():
    """Main IK clustering analysis pipeline."""
    print("=" * 80)
    print("IK Clustering Analysis")
    print("=" * 80)
    
    # Setup
    setup_plotting_style(VIZ_CONFIG)
    create_output_dirs()
    
    # Create cache directory
    IK_CLUSTERING['cache_dir'].mkdir(parents=True, exist_ok=True)
    
    output_dir = FIGURE_ROOT / "ik_clustering"
    
    # Use scenes and grasps from config
    analysis_scenes = IK_CLUSTERING['analysis_scenes']
    analysis_grasps = IK_CLUSTERING['analysis_grasp_ids']
    
    print(f"\nAnalysis Configuration:")
    print(f"  Scenes: {analysis_scenes}")
    print(f"  Grasp IDs: {analysis_grasps}")
    print(f"  Cache enabled: {IK_CLUSTERING['enable_cache']}")
    print(f"  Cache dir: {IK_CLUSTERING['cache_dir']}")
    
    all_results = []
    
    for test_scene in analysis_scenes:
        print(f"\n{'='*80}")
        print(f"Analyzing scene: {test_scene}")
        print(f"{'='*80}")
        
        # Get scene path
        scene_path = str(get_scene_path(test_scene))
        
        for grasp_id in analysis_grasps:
            try:
                result = analyze_ik_clustering_single_grasp(
                    test_scene, scene_path, grasp_id, output_dir
                )
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"Error analyzing grasp {grasp_id}: {e}")
                continue
    
    # Save summary
    summary = {
        'analyzed_scenes': analysis_scenes,
        'analyzed_grasps': analysis_grasps,
        'clustering_params': {
            'default_kmeans': IK_CLUSTERING['default_kmeans_clusters'],
            'default_ap_fallback': IK_CLUSTERING['default_ap_fallback_clusters'],
        },
        'cache_enabled': IK_CLUSTERING['enable_cache'],
        'results': all_results,
    }
    
    summary_path = OUTPUT_ROOT / "ik_clustering_analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("✓ IK Clustering Analysis Complete")
    print(f"✓ Analyzed {len(all_results)} grasps across {len(analysis_scenes)} scene(s)")
    print("="*80)


if __name__ == "__main__":
    main()
