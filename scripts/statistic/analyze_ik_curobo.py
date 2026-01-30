"""
Analyze IK solution distribution across different num_seeds values in cuRobo.

This script demonstrates how the number of IK seeds (num_seeds parameter) affects:
1. The number of raw IK solutions found
2. The distribution coverage in joint space
3. The diversity of solutions
4. Wasserstein distance to reference solution set

It generates beautiful visualizations showing:
- t-SNE embeddings overlaid with different num_seeds
- Density contours showing solution distribution
- Joint space histograms
- Wasserstein distance comparisons
- Solution count and diversity metrics

Author: Statistical Analysis Suite for AutoMoMA
Date: 2025-10-17
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import cdist
from tqdm import tqdm

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from config import *
from automoma.models.object import ObjectDescription
from automoma.models.robot import RobotDescription
from automoma.models.task import TaskDescription, TaskType
from automoma.pipeline import InfinigenScenePipeline, AOGraspPipeline
from automoma.utils.transform import single_axis_self_rotation, matrix_to_pose
from cuakr.planner.planner import AKRPlanner
from cuakr.utils.math import pose_multiply
from utils import (setup_plotting_style, compute_tsne, compute_umap, 
                  compute_joint_statistics, save_figure)


def compute_solution_diversity(ik_solutions: np.ndarray) -> Dict[str, float]:
    """
    Compute diversity metrics for IK solutions.
    
    Metrics include:
    - Mean pairwise distance: average distance between all solution pairs
    - Std of pairwise distances: how varied the distances are
    - Convex hull volume (proxy for coverage)
    
    Args:
        ik_solutions: IK solutions array [N, D] where N is number of solutions
        
    Returns:
        Dictionary with diversity metrics
    """
    if len(ik_solutions) < 2:
        return {
            'mean_pairwise_distance': 0.0,
            'std_pairwise_distance': 0.0,
            'coverage_radius': 0.0,
            'num_solutions': len(ik_solutions)
        }
    
    # Compute pairwise distances
    distances = cdist(ik_solutions, ik_solutions, metric='euclidean')
    
    # Get upper triangle (excluding diagonal)
    upper_tri_indices = np.triu_indices_from(distances, k=1)
    pairwise_distances = distances[upper_tri_indices]
    
    # Compute metrics
    mean_dist = np.mean(pairwise_distances)
    std_dist = np.std(pairwise_distances)
    
    # Coverage radius: mean distance from center to furthest point
    center = np.mean(ik_solutions, axis=0)
    distances_from_center = np.linalg.norm(ik_solutions - center, axis=1)
    coverage_radius = np.mean(distances_from_center)
    
    return {
        'mean_pairwise_distance': float(mean_dist),
        'std_pairwise_distance': float(std_dist),
        'coverage_radius': float(coverage_radius),
        'num_solutions': len(ik_solutions)
    }


def compute_wasserstein_to_reference(ik_solutions: np.ndarray, 
                                     reference_ik: np.ndarray) -> float:
    """
    Compute Wasserstein distance to reference IK solutions.
    
    This measures how well the current solution set represents the reference.
    Lower value = better coverage of reference distribution.
    
    Args:
        ik_solutions: Current IK solutions [N, D]
        reference_ik: Reference IK solutions [M, D]
        
    Returns:
        Wasserstein distance (flattened across all dimensions)
    """
    # Flatten to 1D for overall Wasserstein distance
    sol_flat = ik_solutions.flatten()
    ref_flat = reference_ik.flatten()
    
    return float(wasserstein_distance(sol_flat, ref_flat))


def compute_coverage_score(clustered_solutions: np.ndarray, 
                          reference_ik: np.ndarray) -> float:
    """
    Compute coverage score: how well clustered solutions represent original distribution.
    
    This metric measures the quality of clustering by computing the average distance
    from each reference point to its nearest clustered point.
    
    Lower value = clustered solutions are closer to the original distribution = better coverage.
    
    Args:
        clustered_solutions: Clustered IK solutions [N_clusters, D]
        reference_ik: Reference (original) IK solutions [M, D]
        
    Returns:
        Coverage score: mean distance from reference points to nearest cluster center
    """
    # Compute distance from each reference point to nearest clustered solution
    distances = cdist(reference_ik, clustered_solutions, metric='euclidean')
    
    # For each reference point, find distance to nearest clustered solution
    min_distances = np.min(distances, axis=1)
    
    # Coverage score is the average of these minimum distances
    coverage_score = float(np.mean(min_distances))
    
    return coverage_score


def cache_path(scene_name: str, grasp_id: int, num_seeds: int) -> Path:
    """Get cache path for IK solutions with specific num_seeds."""
    cache_dir = IK_CUROBO['cache_dir']
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{scene_name}_grasp_{grasp_id:04d}_seeds_{num_seeds}_ik.pt"


def save_ik_to_cache(scene_name: str, grasp_id: int, num_seeds: int,
                     start_ik: torch.Tensor, goal_ik: torch.Tensor):
    """Save IK solutions to cache."""
    if not IK_CUROBO['enable_cache']:
        return
    
    path = cache_path(scene_name, grasp_id, num_seeds)
    torch.save({
        'start_ik': start_ik,
        'goal_ik': goal_ik,
        'num_seeds': num_seeds,
        'scene_name': scene_name,
        'grasp_id': grasp_id
    }, path)


def load_ik_from_cache(scene_name: str, grasp_id: int, num_seeds: int) -> Optional[Tuple]:
    """Load IK solutions from cache if available."""
    if not IK_CUROBO['enable_cache']:
        return None
    
    path = cache_path(scene_name, grasp_id, num_seeds)
    if not path.exists():
        return None
    
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
        return (data['start_ik'], data['goal_ik'])
    except Exception as e:
        print(f"    ⚠️  Failed to load cache: {e}")
        return None


def cache_coverage_curve_path(scene_name: str, grasp_id: int, ik_type: str, 
                               curve_type: str) -> Path:
    """Get cache path for coverage curve data (curve_type: 'numseeds' or 'clustering')."""
    cache_dir = IK_CUROBO['cache_dir']
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{scene_name}_grasp_{grasp_id:04d}_{ik_type}_{curve_type}_coverage_curve.pt"


def save_coverage_curve(scene_name: str, grasp_id: int, ik_type: str, curve_type: str,
                       coverage_values: np.ndarray, x_values: np.ndarray):
    """Save coverage curve data to cache."""
    if not IK_CUROBO['enable_cache']:
        return
    
    path = cache_coverage_curve_path(scene_name, grasp_id, ik_type, curve_type)
    torch.save({
        'coverage_values': coverage_values,
        'x_values': x_values,
        'scene_name': scene_name,
        'grasp_id': grasp_id,
        'ik_type': ik_type,
        'curve_type': curve_type
    }, path)
    print(f"    ✓ Saved {curve_type} coverage curve to cache")


def load_coverage_curve(scene_name: str, grasp_id: int, ik_type: str, 
                       curve_type: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load coverage curve data from cache if available."""
    if not IK_CUROBO['enable_cache']:
        return None
    
    path = cache_coverage_curve_path(scene_name, grasp_id, ik_type, curve_type)
    if not path.exists():
        return None
    
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
        return (data['coverage_values'], data['x_values'])
    except Exception as e:
        print(f"    ⚠️  Failed to load coverage curve cache: {e}")
        return None


def create_object() -> ObjectDescription:
    """Create the microwave object."""
    obj = ObjectDescription(
        asset_type="Microwave",
        asset_id="7221",
        scale=0.3562990018302636,
        urdf_path="assets/object/Microwave/7221/7221_0_scaling.urdf",
    )
    obj.set_handle_link("link_0")
    return obj


def load_scene_and_setup(scene_path: str) -> Tuple:
    """Load scene and setup object."""
    scene_pipeline = InfinigenScenePipeline()
    obj = create_object()
    scene_result = scene_pipeline.load_scene(scene_path, [obj])
    
    scene_pose = [0, 0, -0.13, 1, 0, 0, 0]
    scene_result.scene.set_pose(scene_pose)
    
    for o in scene_result.valid_objects:
        obj_pose = scene_result.scene.get_object_matrix(o)
        obj_pose = single_axis_self_rotation(obj_pose, 'z', np.pi)
        obj_pose = matrix_to_pose(obj_pose)
        o.set_pose(pose_multiply(scene_pose, obj_pose))
    
    return scene_result, obj


def plan_ik_with_num_seeds(task: TaskDescription, grasp_pose: np.ndarray, 
                           num_seeds: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Plan IK solutions with specific num_seeds value.
    
    Args:
        task: Task description
        grasp_pose: Grasp pose
        num_seeds: Number of seeds for cuRobo IK solver
        
    Returns:
        Tuple of (start_ik, goal_ik) tensors
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
    
    object_cfg = AKRPlanner.load_object_from_metadata(
        task.scene.metadata_path, object_cfg
    )
    
    # Create planner
    robot_cfg = task.robot.robot_cfg
    planner = AKRPlanner(scene_cfg, object_cfg, robot_cfg)
    
    # Bypass clustering by using no_clustering_value to get raw IK data
    # This ensures we're comparing raw IK distributions without clustering effects
    no_cluster_val = IK_CUROBO['no_clustering_value']
    clustering_params = {
        'kmeans_clusters': no_cluster_val,
        'ap_fallback_clusters': no_cluster_val
    }
    
    # Plan IK with specific num_seeds - modified to pass num_seeds to _solve_ik
    ik_result = planner.plan_ik(
        grasp_pose=grasp_pose,
        start_angle=0.0,
        goal_angle=1.57,
        robot_cfg=robot_cfg,
        clustering_params=clustering_params,
        handle_link="link_0",
        num_ik_seeds=num_seeds  # This parameter needs to be added to plan_ik
    )
    
    return ik_result.start_ik, ik_result.goal_ik


def visualize_ik_distribution_multi_seeds(ik_data: Dict[int, np.ndarray],
                                         scene_name: str, 
                                         grasp_id: int, 
                                         ik_type: str,
                                         output_dir: Path):
    """
    Create comprehensive visualization comparing IK distributions across num_seeds.
    
    This creates a multi-panel figure showing:
    1. t-SNE with all seeds overlaid
    2. Individual density plots for each seed value
    3. Cumulative density showing coverage growth
    """
    num_seeds_values = sorted(ik_data.keys())
    n_seeds = len(num_seeds_values)
    
    print(f"  Generating multi-seed visualization for {ik_type} IKs...")
    
    # ========== Panel 1: t-SNE with all seeds overlaid ==========
    print(f"    Computing t-SNE...")
    
    # Combine all data for t-SNE
    all_data = np.vstack([ik_data[ns] for ns in num_seeds_values])
    tsne_all = compute_tsne(all_data, perplexity=min(30, len(all_data)//3))
    
    # Split back into individual seeds
    idx = 0
    tsne_data = {}
    for ns in num_seeds_values:
        n = len(ik_data[ns])
        tsne_data[ns] = tsne_all[idx:idx+n]
        idx += n
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Main t-SNE plot (spanning top-left and top-middle)
    ax_main = fig.add_subplot(gs[0, :2])
    
    # Plot each num_seeds with its color and marker
    # Plot in REVERSE order (largest to smallest) so smaller seeds appear on top and are visible
    for ns in reversed(num_seeds_values):
        color = IK_CUROBO_VIZ['colors'].get(ns, '#999999')
        marker = IK_CUROBO_VIZ['markers'].get(ns, 'o')
        # Adjust sizes: smaller num_seeds get larger points for visibility
        base_size = IK_CUROBO_VIZ['marker_sizes'].get(ns, 50)
        # Inverse relationship: smaller num_seeds = larger points
        size_factor = (num_seeds_values[-1] / ns) if ns > 0 else 1
        size = base_size * (0.3 + 0.7 * size_factor)  # Scale between 30-100% of base
        
        # Adjust alphas: smaller num_seeds more visible
        base_alpha = IK_CUROBO_VIZ['alphas'].get(ns, 0.6)
        # Increase alpha for smaller num_seeds
        alpha_factor = (ns / num_seeds_values[-1]) if num_seeds_values[-1] > 0 else 1
        alpha = base_alpha * (0.5 + 0.5 * alpha_factor)  # Scale between 50-100% of base
        
        ax_main.scatter(
            tsne_data[ns][:, 0], tsne_data[ns][:, 1],
            c=color, marker=marker, s=size, alpha=alpha,
            label=f'Seeds: {ns} (N={len(ik_data[ns])})',
            edgecolors='black' if ns == IK_CUROBO['reference_num_seeds'] else 'none',
            linewidths=2 if ns == IK_CUROBO['reference_num_seeds'] else 0
        )
    
    ax_main.set_xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
    ax_main.set_title(f't-SNE: IK Distribution Across num_seeds\n{scene_name}, Grasp {grasp_id:04d}, {ik_type.upper()}',
                     fontsize=14, fontweight='bold')
    ax_main.legend(loc='best', fontsize=10, title='num_seeds (N=count)', title_fontsize=11)
    ax_main.grid(True, alpha=0.3)
    
    # Top-right: Solution count vs num_seeds
    ax_count = fig.add_subplot(gs[0, 2])
    counts = [len(ik_data[ns]) for ns in num_seeds_values]
    colors_list = [IK_CUROBO_VIZ['colors'].get(ns, '#999999') for ns in num_seeds_values]
    
    bars = ax_count.bar(range(n_seeds), counts, color=colors_list, edgecolor='black', linewidth=1.5)
    ax_count.set_xticks(range(n_seeds))
    ax_count.set_xticklabels([str(ns) for ns in num_seeds_values], rotation=45, ha='right')
    ax_count.set_xlabel('num_seeds', fontsize=11, fontweight='bold')
    ax_count.set_ylabel('IK Solutions Found', fontsize=11, fontweight='bold')
    ax_count.set_title('Solution Count', fontsize=12, fontweight='bold')
    ax_count.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax_count.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(count)}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Bottom row: Density plots for reference and extremes
    reference_seeds = IK_CUROBO['reference_num_seeds']
    
    # Bottom-left: Smallest seeds density
    smallest_seeds = num_seeds_values[0]
    ax_small = fig.add_subplot(gs[1, 0])
    
    if len(ik_data[smallest_seeds]) > 10:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(tsne_data[smallest_seeds].T)
        x_min, x_max = tsne_data[smallest_seeds][:, 0].min(), tsne_data[smallest_seeds][:, 0].max()
        y_min, y_max = tsne_data[smallest_seeds][:, 1].min(), tsne_data[smallest_seeds][:, 1].max()
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        z = np.reshape(kde(positions).T, xx.shape)
        
        ax_small.contourf(xx, yy, z, levels=10, cmap='Reds', alpha=0.7)
        ax_small.scatter(tsne_data[smallest_seeds][:, 0], tsne_data[smallest_seeds][:, 1],
                        c=IK_CUROBO_VIZ['colors'][smallest_seeds], s=20, alpha=0.3, edgecolors='black', linewidth=0.5)
    
    ax_small.set_xlabel('t-SNE 1', fontsize=10)
    ax_small.set_ylabel('t-SNE 2', fontsize=10)
    ax_small.set_title(f'Density: num_seeds={smallest_seeds}', fontsize=11, fontweight='bold')
    ax_small.grid(True, alpha=0.2)
    
    # Bottom-middle: Reference density
    ax_ref = fig.add_subplot(gs[1, 1])
    
    if reference_seeds in ik_data and len(ik_data[reference_seeds]) > 10:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(tsne_data[reference_seeds].T)
        x_min, x_max = tsne_data[reference_seeds][:, 0].min(), tsne_data[reference_seeds][:, 0].max()
        y_min, y_max = tsne_data[reference_seeds][:, 1].min(), tsne_data[reference_seeds][:, 1].max()
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        z = np.reshape(kde(positions).T, xx.shape)
        
        ax_ref.contourf(xx, yy, z, levels=10, cmap='Greens', alpha=0.7)
        ax_ref.scatter(tsne_data[reference_seeds][:, 0], tsne_data[reference_seeds][:, 1],
                      c=IK_CUROBO_VIZ['colors'][reference_seeds], s=30, alpha=0.4, edgecolors='black', linewidth=0.5)
    
    ax_ref.set_xlabel('t-SNE 1', fontsize=10)
    ax_ref.set_ylabel('t-SNE 2', fontsize=10)
    ax_ref.set_title(f'Density: num_seeds={reference_seeds} (REF)', fontsize=11, fontweight='bold')
    ax_ref.grid(True, alpha=0.2)
    
    # Bottom-right: Largest seeds density
    largest_seeds = num_seeds_values[-1]
    ax_large = fig.add_subplot(gs[1, 2])
    
    if len(ik_data[largest_seeds]) > 10:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(tsne_data[largest_seeds].T)
        x_min, x_max = tsne_data[largest_seeds][:, 0].min(), tsne_data[largest_seeds][:, 0].max()
        y_min, y_max = tsne_data[largest_seeds][:, 1].min(), tsne_data[largest_seeds][:, 1].max()
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        z = np.reshape(kde(positions).T, xx.shape)
        
        ax_large.contourf(xx, yy, z, levels=10, cmap='Purples', alpha=0.7)
        ax_large.scatter(tsne_data[largest_seeds][:, 0], tsne_data[largest_seeds][:, 1],
                        c=IK_CUROBO_VIZ['colors'][largest_seeds], s=20, alpha=0.3, edgecolors='black', linewidth=0.5)
    
    ax_large.set_xlabel('t-SNE 1', fontsize=10)
    ax_large.set_ylabel('t-SNE 2', fontsize=10)
    ax_large.set_title(f'Density: num_seeds={largest_seeds}', fontsize=11, fontweight='bold')
    ax_large.grid(True, alpha=0.2)
    
    plt.suptitle(f'IK cuRobo Analysis: Impact of num_seeds\n{scene_name}, Grasp {grasp_id:04d}',
                fontsize=16, fontweight='bold', y=0.995)
    
    save_figure(fig, output_dir / f"{scene_name}_grasp_{grasp_id:04d}_{ik_type}_multiseed_analysis")
    plt.close()
    
    print(f"    ✓ Saved multi-seed visualization")


def visualize_joint_space_comparison(ik_data: Dict[int, np.ndarray],
                                    scene_name: str,
                                    grasp_id: int,
                                    ik_type: str,
                                    output_dir: Path):
    """
    Create joint space visualizations showing how num_seeds affects joint coverage.
    
    Creates a 5x3 subplot figure with histograms for each joint across num_seeds.
    """
    num_seeds_values = sorted(ik_data.keys())
    n_joints = ik_data[num_seeds_values[0]].shape[1]
    
    print(f"  Generating joint space comparison for {ik_type} IKs...")
    
    # Create figure with subplots for all joints (show only first 10 robot joints, not object angle)
    n_joints_to_show = min(10, n_joints - 1)  # Exclude object angle
    n_cols = 3
    n_rows = (n_joints_to_show + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten()  # Flatten for easier indexing
    
    for j in range(n_joints_to_show):
        ax = axes[j]
        
        # Plot histogram for each num_seeds
        for ns in num_seeds_values:
            joint_values = ik_data[ns][:, j]
            color = IK_CUROBO_VIZ['colors'].get(ns, '#999999')
            alpha = IK_CUROBO_VIZ['alphas'].get(ns, 0.6)
            
            ax.hist(joint_values, bins=IK_CUROBO_VIZ['histogram_bins'],
                   alpha=alpha, label=f'Seeds={ns}', color=color, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Joint Value (rad)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{JOINT_NAMES[j]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Remove extra subplots
    for j in range(n_joints_to_show, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle(f'Joint Space Coverage: num_seeds Comparison\n{scene_name}, Grasp {grasp_id:04d}, {ik_type.upper()}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_figure(fig, output_dir / f"{scene_name}_grasp_{grasp_id:04d}_{ik_type}_joint_space")
    plt.close()
    
    print(f"    ✓ Saved joint space comparison")


def visualize_ik_coverage_comparison(ik_data: Dict[int, np.ndarray],
                                     scene_name: str,
                                     grasp_id: int,
                                     ik_type: str,
                                     output_dir: Path):
    """
    Create coverage comparison visualization showing joint ranges and means across num_seeds.
    
    Shows min/max range as filled areas and mean as lines for each num_seeds value.
    """
    num_seeds_values = sorted(ik_data.keys())
    n_joints = ik_data[num_seeds_values[0]].shape[1]
    n_joints_to_show = min(10, n_joints - 1)  # Exclude object angle
    
    print(f"  Generating coverage comparison for {ik_type} IKs...")
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    for j in range(n_joints_to_show):
        ax = axes[j]
        
        # X-axis will be num_seeds values
        x_pos = np.arange(len(num_seeds_values))
        
        # For each joint, compute min, max, mean for each num_seeds
        means = []
        mins = []
        maxs = []
        stds = []
        
        for ns in num_seeds_values:
            joint_values = ik_data[ns][:, j]
            means.append(np.mean(joint_values))
            mins.append(np.min(joint_values))
            maxs.append(np.max(joint_values))
            stds.append(np.std(joint_values))
        
        means = np.array(means)
        mins = np.array(mins)
        maxs = np.array(maxs)
        stds = np.array(stds)
        
        # Plot range as filled area for each num_seeds
        ranges = maxs - mins
        for i, ns in enumerate(num_seeds_values):
            color = IK_CUROBO_VIZ['colors'].get(ns, '#999999')
            # Plot vertical line for range
            ax.vlines(i, mins[i], maxs[i], colors=color, linewidth=3, alpha=0.6, 
                     label=f'Seeds={ns}' if j == 0 else "")
            # Plot mean point
            ax.scatter(i, means[i], s=100, c=color, marker='o', edgecolors='black', 
                      linewidths=1.5, zorder=5)
        
        # Connect means with lines
        ax.plot(x_pos, means, 'k--', linewidth=1.5, alpha=0.3, zorder=2)
        
        ax.set_xlabel('num_seeds', fontsize=10)
        ax.set_ylabel('Joint Value (rad)', fontsize=10)
        ax.set_title(f'{JOINT_NAMES[j]}', fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(ns) for ns in num_seeds_values], rotation=45, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
    
    # Remove extra subplots
    for j in range(n_joints_to_show, len(axes)):
        fig.delaxes(axes[j])
    
    # Add legend to first subplot
    if n_joints_to_show > 0:
        handles = []
        labels = []
        for ns in num_seeds_values:
            color = IK_CUROBO_VIZ['colors'].get(ns, '#999999')
            handles.append(plt.Line2D([0], [0], color=color, linewidth=3))
            labels.append(f'Seeds={ns}')
        axes[0].legend(handles, labels, loc='upper left', fontsize=9, ncol=2)
    
    plt.suptitle(f'IK Coverage Comparison: Range and Mean\n{scene_name}, Grasp {grasp_id:04d}, {ik_type.upper()}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_figure(fig, output_dir / f"{scene_name}_grasp_{grasp_id:04d}_{ik_type}_coverage_comparison")
    plt.close()
    
    print(f"    ✓ Saved coverage comparison")


def analyze_single_grasp(scene_name: str, scene_path: str, grasp_id: int, output_dir: Path):
    """Analyze IK solutions for a single grasp across different num_seeds."""
    print(f"\n{'='*80}")
    print(f"Analyzing num_seeds Impact: {scene_name}, Grasp {grasp_id}")
    print(f"{'='*80}")
    
    num_seeds_values = IK_CUROBO['num_seeds_values']
    reference_seeds = IK_CUROBO['reference_num_seeds']
    
    # Try loading from cache first
    print("\n1. Checking cache...")
    all_cached = True
    ik_data_start = {}
    ik_data_goal = {}
    
    for ns in num_seeds_values:
        cached = load_ik_from_cache(scene_name, grasp_id, ns)
        if cached is not None:
            start_ik, goal_ik = cached
            ik_data_start[ns] = start_ik.cpu().numpy() if torch.is_tensor(start_ik) else start_ik
            ik_data_goal[ns] = goal_ik.cpu().numpy() if torch.is_tensor(goal_ik) else goal_ik
            print(f"  ✓ Loaded seeds={ns} from cache: start={ik_data_start[ns].shape}, goal={ik_data_goal[ns].shape}")
        else:
            all_cached = False
            break
    
    if not all_cached:
        print("  ⚠️  Cache miss, running planner...")
        
        # Load scene and setup
        print("\n2. Loading scene...")
        scene_result, obj = load_scene_and_setup(scene_path)
        
        # Generate grasps
        print("\n3. Generating grasps...")
        grasp_pipeline = AOGraspPipeline()
        grasps = grasp_pipeline.generate_grasps(obj, IK_CUROBO['num_grasps_to_generate'])
        
        if grasp_id not in grasps:
            print(f"    ✗ Grasp {grasp_id} not found!")
            return None
        
        grasp_pose = grasps[grasp_id]
        
        # Create task
        robot = RobotDescription("summit_franka", "assets/robot/summit_franka/summit_franka.yml")
        task = TaskDescription(
            robot=robot,
            object=obj,
            scene=scene_result.scene,
            task_type=TaskType.ARTICULATE,
        )
        task.update_grasp(grasp_pose)
        
        # Plan IK for each num_seeds value
        print("\n4. Planning IK with different num_seeds...")
        for i, ns in enumerate(num_seeds_values, 1):
            print(f"  [{i}/{len(num_seeds_values)}] num_seeds={ns}...")
            try:
                start_ik, goal_ik = plan_ik_with_num_seeds(task, grasp_pose, ns)
                ik_data_start[ns] = start_ik.cpu().numpy() if torch.is_tensor(start_ik) else start_ik
                ik_data_goal[ns] = goal_ik.cpu().numpy() if torch.is_tensor(goal_ik) else goal_ik
                
                print(f"      ✓ start={ik_data_start[ns].shape}, goal={ik_data_goal[ns].shape}")
                
                # Cache results
                save_ik_to_cache(scene_name, grasp_id, ns, start_ik, goal_ik)
            except Exception as e:
                print(f"      ✗ Error: {e}")
                return None
    
    # Convert to numpy if needed
    ik_data_start = {ns: (v.cpu().numpy() if torch.is_tensor(v) else v) 
                     for ns, v in ik_data_start.items()}
    ik_data_goal = {ns: (v.cpu().numpy() if torch.is_tensor(v) else v) 
                   for ns, v in ik_data_goal.items()}
    
    # Compute statistics
    print("\n5. Computing statistics...")
    stats = {
        'scene_name': scene_name,
        'grasp_id': grasp_id,
        'start_ik': {},
        'goal_ik': {},
    }
    
    for ns in num_seeds_values:
        start_stats = compute_joint_statistics(ik_data_start[ns], JOINT_NAMES)
        goal_stats = compute_joint_statistics(ik_data_goal[ns], JOINT_NAMES)
        
        start_diversity = compute_solution_diversity(ik_data_start[ns])
        goal_diversity = compute_solution_diversity(ik_data_goal[ns])
        
        stats['start_ik'][ns] = {
            'count': len(ik_data_start[ns]),
            'diversity': start_diversity,
            'statistics': {k: float(v) if isinstance(v, (int, np.integer, np.floating)) else v 
                          for k, v in start_stats.items()}
        }
        
        stats['goal_ik'][ns] = {
            'count': len(ik_data_goal[ns]),
            'diversity': goal_diversity,
            'statistics': {k: float(v) if isinstance(v, (int, np.integer, np.floating)) else v 
                          for k, v in goal_stats.items()}
        }
        
        print(f"  num_seeds={ns}: start={len(ik_data_start[ns])}, goal={len(ik_data_goal[ns])}")
    
    # Compute Wasserstein distances to reference
    print(f"\n6. Computing Wasserstein distances to reference (seeds={reference_seeds})...")
    if reference_seeds in ik_data_start:
        ref_start = ik_data_start[reference_seeds]
        ref_goal = ik_data_goal[reference_seeds]
        
        for ns in num_seeds_values:
            if ns != reference_seeds:
                ws_start = compute_wasserstein_to_reference(ik_data_start[ns], ref_start)
                ws_goal = compute_wasserstein_to_reference(ik_data_goal[ns], ref_goal)
                
                stats['start_ik'][ns]['wasserstein_to_ref'] = ws_start
                stats['goal_ik'][ns]['wasserstein_to_ref'] = ws_goal
                
                print(f"  num_seeds={ns}: WD_start={ws_start:.6f}, WD_goal={ws_goal:.6f}")
    
    # Generate visualizations
    print("\n7. Generating visualizations...")
    
    # Multi-seed analysis figure
    visualize_ik_distribution_multi_seeds(
        ik_data_start, scene_name, grasp_id, 'start', output_dir
    )
    visualize_ik_distribution_multi_seeds(
        ik_data_goal, scene_name, grasp_id, 'goal', output_dir
    )
    
    # Coverage comparison figure
    visualize_ik_coverage_comparison(
        ik_data_start, scene_name, grasp_id, 'start', output_dir
    )
    visualize_ik_coverage_comparison(
        ik_data_goal, scene_name, grasp_id, 'goal', output_dir
    )
    
    # Figure 1: Seeds with clustering comparison (raw vs downsampled)
    visualize_seeds_with_clustering_comparison(
        ik_data_start, scene_name, grasp_id, 'start', output_dir
    )
    visualize_seeds_with_clustering_comparison(
        ik_data_goal, scene_name, grasp_id, 'goal', output_dir
    )
    
    # Figure 2: Seed 20k clustering sweep (raw vs 30/50/200 clusters)
    if 20000 in ik_data_start:
        visualize_seed20k_clustering_sweep(
            ik_data_start[20000], scene_name, grasp_id, 'start', output_dir
        )
    if 20000 in ik_data_goal:
        visualize_seed20k_clustering_sweep(
            ik_data_goal[20000], scene_name, grasp_id, 'goal', output_dir
        )
    
    # Figure 3: Coverage curve - num_seeds sweep (1k to 80k with 50 clusters)
    visualize_coverage_curve_numseeds(
        ik_data_start, scene_name, grasp_id, 'start', output_dir
    )
    visualize_coverage_curve_numseeds(
        ik_data_goal, scene_name, grasp_id, 'goal', output_dir
    )
    
    # Figure 4: Coverage curve - clustering sweep (10 to 1000 clusters with 20k seeds)
    if 20000 in ik_data_start:
        visualize_coverage_curve_clustering(
            ik_data_start[20000], scene_name, grasp_id, 'start', output_dir
        )
    if 20000 in ik_data_goal:
        visualize_coverage_curve_clustering(
            ik_data_goal[20000], scene_name, grasp_id, 'goal', output_dir
        )
    
    # Save statistics
    stats_path = output_dir / f"{scene_name}_grasp_{grasp_id:04d}_numseeds_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Saved statistics to {stats_path.name}")
    
    return stats


def visualize_seeds_with_clustering_comparison(ik_data: Dict[int, np.ndarray],
                                               scene_name: str,
                                               grasp_id: int,
                                               ik_type: str,
                                               output_dir: Path):
    """
    Figure 1: Compare raw IK distributions across seeds with ik_clustering.
    
    Layout: 2×3
    - Top (Row 1): Raw data t-SNE with density for 5k, 20k, 80k raw data
    - Bottom (Row 2): Clustered to 50 solutions + density for same seeds (using ik_clustering)
    - Includes Wasserstein Distance (WD) metrics comparing clustered vs 80k raw data
    """
    from cuakr.utils.math import ik_clustering as ik_clustering_func
    
    num_seeds_to_show = [5000, 20000, 80000]
    n_clusters = 50  # Cluster to 50 solutions
    
    print(f"  Generating seeds+clustering comparison for {ik_type} IKs...")
    
    # Apply ik_clustering to each seed value first
    clustered_data = {}
    for ns in num_seeds_to_show:
        clustered = ik_clustering_func(
            torch.from_numpy(ik_data[ns]).float(),
            kmeans_clusters=n_clusters,
            ap_fallback_clusters=n_clusters
        ).cpu().numpy()
        clustered_data[ns] = clustered
        print(f"    Clustered {ns} seeds: {len(ik_data[ns])} → {len(clustered)} solutions")
    
    # Compute coverage score: how well each clustered set covers its original distribution
    # Coverage score = average distance from original points to nearest clustered point
    # Lower score = better coverage (clustered points are close to original distribution)
    coverage_metrics = {}
    for ns in num_seeds_to_show:
        coverage = compute_coverage_score(clustered_data[ns], ik_data[ns])
        coverage_metrics[ns] = coverage
        print(f"    Coverage(clustered_{ns}) = {coverage:.6f} (lower = better coverage of original)")
    
    # Expected trend: coverage(5k) < coverage(20k) < coverage(80k)
    # Because larger original sets have more points to cover
    
    # Compute t-SNE on COMBINED raw + clustered data for consistent embedding space
    # This ensures clustered points are positioned correctly relative to raw distribution
    all_data_combined = []
    raw_indices = []  # Track which indices are raw
    
    for ns in num_seeds_to_show:
        # Add raw data
        all_data_combined.append(ik_data[ns])
        raw_indices.extend([True] * len(ik_data[ns]))
        # Add clustered data
        all_data_combined.append(clustered_data[ns])
        raw_indices.extend([False] * len(clustered_data[ns]))
    
    # Combine and compute t-SNE
    all_data_for_tsne = np.vstack(all_data_combined)
    print(f"    Computing t-SNE on combined {len(all_data_for_tsne)} points (raw + clustered)...")
    tsne_all = compute_tsne(all_data_for_tsne, perplexity=min(30, len(all_data_for_tsne)//3))
    
    # Split t-SNE results back into raw and clustered per seed
    tsne_data_raw = {}
    tsne_data_clustered = {}
    idx = 0
    
    for ns in num_seeds_to_show:
        n_raw = len(ik_data[ns])
        n_clust = len(clustered_data[ns])
        
        # Raw data t-SNE points
        tsne_data_raw[ns] = tsne_all[idx:idx+n_raw]
        idx += n_raw
        
        # Clustered data t-SNE points
        tsne_data_clustered[ns] = tsne_all[idx:idx+n_clust]
        idx += n_clust
    
    print(f"    t-SNE computation complete. Data properly aligned in embedding space.")
    
    # Create figure with GridSpec for 2×3 layout
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # ========== ROW 1: Raw data with density ==========
    for i, ns in enumerate(num_seeds_to_show):
        ax = fig.add_subplot(gs[0, i])
        
        # Plot density contours
        if len(ik_data[ns]) > 10:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(tsne_data_raw[ns].T)
            x_min, x_max = tsne_data_raw[ns][:, 0].min(), tsne_data_raw[ns][:, 0].max()
            y_min, y_max = tsne_data_raw[ns][:, 1].min(), tsne_data_raw[ns][:, 1].max()
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            z = np.reshape(kde(positions).T, xx.shape)
            
            ax.contourf(xx, yy, z, levels=8, cmap='Blues', alpha=0.6)
            ax.contour(xx, yy, z, levels=5, colors='navy', alpha=0.4, linewidths=1)
        
        # Plot all points
        color = IK_CUROBO_VIZ['colors'].get(ns, '#999999')
        size = IK_CUROBO_VIZ['marker_sizes'].get(ns, 50)
        alpha = IK_CUROBO_VIZ['alphas'].get(ns, 0.6)
        
        ax.scatter(tsne_data_raw[ns][:, 0], tsne_data_raw[ns][:, 1],
                  c=color, s=size*0.3, alpha=alpha, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('t-SNE 1', fontsize=10)
        ax.set_ylabel('t-SNE 2', fontsize=10)
        ax.set_title(f'Raw: {ns//1000}k seeds (N={len(ik_data[ns])})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
    
    # ========== ROW 2: Clustered to 50 + density + WD metric ==========
    for i, ns in enumerate(num_seeds_to_show):
        ax = fig.add_subplot(gs[1, i])
        
        # Plot density for full data (raw)
        if len(ik_data[ns]) > 10:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(tsne_data_raw[ns].T)
            x_min, x_max = tsne_data_raw[ns][:, 0].min(), tsne_data_raw[ns][:, 0].max()
            y_min, y_max = tsne_data_raw[ns][:, 1].min(), tsne_data_raw[ns][:, 1].max()
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            z = np.reshape(kde(positions).T, xx.shape)
            
            ax.contourf(xx, yy, z, levels=8, cmap='Greens', alpha=0.5)
            ax.contour(xx, yy, z, levels=5, colors='darkgreen', alpha=0.4, linewidths=1)
        
        # Plot clustered points (larger, with diamond marker)
        color = IK_CUROBO_VIZ['colors'].get(ns, '#999999')
        ax.scatter(tsne_data_clustered[ns][:, 0], tsne_data_clustered[ns][:, 1],
                  c=color, s=150, alpha=0.8, edgecolors='black', linewidth=1.5, marker='D')
        
        # Add Coverage Score metric in large font
        # Coverage = avg distance from original points to nearest clustered point
        # Lower is better (clustered points closer to original distribution)
        coverage_value = coverage_metrics[ns]
        ax.text(0.5, -0.15, f'Coverage = {coverage_value:.4f}', 
               transform=ax.transAxes, fontsize=16, fontweight='bold',
               ha='center', va='top', 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, edgecolor='black', linewidth=2))
        
        ax.set_xlabel('t-SNE 1', fontsize=10)
        ax.set_ylabel('t-SNE 2', fontsize=10)
        ax.set_title(f'Clustered to {n_clusters}: {ns//1000}k seeds', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
    
    plt.suptitle(f'num_seeds Impact with Clustering (ik_clustering): {ik_type.upper()} IK\n{scene_name}, Grasp {grasp_id:04d}\nCoverage = avg distance from original points to nearest cluster (lower = better)',
                fontsize=14, fontweight='bold', y=0.995)
    
    save_figure(fig, output_dir / f"{scene_name}_grasp_{grasp_id:04d}_{ik_type}_seeds_clustering_compare")
    plt.close()
    
    print(f"    ✓ Saved seeds+clustering comparison figure")


def visualize_seed20k_clustering_sweep(ik_data_20k: np.ndarray,
                                      scene_name: str,
                                      grasp_id: int,
                                      ik_type: str,
                                      output_dir: Path):
    """
    Figure 2: For num_seeds=20000, show raw vs clustering to different levels (20, 50, 200).
    
    Layout: 1×4 showing:
    - Column 1: Raw data (20000 seeds, ~7000 solutions) - baseline heatmap
    - Columns 2-4: Clustered to 20/50/200 using same raw heatmap background
    - Includes coverage metric showing how well clustered data represents original distribution
    """
    from cuakr.utils.math import ik_clustering as ik_clustering_func
    
    print(f"  Generating seed 20k clustering sweep for {ik_type} IKs...")
    
    cluster_values = [20, 50, 200]
    
    # Prepare all data for t-SNE
    all_ik_data = [ik_data_20k]
    
    # Apply clustering to different levels
    clustered_data_dict = {}
    for c_val in cluster_values:
        clustered = ik_clustering_func(
            torch.from_numpy(ik_data_20k).float(),
            kmeans_clusters=c_val,
            ap_fallback_clusters=c_val
        ).cpu().numpy()
        all_ik_data.append(clustered)
        clustered_data_dict[c_val] = clustered
    
    # Compute t-SNE for all combined
    all_data_combined = np.vstack(all_ik_data)
    tsne_all = compute_tsne(all_data_combined, perplexity=min(30, len(all_data_combined)//3))
    
    # Split back
    idx = 0
    tsne_split = []
    counts = []
    for data in all_ik_data:
        n = len(data)
        tsne_split.append(tsne_all[idx:idx+n])
        counts.append(n)
        idx += n
    
    # Extract t-SNE for raw data (used for all heatmaps)
    tsne_raw = tsne_split[0]
    
    # Compute coverage metrics: for each clustered set, how well does it cover the 20k raw distribution?
    # Coverage = avg distance from 20k raw points to nearest clustered point
    # Lower coverage = better representation of original distribution
    coverage_metrics = {}
    for c_val in cluster_values:
        coverage = compute_coverage_score(clustered_data_dict[c_val], ik_data_20k)
        coverage_metrics[c_val] = coverage
        print(f"    Clustered to {c_val:3d}: Coverage = {coverage:.6f} (lower = better)")
    
    # Expected trend: coverage(20) > coverage(50) > coverage(200)
    # Because more clusters can get closer to every original point
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    labels = ['Raw (20k seeds)', f'Clustered to {cluster_values[0]}', 
              f'Clustered to {cluster_values[1]}', f'Clustered to {cluster_values[2]}']
    colors_plot = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    
    # Compute baseline heatmap from raw data (used for all plots)
    from scipy.stats import gaussian_kde
    kde_raw = gaussian_kde(tsne_raw.T)
    x_min, x_max = tsne_raw[:, 0].min(), tsne_raw[:, 0].max()
    y_min, y_max = tsne_raw[:, 1].min(), tsne_raw[:, 1].max()
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    z_raw = np.reshape(kde_raw(positions).T, xx.shape)
    
    for col, (ax, label, color) in enumerate(zip(axes, labels, colors_plot)):
        
        # All subplots use the same raw data heatmap background
        ax.contourf(xx, yy, z_raw, levels=10, cmap='RdYlBu_r', alpha=0.6)
        ax.contour(xx, yy, z_raw, levels=6, colors='black', alpha=0.3, linewidths=0.8)
        
        # Plot points (different for each column)
        if col == 0:
            # Raw points smaller
            ax.scatter(tsne_raw[:, 0], tsne_raw[:, 1],
                      c=color, s=30, alpha=0.5, edgecolors='none')
            count = counts[0]
            metric_text = f'N = {count}'
        else:
            # Clustered points as diamonds, larger
            c_idx = col - 1
            c_val = cluster_values[c_idx]
            tsne_clust = tsne_split[col]
            
            ax.scatter(tsne_clust[:, 0], tsne_clust[:, 1],
                      c=color, s=150, alpha=0.8, edgecolors='black', linewidth=1.5, marker='D')
            count = counts[col]
            coverage_cov = coverage_metrics[c_val]
            metric_text = f'N = {count}\nCov = {coverage_cov:.4f}'
        
        ax.set_xlabel('t-SNE 1', fontsize=11)
        ax.set_ylabel('t-SNE 2', fontsize=11)
        
        # Add count and coverage metric to title
        ax.set_title(f'{label}\n{metric_text}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'num_seeds=20k Clustering Sweep: {ik_type.upper()} IK\n{scene_name}, Grasp {grasp_id:04d}\nAll heatmaps show raw data distribution | Cov = Coverage (lower = better)',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    save_figure(fig, output_dir / f"{scene_name}_grasp_{grasp_id:04d}_{ik_type}_clustering_sweep_20k")
    plt.close()
    
    print(f"    ✓ Saved seed 20k clustering sweep figure")


def visualize_coverage_curve_numseeds(ik_data: Dict[int, np.ndarray], scene_name: str, 
                                       grasp_id: int, ik_type: str, output_dir: Path):
    """
    Figure 3: Coverage curve showing how coverage changes with num_seeds.
    
    Plots coverage score (average distance to nearest cluster) vs num_seeds
    from available num_seeds, with 50 clusters fixed.
    
    Coverage = average distance from original points to nearest cluster point.
    Lower coverage = better (clusters closer to original distribution)
    """
    from cuakr.utils.math import ik_clustering as ik_clustering_func
    
    print(f"  Generating coverage curve (num_seeds) for {ik_type} IKs...")
    
    # Check if cached
    cached = load_coverage_curve(scene_name, grasp_id, ik_type, 'numseeds')
    if cached is not None:
        coverage_values, num_seeds_values = cached
        print(f"    ✓ Loaded from cache")
    else:
        # Generate continuous sweep from 5k to 80k with 5k steps
        num_seeds_values_list = list(np.arange(5000, 85000, 5000))  # 5k, 10k, 15k, ..., 80k
        n_clusters = 50  # Fixed cluster count
        coverage_values = []
        valid_num_seeds = []
        
        # Get largest dataset
        max_seed_count = max(ik_data.keys())
        
        print(f"    Computing coverage for {len(num_seeds_values_list)} num_seeds values (5k to 80k)...")
        print(f"    Available num_seeds in data: {sorted(ik_data.keys())}")
        
        for ns in tqdm(num_seeds_values_list, desc="num_seeds sweep"):
            if ns > max_seed_count:
                # Skip values beyond available data
                continue
                
            if ns in ik_data:
                # Use actual data if available
                data_to_cluster = ik_data[ns]
            else:
                # Subsample from the 80k raw data to approximate intermediate num_seeds
                # This creates a pseudo-dataset by randomly selecting from 80k
                full_data = ik_data[max_seed_count]
                np.random.seed(42)  # For reproducibility
                
                # Can only subsample if requested size <= available size
                if ns <= len(full_data):
                    indices = np.random.choice(len(full_data), size=ns, replace=False)
                    data_to_cluster = full_data[indices]
                else:
                    # Skip this value - can't create larger sample from smaller population
                    continue
            
            # Cluster to 50
            clustered = ik_clustering_func(
                torch.from_numpy(data_to_cluster).float(),
                kmeans_clusters=n_clusters,
                ap_fallback_clusters=n_clusters
            ).cpu().numpy()
            
            # Compute coverage
            coverage = compute_coverage_score(clustered, data_to_cluster)
            coverage_values.append(coverage)
            valid_num_seeds.append(ns)
        
        coverage_values = np.array(coverage_values)
        num_seeds_values = np.array(valid_num_seeds)
        
        print(f"    Generated {len(valid_num_seeds)} coverage curve points: {valid_num_seeds}")
        
        # Save to cache
        save_coverage_curve(scene_name, grasp_id, ik_type, 'numseeds', 
                          coverage_values, num_seeds_values)
    
    # Create line plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(num_seeds_values / 1000, coverage_values, 'o-', linewidth=2.5, 
           markersize=8, color='#2E86AB', markeredgecolor='black', markeredgewidth=1.5)
    
    ax.set_xlabel('num_seeds (×1000)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Coverage Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Coverage vs num_seeds (50 clusters fixed)\n{scene_name}, Grasp {grasp_id:04d}, {ik_type.upper()} IK',
                fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add annotations
    ax.text(0.02, 0.98, f'Lower = Better Coverage', transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    save_figure(fig, output_dir / f"{scene_name}_grasp_{grasp_id:04d}_{ik_type}_seeds_clustering_compare_curve")
    plt.close()
    
    print(f"    ✓ Saved num_seeds coverage curve figure")


def visualize_coverage_curve_clustering(ik_data_20k: np.ndarray, scene_name: str, 
                                        grasp_id: int, ik_type: str, output_dir: Path):
    """
    Figure 4: Coverage curve showing how coverage changes with num_clusters.
    
    Plots coverage score (average distance to nearest cluster) vs num_clusters
    from 10 to 1000 clusters, with 20k num_seeds fixed.
    
    Coverage = average distance from original points to nearest cluster point.
    Lower coverage = better (clusters closer to original distribution)
    """
    from cuakr.utils.math import ik_clustering as ik_clustering_func
    
    print(f"  Generating coverage curve (clustering sweep) for {ik_type} IKs...")
    
    # Check if cached
    cached = load_coverage_curve(scene_name, grasp_id, ik_type, 'clustering')
    if cached is not None:
        coverage_values, cluster_values = cached
        print(f"    ✓ Loaded from cache")
    else:
        # Generate coverage values for different num_clusters
        cluster_values_to_sweep = list(range(10, 1001, 10))  # 10 to 1000 by 10
        coverage_values = []
        
        print(f"    Computing coverage for {len(cluster_values_to_sweep)} cluster values...")
        for n_clust in tqdm(cluster_values_to_sweep, desc="clustering sweep"):
            # Cluster to n_clust
            clustered = ik_clustering_func(
                torch.from_numpy(ik_data_20k).float(),
                kmeans_clusters=n_clust,
                ap_fallback_clusters=n_clust
            ).cpu().numpy()
            
            # Compute coverage
            coverage = compute_coverage_score(clustered, ik_data_20k)
            coverage_values.append(coverage)
        
        coverage_values = np.array(coverage_values)
        cluster_values = np.array(cluster_values_to_sweep)
        
        # Save to cache
        save_coverage_curve(scene_name, grasp_id, ik_type, 'clustering', 
                          coverage_values, cluster_values)
    
    # Create line plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(cluster_values, coverage_values, 'o-', linewidth=2.5, 
           markersize=6, color='#A23B72', markeredgecolor='black', markeredgewidth=1.5)
    
    ax.set_xlabel('Number of Clusters', fontsize=14, fontweight='bold')
    ax.set_ylabel('Coverage Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Coverage vs Number of Clusters (20k seeds fixed)\n{scene_name}, Grasp {grasp_id:04d}, {ik_type.upper()} IK',
                fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add annotations
    ax.text(0.02, 0.98, f'Lower = Better Coverage', transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    save_figure(fig, output_dir / f"{scene_name}_grasp_{grasp_id:04d}_{ik_type}_clustering_sweep_20k_curve")
    plt.close()
    
    print(f"    ✓ Saved clustering coverage curve figure")


def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("IK cuRobo num_seeds Impact Analysis")
    print("=" * 80)
    
    # Setup
    setup_plotting_style(VIZ_CONFIG)
    create_output_dirs()
    
    output_dir = FIGURE_ROOT / "ik_curobo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    analysis_scenes = IK_CUROBO['analysis_scenes']
    analysis_grasps = IK_CUROBO['analysis_grasp_ids']
    num_seeds_values = IK_CUROBO['num_seeds_values']
    
    print(f"\nAnalysis Configuration:")
    print(f"  Scenes: {len(analysis_scenes)} scenes")
    print(f"  Grasp IDs: {analysis_grasps}")
    print(f"  num_seeds values: {num_seeds_values}")
    print(f"  Reference num_seeds: {IK_CUROBO['reference_num_seeds']}")
    print(f"  Cache enabled: {IK_CUROBO['enable_cache']}")
    print(f"  Cache dir: {IK_CUROBO['cache_dir']}")
    
    # Create cache directory
    IK_CUROBO['cache_dir'].mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for scene_idx, scene_name in enumerate(analysis_scenes, 1):
        print(f"\n\n{'#'*80}")
        print(f"Scene {scene_idx}/{len(analysis_scenes)}: {scene_name}")
        print(f"{'#'*80}")
        
        scene_path = get_scene_path(scene_name)
        
        for grasp_id in analysis_grasps:
            try:
                result = analyze_single_grasp(scene_name, str(scene_path), grasp_id, output_dir)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"\n✗ Error analyzing {scene_name}, grasp {grasp_id}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save summary
    summary = {
        'analyzed_scenes': analysis_scenes,
        'analyzed_grasp_ids': analysis_grasps,
        'num_seeds_values': num_seeds_values,
        'reference_num_seeds': IK_CUROBO['reference_num_seeds'],
        'total_grasps_analyzed': len(all_results),
        'results': all_results,
    }
    
    summary_path = OUTPUT_ROOT / "ik_curobo_numseeds_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("✓ IK cuRobo num_seeds Analysis Complete")
    print(f"✓ Analyzed {len(all_results)} grasps")
    print(f"✓ Output directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
