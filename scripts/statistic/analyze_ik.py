"""
Analyze IK distribution comparison: raw_ik, success_ik, and filtered_ik.

This script compares three types of IK solutions:
1. Raw IK: Clustered IK solutions from ik_data.pt (fewest points)
2. Success IK: IK from successful trajectories in traj_data.pt (most points)
3. Filtered IK: IK from filtered_traj_data.pt (post-processed successful trajectories)

Author: Statistical Analysis Suite for AutoMoMA
Date: 2025-10-16
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import json

from config import *
from data_loader import (load_all_ik_data, load_all_trajectory_data, 
                         extract_ik_from_trajectories, filter_successful_trajectories)
from utils import (setup_plotting_style, compute_tsne, compute_umap, 
                  compute_joint_statistics, plot_2d_embedding, 
                  plot_joint_distributions, plot_comparison_boxplot, save_figure)


def analyze_ik_coverage(scene_name: str, grasp_id: int, 
                       ik_data_dict: Dict, traj_raw_dict: Dict, traj_filtered_dict: Dict):
    """
    Analyze IK distribution comparison: raw_ik, success_ik, and filtered_ik.
    
    - raw_ik: Clustered IK from ik_data.pt (fewest, plotted on upper layer)
    - success_ik: IK from traj_data.pt where success=True (most)
    - filtered_ik: IK from filtered_traj_data.pt where success=True
    """
    if scene_name not in ik_data_dict or grasp_id not in ik_data_dict[scene_name]:
        print(f"No IK data for {scene_name}, grasp {grasp_id}")
        return
    
    if scene_name not in traj_raw_dict or grasp_id not in traj_raw_dict[scene_name]:
        print(f"No raw trajectory data for {scene_name}, grasp {grasp_id}")
        return
    
    if scene_name not in traj_filtered_dict or grasp_id not in traj_filtered_dict[scene_name]:
        print(f"No filtered trajectory data for {scene_name}, grasp {grasp_id}")
        return
    
    ik_data = ik_data_dict[scene_name][grasp_id]
    traj_raw = traj_raw_dict[scene_name][grasp_id]
    traj_filtered = traj_filtered_dict[scene_name][grasp_id]
    
    # Extract raw_ik (clustered IK from ik_data.pt)
    raw_start_iks = ik_data.start_iks.cpu().numpy()
    raw_goal_iks = ik_data.goal_iks.cpu().numpy()
    
    # Extract success_ik (successful trajectories from traj_data.pt)
    successful_traj = filter_successful_trajectories(traj_raw)
    success_start_iks, success_goal_iks = extract_ik_from_trajectories(successful_traj)
    
    # Extract filtered_ik (from filtered_traj_data.pt, all are successful)
    filtered_start_iks, filtered_goal_iks = extract_ik_from_trajectories(traj_filtered)
    
    print(f"\n{'='*80}")
    print(f"IK Distribution Comparison: {scene_name}, Grasp {grasp_id:04d}")
    print(f"{'='*80}")
    print(f"Raw IK (Clustered from ik_data.pt):")
    print(f"  Start IKs: {len(raw_start_iks)}")
    print(f"  Goal IKs: {len(raw_goal_iks)}")
    print(f"\nSuccess IK (CuRobo successful from traj_data.pt):")
    print(f"  Start IKs: {len(success_start_iks)}")
    print(f"  Goal IKs: {len(success_goal_iks)}")
    print(f"\nFiltered IK (from filtered_traj_data.pt):")
    print(f"  Start IKs: {len(filtered_start_iks)}")
    print(f"  Goal IKs: {len(filtered_goal_iks)}")
    
    # Visualize with t-SNE
    output_dir = FIGURE_ROOT / "ik_comparison" / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine all start IKs
    # Plot order: success_ik first (most points), then filtered_ik, then raw_ik (on top)
    all_start = np.vstack([success_start_iks, filtered_start_iks, raw_start_iks])
    labels_start = (['Success IK'] * len(success_start_iks) + 
                   ['Filtered IK'] * len(filtered_start_iks) + 
                   ['Raw IK'] * len(raw_start_iks))
    
    print("\nComputing t-SNE for start IKs...")
    tsne_start = compute_tsne(all_start, perplexity=min(30, len(all_start)//3))
    
    fig, ax = plot_2d_embedding(
        tsne_start, 
        labels=np.array(labels_start),
        colors=[VIZ_CONFIG['color_success'], VIZ_CONFIG['color_filtered'], VIZ_CONFIG['color_raw']],
        title=f"Start IK Distribution - {scene_name}, Grasp {grasp_id:04d}",
        alpha=VIZ_CONFIG['alpha_scatter']
    )
    save_figure(fig, output_dir / f"start_ik_distribution_grasp_{grasp_id:04d}")
    plt.close()
    
    # Combine all goal IKs
    # Plot order: success_ik first (most points), then filtered_ik, then raw_ik (on top)
    all_goal = np.vstack([success_goal_iks, filtered_goal_iks, raw_goal_iks])
    labels_goal = (['Success IK'] * len(success_goal_iks) + 
                  ['Filtered IK'] * len(filtered_goal_iks) + 
                  ['Raw IK'] * len(raw_goal_iks))
    
    print("Computing t-SNE for goal IKs...")
    tsne_goal = compute_tsne(all_goal, perplexity=min(30, len(all_goal)//3))
    
    fig, ax = plot_2d_embedding(
        tsne_goal,
        labels=np.array(labels_goal),
        colors=[VIZ_CONFIG['color_success'], VIZ_CONFIG['color_filtered'], VIZ_CONFIG['color_raw']],
        title=f"Goal IK Distribution - {scene_name}, Grasp {grasp_id:04d}",
        alpha=VIZ_CONFIG['alpha_scatter']
    )
    save_figure(fig, output_dir / f"goal_ik_distribution_grasp_{grasp_id:04d}")
    plt.close()


def analyze_ik_coverage_aggregated(scene_name: str, grasp_ids: list,
                                   ik_data_dict: Dict, traj_raw_dict: Dict, traj_filtered_dict: Dict):
    """
    Analyze IK distribution for all grasps in a scene combined.
    
    Creates visualizations showing:
    - How filtered_ik covers success_ik and raw_ik
    - Overlap regions between the three distributions
    """
    print(f"\n{'='*80}")
    print(f"Aggregated IK Distribution Analysis: {scene_name}")
    print(f"Analyzing {len(grasp_ids)} grasps")
    print(f"{'='*80}")
    
    # Collect all IK data across grasps
    all_raw_start = []
    all_raw_goal = []
    all_success_start = []
    all_success_goal = []
    all_filtered_start = []
    all_filtered_goal = []
    
    valid_grasps = 0
    
    for grasp_id in grasp_ids:
        if scene_name not in ik_data_dict or grasp_id not in ik_data_dict[scene_name]:
            continue
        if scene_name not in traj_raw_dict or grasp_id not in traj_raw_dict[scene_name]:
            continue
        if scene_name not in traj_filtered_dict or grasp_id not in traj_filtered_dict[scene_name]:
            continue
        
        valid_grasps += 1
        
        # Raw IK (clustered)
        ik_data = ik_data_dict[scene_name][grasp_id]
        all_raw_start.append(ik_data.start_iks.cpu().numpy())
        all_raw_goal.append(ik_data.goal_iks.cpu().numpy())
        
        # Success IK (from raw trajectories)
        traj_raw = traj_raw_dict[scene_name][grasp_id]
        successful_traj = filter_successful_trajectories(traj_raw)
        success_start, success_goal = extract_ik_from_trajectories(successful_traj)
        all_success_start.append(success_start)
        all_success_goal.append(success_goal)
        
        # Filtered IK
        traj_filtered = traj_filtered_dict[scene_name][grasp_id]
        filtered_start, filtered_goal = extract_ik_from_trajectories(traj_filtered)
        all_filtered_start.append(filtered_start)
        all_filtered_goal.append(filtered_goal)
    
    if valid_grasps == 0:
        print(f"⚠️  No valid grasps found for {scene_name}")
        return
    
    # Concatenate all data
    raw_start_iks = np.vstack(all_raw_start)
    raw_goal_iks = np.vstack(all_raw_goal)
    success_start_iks = np.vstack(all_success_start)
    success_goal_iks = np.vstack(all_success_goal)
    filtered_start_iks = np.vstack(all_filtered_start)
    filtered_goal_iks = np.vstack(all_filtered_goal)
    
    print(f"\nAggregated Statistics ({valid_grasps} grasps):")
    print(f"Raw IK (Clustered):")
    print(f"  Start IKs: {len(raw_start_iks)}")
    print(f"  Goal IKs: {len(raw_goal_iks)}")
    print(f"Success IK (CuRobo successful):")
    print(f"  Start IKs: {len(success_start_iks)}")
    print(f"  Goal IKs: {len(success_goal_iks)}")
    print(f"Filtered IK:")
    print(f"  Start IKs: {len(filtered_start_iks)}")
    print(f"  Goal IKs: {len(filtered_goal_iks)}")
    
    # DEBUG: Print statistics to investigate start IK mismatch
    print(f"\nDEBUG - Start IK Statistics:")
    print(f"Raw start IKs shape: {raw_start_iks.shape}, mean: {np.mean(raw_start_iks, axis=0)}")
    print(f"Success start IKs shape: {success_start_iks.shape}, mean: {np.mean(success_start_iks, axis=0)}")
    print(f"Filtered start IKs shape: {filtered_start_iks.shape}, mean: {np.mean(filtered_start_iks, axis=0)}")
    
    # DEBUG: Print statistics for goal IKs
    print(f"\nDEBUG - Goal IK Statistics:")
    print(f"Raw goal IKs shape: {raw_goal_iks.shape}, mean: {np.mean(raw_goal_iks, axis=0)}")
    print(f"Success goal IKs shape: {success_goal_iks.shape}, mean: {np.mean(success_goal_iks, axis=0)}")
    print(f"Filtered goal IKs shape: {filtered_goal_iks.shape}, mean: {np.mean(filtered_goal_iks, axis=0)}")
    
    output_dir = FIGURE_ROOT / "ik_comparison" / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize START IKs with better overlap visualization
    print("\nGenerating start IK visualizations...")
    _visualize_ik_overlap(
        raw_start_iks, success_start_iks, filtered_start_iks,
        scene_name, "start", output_dir
    )
    
    # Visualize GOAL IKs
    print("Generating goal IK visualizations...")
    _visualize_ik_overlap(
        raw_goal_iks, success_goal_iks, filtered_goal_iks,
        scene_name, "goal", output_dir
    )


def _visualize_ik_overlap(raw_iks, success_iks, filtered_iks, scene_name, ik_type, output_dir):
    """
    Create visualization showing how success and filtered IK cover raw IK distribution.
    
    Key relationships:
    - Raw IK: Smallest, most selective (clustered from ik_data.pt, ~50 solutions)
    - Success IK: Largest, from successful trajectories (~5000 solutions)
    - Filtered IK: Medium, post-processed subset (~2000 solutions)
    
    Strategy:
    1. Left: Show raw as points (evenly distributed), success/filtered as density
    2. Right: Layered scatter showing all three distributions
    """
    # DEBUG: Check for NaN or inf values
    print(f"\n  DEBUG ({ik_type} IK overlap visualization):")
    print(f"    Raw IKs: min={np.nanmin(raw_iks):.3f}, max={np.nanmax(raw_iks):.3f}, has_nan={np.any(np.isnan(raw_iks))}")
    print(f"    Success IKs: min={np.nanmin(success_iks):.3f}, max={np.nanmax(success_iks):.3f}, has_nan={np.any(np.isnan(success_iks))}")
    print(f"    Filtered IKs: min={np.nanmin(filtered_iks):.3f}, max={np.nanmax(filtered_iks):.3f}, has_nan={np.any(np.isnan(filtered_iks))}")
    
    # Combine all data for t-SNE
    all_data = np.vstack([raw_iks, success_iks, filtered_iks])
    n_raw = len(raw_iks)
    n_success = len(success_iks)
    n_filtered = len(filtered_iks)
    
    print(f"  Data sizes: Raw={n_raw}, Success={n_success}, Filtered={n_filtered}")
    print(f"  Computing t-SNE for {ik_type} IKs ({len(all_data)} points)...")
    tsne_result = compute_tsne(all_data, perplexity=min(30, len(all_data)//3))
    
    # Split back into groups
    raw_tsne = tsne_result[:n_raw]
    success_tsne = tsne_result[n_raw:n_raw+n_success]
    filtered_tsne = tsne_result[n_raw+n_success:]
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Density plot - Show raw as points, success/filtered as density
    ax = axes[0]
    
    from scipy.stats import gaussian_kde
    
    # Get bounds from all data
    x_min, x_max = tsne_result[:, 0].min(), tsne_result[:, 0].max()
    y_min, y_max = tsne_result[:, 1].min(), tsne_result[:, 1].max()
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Success IK density (base layer - broadest coverage)
    if len(success_tsne) > 10:
        kde_success = gaussian_kde(success_tsne.T)
        z_success = np.reshape(kde_success(positions).T, xx.shape)
        contour1 = ax.contourf(xx, yy, z_success, levels=VIZ_CONFIG['contour_levels_filled'], 
                              cmap='Blues', alpha=VIZ_CONFIG['alpha_density_fill'])
        ax.contour(xx, yy, z_success, levels=VIZ_CONFIG['contour_levels'], 
                  colors=VIZ_CONFIG['color_success'],
                  alpha=VIZ_CONFIG['alpha_density_line'], 
                  linewidths=VIZ_CONFIG['linewidth_medium'], linestyles='solid')
    
    # Filtered IK density (middle layer)
    if len(filtered_tsne) > 10:
        kde_filtered = gaussian_kde(filtered_tsne.T)
        z_filtered = np.reshape(kde_filtered(positions).T, xx.shape)
        ax.contour(xx, yy, z_filtered, levels=VIZ_CONFIG['contour_levels'], 
                  colors=VIZ_CONFIG['color_filtered'],
                  alpha=VIZ_CONFIG['alpha_density_line'], 
                  linewidths=VIZ_CONFIG['linewidth_medium'], linestyles='dashed')
    
    # Raw IK as prominent points (top layer - should be covered by densities)
    ax.scatter(raw_tsne[:, 0], raw_tsne[:, 1],
              c=VIZ_CONFIG['color_raw'], s=VIZ_CONFIG['marker_size_raw_ik'], 
              alpha=VIZ_CONFIG['alpha_raw_points'],
              edgecolors='black', linewidths=VIZ_CONFIG['linewidth_medium'], marker='D', 
              label=f'Raw IK ({n_raw})', zorder=10)
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title(f'{ik_type.capitalize()} IK: Coverage of Raw by Success/Filtered\n{scene_name}', fontsize=14)
    ax.legend(['Raw IK (clustered)', 'Success IK density', 'Filtered IK density'], 
             loc='best', fontsize=10)
    ax.grid(True, alpha=VIZ_CONFIG['alpha_grid'])
    
    # Add statistics text
    coverage_text = (f'Raw: {n_raw} (clustered)\n'
                    f'Success: {n_success} ({n_success/n_raw:.1f}x)\n'
                    f'Filtered: {n_filtered} ({n_filtered/n_raw:.1f}x)')
    ax.text(0.02, 0.98, coverage_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=VIZ_CONFIG['alpha_infobox']))
    
    # Right: Layered scatter showing all distributions
    ax = axes[1]
    
    # Plot in reverse order: largest first (bottom), smallest last (top)
    # Success (largest, bottom layer, very transparent)
    ax.scatter(success_tsne[:, 0], success_tsne[:, 1],
              c=VIZ_CONFIG['color_success'], alpha=VIZ_CONFIG['alpha_success_layer'], 
              s=VIZ_CONFIG['marker_size_small'], 
              label=f'Success IK ({n_success})')
    
    # Filtered (medium, middle layer, semi-transparent)
    ax.scatter(filtered_tsne[:, 0], filtered_tsne[:, 1],
              c=VIZ_CONFIG['color_filtered'], alpha=VIZ_CONFIG['alpha_filtered_layer'], 
              s=VIZ_CONFIG['marker_size_small'],
              label=f'Filtered IK ({n_filtered})')
    
    # Raw (smallest, top layer, very visible)
    ax.scatter(raw_tsne[:, 0], raw_tsne[:, 1],
              c=VIZ_CONFIG['color_raw'], alpha=VIZ_CONFIG['alpha_raw_points'], 
              s=VIZ_CONFIG['marker_size_large'],
              edgecolors='black', linewidths=VIZ_CONFIG['linewidth_medium'], marker='D',
              label=f'Raw IK ({n_raw})', zorder=10)
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title(f'{ik_type.capitalize()} IK: All Distributions\n{scene_name}', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=VIZ_CONFIG['alpha_grid'])
    
    plt.tight_layout()
    save_figure(fig, output_dir / f"{ik_type}_ik_coverage")
    plt.close()
    
    print(f"  ✓ Saved {ik_type} IK coverage visualization")


def analyze_object_angle_ik(ik_data_dict: Dict, traj_raw_dict: Dict, traj_filtered_dict: Dict):
    """
    Analyze object angle (11th dimension) across IK datasets.
    """
    print(f"\n{'='*80}")
    print("Object Angle Analysis (IK Data)")
    print(f"{'='*80}")
    
    # Collect object angles from different sources
    precomputed_start_angles = []
    precomputed_goal_angles = []
    traj_raw_start_angles = []
    traj_raw_goal_angles = []
    traj_filtered_start_angles = []
    traj_filtered_goal_angles = []
    
    obj_idx = TRAJ_ANALYSIS['object_joint_index']
    
    for scene_name in ik_data_dict.keys():
        for grasp_id, ik_data in ik_data_dict[scene_name].items():
            precomputed_start_angles.extend(ik_data.start_iks[:, obj_idx].cpu().numpy())
            precomputed_goal_angles.extend(ik_data.goal_iks[:, obj_idx].cpu().numpy())
        
        if scene_name in traj_raw_dict:
            for grasp_id, traj_data in traj_raw_dict[scene_name].items():
                start_iks, goal_iks = extract_ik_from_trajectories(traj_data)
                # Filter successful only
                successful = filter_successful_trajectories(traj_data)
                success_start, success_goal = extract_ik_from_trajectories(successful)
                
                traj_raw_start_angles.extend(success_start[:, obj_idx])
                traj_raw_goal_angles.extend(success_goal[:, obj_idx])
        
        if scene_name in traj_filtered_dict:
            for grasp_id, traj_data in traj_filtered_dict[scene_name].items():
                start_iks, goal_iks = extract_ik_from_trajectories(traj_data)
                traj_filtered_start_angles.extend(start_iks[:, obj_idx])
                traj_filtered_goal_angles.extend(goal_iks[:, obj_idx])
    
    # Convert to arrays
    precomputed_start_angles = np.array(precomputed_start_angles)
    precomputed_goal_angles = np.array(precomputed_goal_angles)
    traj_raw_start_angles = np.array(traj_raw_start_angles)
    traj_raw_goal_angles = np.array(traj_raw_goal_angles)
    traj_filtered_start_angles = np.array(traj_filtered_start_angles)
    traj_filtered_goal_angles = np.array(traj_filtered_goal_angles)
    
    print(f"\nObject Angle Statistics:")
    print(f"Precomputed IK:")
    print(f"  Start angles: mean={np.mean(precomputed_start_angles):.3f}, std={np.std(precomputed_start_angles):.3f}")
    print(f"  Goal angles: mean={np.mean(precomputed_goal_angles):.3f}, std={np.std(precomputed_goal_angles):.3f}")
    print(f"\nTrajectory (successful, raw):")
    print(f"  Start angles: mean={np.mean(traj_raw_start_angles):.3f}, std={np.std(traj_raw_start_angles):.3f}")
    print(f"  Goal angles: mean={np.mean(traj_raw_goal_angles):.3f}, std={np.std(traj_raw_goal_angles):.3f}")
    print(f"\nTrajectory (filtered):")
    print(f"  Start angles: mean={np.mean(traj_filtered_start_angles):.3f}, std={np.std(traj_filtered_start_angles):.3f}")
    print(f"  Goal angles: mean={np.mean(traj_filtered_goal_angles):.3f}, std={np.std(traj_filtered_goal_angles):.3f}")
    
    # Plot comparisons
    output_dir = FIGURE_ROOT / "ik_precomputed"
    
    # Start angles comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    data_dict = {
        'Precomputed': precomputed_start_angles,
        'Traj (Success)': traj_raw_start_angles,
        'Traj (Filtered)': traj_filtered_start_angles
    }
    
    positions = range(len(data_dict))
    colors = [VIZ_CONFIG['color_raw'], VIZ_CONFIG['color_success'], VIZ_CONFIG['color_filtered']]
    
    for i, (label, data) in enumerate(data_dict.items()):
        parts = ax.violinplot([data], positions=[i], widths=0.7, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(data_dict.keys())
    ax.set_ylabel('Object Angle (rad)')
    ax.set_title('Start Object Angle Distribution Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    save_figure(fig, output_dir / "object_angle_start_comparison")
    plt.close()
    
    # Goal angles comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    data_dict = {
        'Precomputed': precomputed_goal_angles,
        'Traj (Success)': traj_raw_goal_angles,
        'Traj (Filtered)': traj_filtered_goal_angles
    }
    
    for i, (label, data) in enumerate(data_dict.items()):
        parts = ax.violinplot([data], positions=[i], widths=0.7, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(data_dict.keys())
    ax.set_ylabel('Object Angle (rad)')
    ax.set_title('Goal Object Angle Distribution Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    save_figure(fig, output_dir / "object_angle_goal_comparison")
    plt.close()


def main():
    """Main analysis pipeline for IK distribution comparison."""
    print("=" * 80)
    print("IK Distribution Comparison Analysis")
    print("=" * 80)
    
    # Setup
    setup_plotting_style(VIZ_CONFIG)
    create_output_dirs()
    
    # Load data
    print("\n" + "="*80)
    print("Loading Data")
    print("="*80)
    
    ik_data = load_all_ik_data(ROBOT_NAME, SCENE_NAMES, OBJECT_NAMES, 
                               GRASP_IDS, DATA_ROOT, verbose=True)
    
    traj_raw_data = load_all_trajectory_data(ROBOT_NAME, SCENE_NAMES, OBJECT_NAMES,
                                            GRASP_IDS, DATA_ROOT, filtered=False, verbose=True)
    
    traj_filtered_data = load_all_trajectory_data(ROBOT_NAME, SCENE_NAMES, OBJECT_NAMES,
                                                  GRASP_IDS, DATA_ROOT, filtered=True, verbose=True)
    
    # Analysis 1: Aggregated IK coverage for all grasps in first scene
    print("\n" + "="*80)
    print("Analysis 1: Aggregated IK Coverage (All Scenes)")
    print("="*80)
    
    # Analyze all scenes (not just the first one)
    for scene_name in SCENE_NAMES:
        print(f"\nAnalyzing scene: {scene_name}")
        analyze_ik_coverage_aggregated(scene_name, GRASP_IDS, ik_data, traj_raw_data, traj_filtered_data)
    
    # Analysis 2: Object angle analysis
    print("\n" + "="*80)
    print("Analysis 2: Object Angle Distribution")
    print("="*80)
    
    analyze_object_angle_ik(ik_data, traj_raw_data, traj_filtered_data)
    
    # Save summary statistics
    summary = {
        'total_scenes': len(ik_data),
        'total_grasps': sum(len(grasps) for grasps in ik_data.values()),
        'ik_statistics': {},
    }
    
    for scene_name, scene_data in ik_data.items():
        for grasp_id, ik in scene_data.items():
            key = f"{scene_name}_grasp_{grasp_id:04d}"
            summary['ik_statistics'][key] = {
                'num_start_iks': ik.num_start_iks,
                'num_goal_iks': ik.num_goal_iks,
            }
    
    summary_path = OUTPUT_ROOT / "ik_analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("✓ Precomputed IK Analysis Complete")
    print("="*80)


if __name__ == "__main__":
    main()
