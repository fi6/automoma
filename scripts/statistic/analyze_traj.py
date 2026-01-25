"""
Comprehensive trajectory data analysis.

This script performs full statistical analysis of trajectory data including:
- Dataset scale statistics
- Success rate analysis  
- Diversity metrics
- Object angle analysis
- Quality metrics

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
from collections import defaultdict

from config import *
from data_loader import (load_all_trajectory_data, filter_successful_trajectories,
                         TrajectoryData)
from utils import (setup_plotting_style, compute_tsne, compute_joint_statistics,
                  compute_diversity_metrics, compute_trajectory_variance,
                  plot_2d_embedding, plot_trajectory_heatmap, 
                  plot_success_rate_comparison, save_figure)


def compute_dataset_statistics(traj_data_dict: Dict, data_type: str = "raw") -> Dict:
    """Compute overall dataset statistics."""
    stats = {
        'data_type': data_type,
        'num_scenes': len(traj_data_dict),
        'num_grasps_total': 0,
        'num_trajectories_total': 0,
        'num_successful_total': 0,
        'per_scene_stats': {},
        'per_grasp_stats': {},
    }
    
    for scene_name, scene_data in traj_data_dict.items():
        scene_traj_count = 0
        scene_success_count = 0
        
        for grasp_id, traj_data in scene_data.items():
            stats['num_grasps_total'] += 1
            stats['num_trajectories_total'] += traj_data.num_trajectories
            stats['num_successful_total'] += traj_data.num_successful
            
            scene_traj_count += traj_data.num_trajectories
            scene_success_count += traj_data.num_successful
            
            # Per-grasp stats
            key = f"{scene_name}_grasp_{grasp_id:04d}"
            stats['per_grasp_stats'][key] = {
                'num_trajectories': traj_data.num_trajectories,
                'num_successful': traj_data.num_successful,
                'success_rate': traj_data.success_rate,
            }
        
        # Per-scene stats
        stats['per_scene_stats'][scene_name] = {
            'num_grasps': len(scene_data),
            'num_trajectories': scene_traj_count,
            'num_successful': scene_success_count,
            'success_rate': scene_success_count / scene_traj_count if scene_traj_count > 0 else 0,
        }
    
    stats['overall_success_rate'] = (stats['num_successful_total'] / 
                                    stats['num_trajectories_total'] 
                                    if stats['num_trajectories_total'] > 0 else 0)
    
    return stats


def plot_dataset_scale_statistics(raw_stats: Dict, filtered_stats: Dict, output_dir: Path):
    """Plot dataset scale statistics."""
    print("\nGenerating dataset scale visualizations...")
    
    # Bar chart: total counts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    categories = ['Scenes', 'Grasps', 'Trajectories']
    raw_counts = [raw_stats['num_scenes'], raw_stats['num_grasps_total'], 
                  raw_stats['num_trajectories_total']]
    filtered_counts = [filtered_stats['num_scenes'], filtered_stats['num_grasps_total'],
                      filtered_stats['num_trajectories_total']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0].bar(x - width/2, raw_counts, width, label='Raw', 
               color=VIZ_CONFIG['color_raw'], edgecolor='black')
    axes[0].bar(x + width/2, filtered_counts, width, label='Filtered',
               color=VIZ_CONFIG['color_filtered'], edgecolor='black')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Dataset Scale')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (r, f) in enumerate(zip(raw_counts, filtered_counts)):
        axes[0].text(i - width/2, r, f'{r:,}', ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width/2, f, f'{f:,}', ha='center', va='bottom', fontsize=9)
    
    # Success rate comparison
    # Raw: success rate of CuRobo
    # Filtered: ratio of filtered successful to raw successful
    categories = ['Trajectory (CuRobo)', 'Trajectory (Filter)']
    raw_success_rate = raw_stats['overall_success_rate'] * 100
    filter_success_ratio = (filtered_stats['num_successful_total'] / raw_stats['num_successful_total'] * 100
                           if raw_stats['num_successful_total'] > 0 else 0)
    rates = [raw_success_rate, filter_success_ratio]
    colors = [VIZ_CONFIG['color_success'], VIZ_CONFIG['color_filtered']]
    
    bars = axes[1].bar(categories, rates, color=colors, edgecolor='black')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_title('Success Rate vs Filter Ratio')
    axes[1].set_ylim([0, 105])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    save_figure(fig, output_dir / "dataset_scale")
    plt.close()


def analyze_diversity(traj_data_dict: Dict, output_dir: Path, max_samples: int = 5000):
    """Analyze trajectory diversity."""
    print("\nAnalyzing trajectory diversity...")
    
    # Collect all start and goal states
    all_start_states = []
    all_goal_states = []
    
    for scene_data in traj_data_dict.values():
        for traj_data in scene_data.values():
            all_start_states.append(traj_data.start_state.cpu().numpy())
            all_goal_states.append(traj_data.goal_state.cpu().numpy())
    
    all_start_states = np.vstack(all_start_states)
    all_goal_states = np.vstack(all_goal_states)
    
    # Subsample if too large
    if len(all_start_states) > max_samples:
        indices = np.random.choice(len(all_start_states), max_samples, replace=False)
        all_start_states = all_start_states[indices]
        all_goal_states = all_goal_states[indices]
        print(f"  Subsampled to {max_samples} trajectories for visualization")
    
    # Compute diversity metrics
    print("  Computing diversity metrics...")
    start_diversity = compute_diversity_metrics(all_start_states)
    goal_diversity = compute_diversity_metrics(all_goal_states)
    
    print(f"  Start state diversity:")
    print(f"    Mean pairwise distance: {start_diversity['mean_distance']:.3f}")
    print(f"    Std pairwise distance: {start_diversity['std_distance']:.3f}")
    print(f"  Goal state diversity:")
    print(f"    Mean pairwise distance: {goal_diversity['mean_distance']:.3f}")
    print(f"    Std pairwise distance: {goal_diversity['std_distance']:.3f}")
    
    # t-SNE visualization
    print("  Computing t-SNE embeddings...")
    start_tsne = compute_tsne(all_start_states, perplexity=min(30, len(all_start_states)//3))
    goal_tsne = compute_tsne(all_goal_states, perplexity=min(30, len(all_goal_states)//3))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    axes[0].scatter(start_tsne[:, 0], start_tsne[:, 1], 
                   c=VIZ_CONFIG['color_raw'], alpha=VIZ_CONFIG['alpha_scatter'], s=20)
    axes[0].set_title('Start State Diversity (t-SNE)')
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(goal_tsne[:, 0], goal_tsne[:, 1],
                   c=VIZ_CONFIG['color_success'], alpha=VIZ_CONFIG['alpha_scatter'], s=20)
    axes[1].set_title('Goal State Diversity (t-SNE)')
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_dir / "diversity_tsne")
    plt.close()
    
    return {'start': start_diversity, 'goal': goal_diversity}


def analyze_object_angle_trajectories(traj_raw_dict: Dict, traj_filtered_dict: Dict, 
                                      output_dir: Path):
    """Analyze object angle evolution in trajectories."""
    print("\nAnalyzing object angle in trajectories...")
    
    obj_idx = TRAJ_ANALYSIS['object_joint_index']
    
    # Collect object angle data
    raw_initial_angles = []
    raw_final_angles = []
    raw_delta_angles = []
    
    filtered_initial_angles = []
    filtered_final_angles = []
    filtered_delta_angles = []
    
    for scene_data in traj_raw_dict.values():
        for traj_data in scene_data.values():
            # Filter successful only
            successful = filter_successful_trajectories(traj_data)
            trajs = successful.traj.cpu().numpy()
            
            initial = trajs[:, 0, obj_idx]
            final = trajs[:, -1, obj_idx]
            delta = final - initial
            
            raw_initial_angles.extend(initial)
            raw_final_angles.extend(final)
            raw_delta_angles.extend(delta)
    
    for scene_data in traj_filtered_dict.values():
        for traj_data in scene_data.values():
            trajs = traj_data.traj.cpu().numpy()
            
            initial = trajs[:, 0, obj_idx]
            final = trajs[:, -1, obj_idx]
            delta = final - initial
            
            filtered_initial_angles.extend(initial)
            filtered_final_angles.extend(final)
            filtered_delta_angles.extend(delta)
    
    raw_initial_angles = np.array(raw_initial_angles)
    raw_final_angles = np.array(raw_final_angles)
    raw_delta_angles = np.array(raw_delta_angles)
    
    filtered_initial_angles = np.array(filtered_initial_angles)
    filtered_final_angles = np.array(filtered_final_angles)
    filtered_delta_angles = np.array(filtered_delta_angles)
    
    print(f"  Raw (successful) object angles:")
    print(f"    Initial: mean={np.mean(raw_initial_angles):.3f}, std={np.std(raw_initial_angles):.3f}")
    print(f"    Final: mean={np.mean(raw_final_angles):.3f}, std={np.std(raw_final_angles):.3f}")
    print(f"    Delta: mean={np.mean(raw_delta_angles):.3f}, std={np.std(raw_delta_angles):.3f}")
    print(f"  Filtered object angles:")
    print(f"    Initial: mean={np.mean(filtered_initial_angles):.3f}, std={np.std(filtered_initial_angles):.3f}")
    print(f"    Final: mean={np.mean(filtered_final_angles):.3f}, std={np.std(filtered_final_angles):.3f}")
    print(f"    Delta: mean={np.mean(filtered_delta_angles):.3f}, std={np.std(filtered_delta_angles):.3f}")
    
    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Initial angles
    axes[0].hist(raw_initial_angles, bins=50, alpha=0.6, label='Raw (Success)',
                color=VIZ_CONFIG['color_success'], edgecolor='black')
    axes[0].hist(filtered_initial_angles, bins=50, alpha=0.6, label='Filtered',
                color=VIZ_CONFIG['color_filtered'], edgecolor='black')
    axes[0].set_xlabel('Object Angle (rad)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Initial Object Angle Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Final angles
    axes[1].hist(raw_final_angles, bins=50, alpha=0.6, label='Raw (Success)',
                color=VIZ_CONFIG['color_success'], edgecolor='black')
    axes[1].hist(filtered_final_angles, bins=50, alpha=0.6, label='Filtered',
                color=VIZ_CONFIG['color_filtered'], edgecolor='black')
    axes[1].set_xlabel('Object Angle (rad)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Final Object Angle Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Delta angles
    axes[2].hist(raw_delta_angles, bins=50, alpha=0.6, label='Raw (Success)',
                color=VIZ_CONFIG['color_success'], edgecolor='black')
    axes[2].hist(filtered_delta_angles, bins=50, alpha=0.6, label='Filtered',
                color=VIZ_CONFIG['color_filtered'], edgecolor='black')
    axes[2].set_xlabel('Δ Object Angle (rad)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Object Angle Change Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_dir / "object_angle_distributions")
    plt.close()


def analyze_trajectory_variance(traj_data_dict: Dict, output_dir: Path, 
                                max_trajs: int = 1000):
    """Analyze variance across trajectory timesteps."""
    print("\nAnalyzing trajectory variance...")
    
    # Collect sample trajectories
    all_trajs = []
    for scene_data in traj_data_dict.values():
        for traj_data in scene_data.values():
            all_trajs.append(traj_data.traj.cpu().numpy())
    
    all_trajs = np.vstack(all_trajs)
    
    # Subsample if needed
    if len(all_trajs) > max_trajs:
        indices = np.random.choice(len(all_trajs), max_trajs, replace=False)
        all_trajs = all_trajs[indices]
    
    print(f"  Analyzing {len(all_trajs)} trajectories...")
    
    # Compute variance heatmap
    variance = compute_trajectory_variance(all_trajs)
    
    fig, ax = plot_trajectory_heatmap(
        all_trajs, JOINT_NAMES,
        title=f"Trajectory Variance Over Time ({len(all_trajs)} trajectories)"
    )
    save_figure(fig, output_dir / "trajectory_variance_heatmap")
    plt.close()
    
    # Plot average trajectory per joint with std bands
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    mean_traj = np.mean(all_trajs, axis=0)
    std_traj = np.std(all_trajs, axis=0)
    timesteps = np.arange(mean_traj.shape[0])
    
    for i, (ax, joint_name) in enumerate(zip(axes[:len(JOINT_NAMES)], JOINT_NAMES)):
        ax.plot(timesteps, mean_traj[:, i], linewidth=2, label='Mean')
        ax.fill_between(timesteps, 
                        mean_traj[:, i] - std_traj[:, i],
                        mean_traj[:, i] + std_traj[:, i],
                        alpha=0.3, label='±1 std')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Joint Value')
        ax.set_title(joint_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    for ax in axes[len(JOINT_NAMES):]:
        ax.axis('off')
    
    plt.tight_layout()
    save_figure(fig, output_dir / "average_trajectory_per_joint")
    plt.close()


def main():
    """Main trajectory analysis pipeline."""
    print("=" * 80)
    print("Trajectory Data Analysis")
    print("=" * 80)
    
    # Setup
    setup_plotting_style(VIZ_CONFIG)
    create_output_dirs()
    
    output_dir = FIGURE_ROOT / "trajectory"
    
    # Load data
    print("\n" + "="*80)
    print("Loading Data")
    print("="*80)
    
    traj_raw = load_all_trajectory_data(ROBOT_NAME, SCENE_NAMES, OBJECT_NAMES,
                                       GRASP_IDS, DATA_ROOT, filtered=False, verbose=True)
    
    traj_filtered = load_all_trajectory_data(ROBOT_NAME, SCENE_NAMES, OBJECT_NAMES,
                                            GRASP_IDS, DATA_ROOT, filtered=True, verbose=True)
    
    # Analysis 1: Dataset statistics
    print("\n" + "="*80)
    print("Analysis 1: Dataset Scale Statistics")
    print("="*80)
    
    raw_stats = compute_dataset_statistics(traj_raw, "raw")
    filtered_stats = compute_dataset_statistics(traj_filtered, "filtered")
    
    print(f"\nRaw Dataset:")
    print(f"  Scenes: {raw_stats['num_scenes']}")
    print(f"  Grasps: {raw_stats['num_grasps_total']}")
    print(f"  Total trajectories: {raw_stats['num_trajectories_total']:,}")
    print(f"  Successful: {raw_stats['num_successful_total']:,} ({raw_stats['overall_success_rate']:.2%})")
    
    print(f"\nFiltered Dataset:")
    print(f"  Scenes: {filtered_stats['num_scenes']}")
    print(f"  Grasps: {filtered_stats['num_grasps_total']}")
    print(f"  Total trajectories: {filtered_stats['num_trajectories_total']:,}")
    print(f"  Successful: {filtered_stats['num_successful_total']:,} ({filtered_stats['overall_success_rate']:.2%})")
    
    plot_dataset_scale_statistics(raw_stats, filtered_stats, output_dir)
    
    # Analysis 2: Diversity
    print("\n" + "="*80)
    print("Analysis 2: Trajectory Diversity")
    print("="*80)
    
    diversity_metrics = analyze_diversity(traj_filtered, FIGURE_ROOT / "diversity")
    
    # Analysis 3: Object angle
    print("\n" + "="*80)
    print("Analysis 3: Object Angle Analysis")
    print("="*80)
    
    analyze_object_angle_trajectories(traj_raw, traj_filtered, FIGURE_ROOT / "object_angle")
    
    # Analysis 4: Trajectory variance
    print("\n" + "="*80)
    print("Analysis 4: Trajectory Variance")
    print("="*80)
    
    analyze_trajectory_variance(traj_filtered, output_dir)
    
    # Save comprehensive summary
    summary = {
        'raw_statistics': raw_stats,
        'filtered_statistics': filtered_stats,
        'diversity_metrics': diversity_metrics,
    }
    
    summary_path = OUTPUT_ROOT / "trajectory_analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("✓ Trajectory Analysis Complete")
    print("="*80)


if __name__ == "__main__":
    main()
