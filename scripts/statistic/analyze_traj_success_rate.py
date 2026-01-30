"""
Analyze and visualize two-stage trajectory success rates from pipeline statistics.

This script loads pipeline statistics and generates visualizations for:
- Stage 1 (Raw): traj_success_count / traj_total_count (CuRobo planning success)
- Stage 2 (Filtered): filtered_success_count / traj_total_count (Trajectory filtering success)
- Comparison across scenes and mean rates
- Statistical summaries

Author: AutoMoMA Analysis Suite
Date: 2025-11-08
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_pipeline_statistics(stats_path: str) -> Dict:
    """Load pipeline statistics from JSON file."""
    print(f"Loading pipeline statistics from: {stats_path}")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    print(f"✓ Loaded statistics for {stats['total_scenes']} scenes")
    return stats


def compute_two_stage_success_rates(stats: Dict) -> Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute two-stage success rates for each scene.
    
    Stage 1 (Raw): traj_success_count / traj_total_count
    Stage 2 (Filtered): filtered_success_count / traj_total_count
    
    Returns:
        - stage1_rates: Dict mapping scene_name to stage1 success_rate
        - stage2_rates: Dict mapping scene_name to stage2 success_rate
        - scene_names: Array of scene names (sorted)
        - stage1_array: Array of stage1 success rates
        - stage2_array: Array of stage2 success rates
    """
    stage1_rates = {}
    stage2_rates = {}
    
    for scene_name, scene_data in stats['statistics_by_scene'].items():
        if not scene_data['has_results']:
            continue
        
        total_raw_success = 0
        total_filtered_success = 0
        total_count = 0
        
        for grasp_id, grasp_data in scene_data['grasp_results'].items():
            total_raw_success += grasp_data['traj_success_count']
            total_filtered_success += grasp_data['filtered_success_count']
            total_count += grasp_data['traj_total_count']
        
        if total_count > 0:
            stage1_rate = total_raw_success / total_count  # Raw planning success
            stage2_rate = total_filtered_success / total_count  # Filtered success
            stage1_rates[scene_name] = stage1_rate
            stage2_rates[scene_name] = stage2_rate
    
    # Sort by scene name (extract scene number for proper sorting)
    sorted_scenes = sorted(
        stage1_rates.keys(),
        key=lambda x: int(x.split('_')[1])
    )
    scene_names = np.array(sorted_scenes)
    stage1_array = np.array([stage1_rates[s] for s in sorted_scenes])
    stage2_array = np.array([stage2_rates[s] for s in sorted_scenes])
    
    return stage1_rates, stage2_rates, scene_names, stage1_array, stage2_array


def print_statistics(stage1_rates: Dict, stage2_rates: Dict, 
                     stage1_array: np.ndarray, stage2_array: np.ndarray):
    """Print statistical summary for both stages."""
    print("\n" + "="*80)
    print("TWO-STAGE SUCCESS RATE STATISTICS")
    print("="*80)
    
    print("\n" + "-"*80)
    print("STAGE 1 (Raw CuRobo Planning): traj_success_count / traj_total_count")
    print("-"*80)
    print(f"Mean success rate: {np.mean(stage1_array):.2%}")
    print(f"Std deviation: {np.std(stage1_array):.2%}")
    print(f"Min success rate: {np.min(stage1_array):.2%}")
    print(f"Max success rate: {np.max(stage1_array):.2%}")
    print(f"Median success rate: {np.median(stage1_array):.2%}")
    
    print("\n" + "-"*80)
    print("STAGE 2 (Filtered): filtered_success_count / traj_total_count")
    print("-"*80)
    print(f"Mean success rate: {np.mean(stage2_array):.2%}")
    print(f"Std deviation: {np.std(stage2_array):.2%}")
    print(f"Min success rate: {np.min(stage2_array):.2%}")
    print(f"Max success rate: {np.max(stage2_array):.2%}")
    print(f"Median success rate: {np.median(stage2_array):.2%}")
    
    print("\n" + "-"*80)
    print("FILTERING IMPACT")
    print("-"*80)
    filter_ratio = stage2_array / stage1_array
    print(f"Average filter ratio (Stage2/Stage1): {np.mean(filter_ratio):.2%}")
    print(f"Std deviation: {np.std(filter_ratio):.2%}")
    print(f"Min filter ratio: {np.min(filter_ratio):.2%}")
    print(f"Max filter ratio: {np.max(filter_ratio):.2%}")


def print_per_scene_statistics(stage1_rates: Dict, stage2_rates: Dict):
    """Print per-scene statistics."""
    print("\n" + "-"*80)
    print("PER-SCENE STATISTICS")
    print("-"*80)
    print(f"{'Scene':<20} {'Stage1 (Raw)':<15} {'Stage2 (Filter)':<15} {'Ratio':<10}")
    print("-"*80)
    
    sorted_scenes = sorted(stage1_rates.keys(), 
                          key=lambda x: int(x.split('_')[1]))
    for scene_name in sorted_scenes:
        s1 = stage1_rates[scene_name]
        s2 = stage2_rates[scene_name]
        ratio = s2 / s1 if s1 > 0 else 0
        print(f"{scene_name:<20} {s1:>13.2%}  {s2:>13.2%}  {ratio:>8.2%}")



def plot_stage1_success_rates(scene_names: np.ndarray, stage1_array: np.ndarray, 
                              output_dir: Path):
    """Create visualization for Stage 1 (Raw CuRobo planning) success rates."""
    print("\nGenerating Stage 1 visualization...")
    
    # Set up the style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.figsize': (16, 8),
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Calculate statistics
    mean_rate = np.mean(stage1_array)
    std_rate = np.std(stage1_array)
    
    # Plot 1: Bar chart with mean line
    colors = ['#3498db' if rate >= mean_rate else '#e67e22' 
              for rate in stage1_array]
    
    ax1 = axes[0]
    bars = ax1.bar(range(len(scene_names)), stage1_array * 100, 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add mean line
    ax1.axhline(y=mean_rate * 100, color='#e74c3c', linestyle='--', 
                linewidth=2.5, label=f'Mean: {mean_rate:.2%}')
    
    # Add +/- 1 std dev band
    ax1.fill_between(range(len(scene_names)), 
                      (mean_rate - std_rate) * 100,
                      (mean_rate + std_rate) * 100,
                      alpha=0.2, color='#e74c3c', label=f'±1 Std Dev')
    
    ax1.set_xlabel('Scene', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Stage 1: Raw CuRobo Planning Success Rate (traj_success_count / traj_total_count)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(scene_names)))
    ax1.set_xticklabels(scene_names, rotation=45, ha='right')
    ax1.set_ylim([0, max(stage1_array * 100) * 1.15])
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, stage1_array)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Distribution histogram
    ax2 = axes[1]
    n, bins, patches = ax2.hist(stage1_array * 100, bins=15, 
                                 color='#3498db', alpha=0.7, 
                                 edgecolor='black', linewidth=1.5)
    
    # Color the mean
    ax2.axvline(x=mean_rate * 100, color='#e74c3c', linestyle='--', 
               linewidth=2.5, label=f'Mean: {mean_rate:.2%}')
    ax2.axvline(x=np.median(stage1_array) * 100, color='#f39c12', 
               linestyle='--', linewidth=2.5, label=f'Median: {np.median(stage1_array):.2%}')
    
    ax2.set_xlabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Stage 1 Success Rates', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text box
    stats_text = (
        f'N Scenes: {len(stage1_array)}\n'
        f'Mean: {mean_rate:.2%}\n'
        f'Std Dev: {std_rate:.2%}\n'
        f'Min: {np.min(stage1_array):.2%}\n'
        f'Max: {np.max(stage1_array):.2%}'
    )
    ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "stage1_raw_success_rate.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Stage 1 visualization saved to: {output_path}")
    
    plt.close()


def plot_stage2_success_rates(scene_names: np.ndarray, stage2_array: np.ndarray, 
                              output_dir: Path):
    """Create visualization for Stage 2 (Filtered) success rates."""
    print("\nGenerating Stage 2 visualization...")
    
    # Set up the style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.figsize': (16, 8),
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Calculate statistics
    mean_rate = np.mean(stage2_array)
    std_rate = np.std(stage2_array)
    
    # Plot 1: Bar chart with mean line
    colors = ['#2ecc71' if rate >= mean_rate else '#e74c3c' 
              for rate in stage2_array]
    
    ax1 = axes[0]
    bars = ax1.bar(range(len(scene_names)), stage2_array * 100, 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add mean line
    ax1.axhline(y=mean_rate * 100, color='#9b59b6', linestyle='--', 
                linewidth=2.5, label=f'Mean: {mean_rate:.2%}')
    
    # Add +/- 1 std dev band
    ax1.fill_between(range(len(scene_names)), 
                      (mean_rate - std_rate) * 100,
                      (mean_rate + std_rate) * 100,
                      alpha=0.2, color='#9b59b6', label=f'±1 Std Dev')
    
    ax1.set_xlabel('Scene', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Stage 2: Filtered Success Rate (filtered_success_count / traj_total_count)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(scene_names)))
    ax1.set_xticklabels(scene_names, rotation=45, ha='right')
    ax1.set_ylim([0, max(stage2_array * 100) * 1.15])
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, stage2_array)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Distribution histogram
    ax2 = axes[1]
    n, bins, patches = ax2.hist(stage2_array * 100, bins=15, 
                                 color='#2ecc71', alpha=0.7, 
                                 edgecolor='black', linewidth=1.5)
    
    # Color the mean
    ax2.axvline(x=mean_rate * 100, color='#9b59b6', linestyle='--', 
               linewidth=2.5, label=f'Mean: {mean_rate:.2%}')
    ax2.axvline(x=np.median(stage2_array) * 100, color='#f39c12', 
               linestyle='--', linewidth=2.5, label=f'Median: {np.median(stage2_array):.2%}')
    
    ax2.set_xlabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Stage 2 Success Rates', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text box
    stats_text = (
        f'N Scenes: {len(stage2_array)}\n'
        f'Mean: {mean_rate:.2%}\n'
        f'Std Dev: {std_rate:.2%}\n'
        f'Min: {np.min(stage2_array):.2%}\n'
        f'Max: {np.max(stage2_array):.2%}'
    )
    ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "stage2_filtered_success_rate.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Stage 2 visualization saved to: {output_path}")
    
    plt.close()


def plot_comparison(scene_names: np.ndarray, stage1_array: np.ndarray, 
                   stage2_array: np.ndarray, output_dir: Path):
    """Create comparison visualization between Stage 1 and Stage 2."""
    print("\nGenerating comparison visualization...")
    
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.figsize': (16, 10),
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Side-by-side comparison
    x = np.arange(len(scene_names))
    width = 0.35
    
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, stage1_array * 100, width, label='Stage 1 (Raw Planning)',
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, stage2_array * 100, width, label='Stage 2 (Filtered)',
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Scene', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Two-Stage Success Rate Comparison Across Scenes', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scene_names, rotation=45, ha='right')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add mean lines
    mean1 = np.mean(stage1_array)
    mean2 = np.mean(stage2_array)
    ax1.axhline(y=mean1 * 100, color='#3498db', linestyle=':', linewidth=2, alpha=0.7)
    ax1.axhline(y=mean2 * 100, color='#2ecc71', linestyle=':', linewidth=2, alpha=0.7)
    
    # Plot 2: Filter ratio
    filter_ratio = stage2_array / stage1_array
    colors_ratio = ['#27ae60' if r >= np.mean(filter_ratio) else '#e67e22' 
                    for r in filter_ratio]
    
    ax2 = axes[1]
    bars3 = ax2.bar(range(len(scene_names)), filter_ratio * 100, 
                    color=colors_ratio, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    mean_ratio = np.mean(filter_ratio)
    ax2.axhline(y=mean_ratio * 100, color='#c0392b', linestyle='--', 
               linewidth=2.5, label=f'Mean Ratio: {mean_ratio:.2%}')
    ax2.fill_between(range(len(scene_names)), 
                     (mean_ratio - np.std(filter_ratio)) * 100,
                     (mean_ratio + np.std(filter_ratio)) * 100,
                     alpha=0.2, color='#c0392b')
    
    ax2.set_xlabel('Scene', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Filter Ratio (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Filtering Impact: Stage 2 / Stage 1', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(range(len(scene_names)))
    ax2.set_xticklabels(scene_names, rotation=45, ha='right')
    ax2.set_ylim([0, max(filter_ratio * 100) * 1.2])
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, ratio) in enumerate(zip(bars3, filter_ratio)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{ratio:.1%}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "comparison_two_stage.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison visualization saved to: {output_path}")
    
    plt.close()



def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Analyze two-stage trajectory success rates from pipeline statistics"
    )
    parser.add_argument(
        '--stats_path',
        type=str,
        default='output/collect/traj/summit_franka/pipeline_statistics.json',
        help='Path to pipeline statistics JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/statistics/trajectory_success_rate',
        help='Output directory for visualizations'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load statistics
    stats = load_pipeline_statistics(args.stats_path)
    
    # Compute two-stage success rates
    stage1_rates, stage2_rates, scene_names, stage1_array, stage2_array = (
        compute_two_stage_success_rates(stats)
    )
    
    # Print statistics
    print_statistics(stage1_rates, stage2_rates, stage1_array, stage2_array)
    print_per_scene_statistics(stage1_rates, stage2_rates)
    
    # Create visualizations
    plot_stage1_success_rates(scene_names, stage1_array, output_dir)
    plot_stage2_success_rates(scene_names, stage2_array, output_dir)
    plot_comparison(scene_names, stage1_array, stage2_array, output_dir)
    
    # Save detailed report
    filter_ratio = stage2_array / stage1_array
    report = {
        'stage1_raw_planning': {
            'mean_success_rate': float(np.mean(stage1_array)),
            'std_success_rate': float(np.std(stage1_array)),
            'min_success_rate': float(np.min(stage1_array)),
            'max_success_rate': float(np.max(stage1_array)),
            'median_success_rate': float(np.median(stage1_array)),
        },
        'stage2_filtered': {
            'mean_success_rate': float(np.mean(stage2_array)),
            'std_success_rate': float(np.std(stage2_array)),
            'min_success_rate': float(np.min(stage2_array)),
            'max_success_rate': float(np.max(stage2_array)),
            'median_success_rate': float(np.median(stage2_array)),
        },
        'filtering_impact': {
            'mean_filter_ratio': float(np.mean(filter_ratio)),
            'std_filter_ratio': float(np.std(filter_ratio)),
            'min_filter_ratio': float(np.min(filter_ratio)),
            'max_filter_ratio': float(np.max(filter_ratio)),
        },
        'num_scenes': len(stage1_array),
        'scene_stage1_rates': {str(k): float(v) for k, v in stage1_rates.items()},
        'scene_stage2_rates': {str(k): float(v) for k, v in stage2_rates.items()},
        'scene_filter_ratios': {
            str(k): float(stage2_rates[k] / stage1_rates[k]) 
            if stage1_rates[k] > 0 else 0.0
            for k in stage1_rates.keys()
        }
    }
    
    report_path = output_dir / "success_rate_analysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved to: {report_path}")
    
    print("\n" + "="*80)
    print("✓ Two-Stage Success Rate Analysis Complete!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  1. stage1_raw_success_rate.png")
    print(f"  2. stage2_filtered_success_rate.png")
    print(f"  3. comparison_two_stage.png")
    print(f"  4. success_rate_analysis_report.json")


if __name__ == "__main__":
    main()

