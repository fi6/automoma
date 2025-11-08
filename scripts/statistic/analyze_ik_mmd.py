"""
Analyze IK distribution using MMD (Maximum Mean Discrepancy) with IMQ kernel.

This script calculates MMD scores for seed quantity ablation study, comparing
generated IK distributions at different seed quantities (5k-80k) against a reference
distribution using the Inverse Multiquadric (IMQ) kernel.

The MMD metric measures the distance between two probability distributions:
- Lower MMD = distributions are more similar
- MMD = 0 means distributions are identical

IMQ Kernel: k(x,y) = C / (C + ||x-y||^2)^α where α = -0.5 (standard)

Author: Statistical Analysis Suite for AutoMoMA
Date: 2025-11-08
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from config import *
from utils import setup_plotting_style, save_figure


def imq_kernel(X: np.ndarray, Y: np.ndarray, alpha: float = -0.5, C: float = 1.0) -> np.ndarray:
    """
    Compute Inverse Multiquadric (IMQ) kernel between two sets of samples.
    
    K(x, y) = C / (C + ||x - y||^2)^α
    
    Args:
        X: First set of samples, shape [n, d]
        Y: Second set of samples, shape [m, d]
        alpha: Kernel parameter (default: -0.5, standard for IMQ)
        C: Kernel parameter (default: 1.0)
        
    Returns:
        Kernel matrix of shape [n, m]
    """
    # Compute pairwise squared Euclidean distances: ||x - y||^2
    # Using broadcasting: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    X_norm = np.sum(X**2, axis=1, keepdims=True)  # [n, 1]
    Y_norm = np.sum(Y**2, axis=1, keepdims=True)  # [m, 1]
    
    # [n, m] = [n, 1] + [1, m] - 2 * [n, d] @ [d, m]
    sq_dists = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
    
    # Ensure numerical stability (distances should be non-negative)
    sq_dists = np.maximum(sq_dists, 0.0)
    
    # Apply IMQ kernel formula
    kernel_matrix = C / np.power(C + sq_dists, -alpha)
    
    return kernel_matrix


def compute_mmd_imq(X: np.ndarray, Y: np.ndarray, alpha: float = -0.5, C: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions using IMQ kernel.
    
    MMD²(X, Y) = E[k(x, x')] + E[k(y, y')] - 2E[k(x, y)]
    
    Empirical estimator:
    MMD²(X,Y) = (1/n²)ΣΣk(xi,xj) + (1/m²)ΣΣk(yi,yj) - (2/nm)ΣΣk(xi,yj)
    
    Args:
        X: First sample set (e.g., generated data), shape [n, d]
        Y: Second sample set (e.g., reference data), shape [m, d]
        alpha: IMQ kernel parameter (default: -0.5)
        C: IMQ kernel parameter (default: 1.0)
        
    Returns:
        MMD score (non-negative scalar)
    """
    n = X.shape[0]
    m = Y.shape[0]
    
    # Compute kernel matrices
    K_XX = imq_kernel(X, X, alpha, C)
    K_YY = imq_kernel(Y, Y, alpha, C)
    K_XY = imq_kernel(X, Y, alpha, C)
    
    # Compute MMD² using the empirical estimator
    # Note: We exclude diagonal terms for unbiased estimate
    # E[k(x, x')] ≈ (1/(n(n-1))) * Σ_{i≠j} k(xi, xj)
    # E[k(y, y')] ≈ (1/(m(m-1))) * Σ_{i≠j} k(yi, yj)
    # E[k(x, y)] ≈ (1/(nm)) * Σ_i Σ_j k(xi, yj)
    
    # Sum off-diagonal elements (exclude diagonal where i=j)
    K_XX_sum = np.sum(K_XX) - np.trace(K_XX)
    K_YY_sum = np.sum(K_YY) - np.trace(K_YY)
    K_XY_sum = np.sum(K_XY)
    
    # Unbiased MMD² estimator
    mmd_squared = (K_XX_sum / (n * (n - 1)) if n > 1 else 0.0) + \
                  (K_YY_sum / (m * (m - 1)) if m > 1 else 0.0) - \
                  (2.0 * K_XY_sum / (n * m))
    
    # MMD should be non-negative; take square root
    # (small negative values may occur due to numerical errors)
    mmd = np.sqrt(np.maximum(mmd_squared, 0.0))
    
    return float(mmd)


def load_ik_from_cache(scene_name: str, grasp_id: int, num_seeds: int) -> Optional[Tuple]:
    """Load cached IK data if available.
    
    Uses the same cache path format as analyze_ik_curobo.py:
    {cache_dir}/{scene_name}_grasp_{grasp_id:04d}_seeds_{num_seeds}_ik.pt
    """
    cache_dir = IK_CUROBO['cache_dir']
    if not IK_CUROBO['enable_cache'] or not cache_dir.exists():
        return None
    
    cache_file = cache_dir / f"{scene_name}_grasp_{grasp_id:04d}_seeds_{num_seeds}_ik.pt"
    if not cache_file.exists():
        return None
    
    try:
        data = torch.load(cache_file)
        return data['start_ik'], data['goal_ik']
    except Exception as e:
        print(f"Warning: Failed to load cache {cache_file}: {e}")
        return None


def compute_mmd_ablation_study(output_dir: Path = None) -> pd.DataFrame:
    """
    Compute MMD scores for seed quantity ablation study.
    
    Uses configuration from config.py:
    - Scene names from IK_CUROBO['analysis_scenes'] or SCENE_NAMES
    - Grasp IDs from IK_CUROBO['analysis_grasp_ids'] or GRASP_IDS
    - Seed values from IK_CUROBO['num_seeds_values']
    - Reference seeds: highest value in num_seeds_values
    
    This function:
    1. Loads IK data for different seed quantities from cache
    2. Uses the highest seed count as reference "real" distribution
    3. Computes MMD with IMQ kernel between each seed quantity and reference
    4. Generates a comprehensive table and visualizations
    
    Args:
        output_dir: Directory to save results (default: from config)
        
    Returns:
        DataFrame with MMD scores for each seed quantity
    """
    if output_dir is None:
        output_dir = FIGURE_ROOT / "ik_mmd_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get configuration
    scene_names = IK_CUROBO.get('analysis_scenes', SCENE_NAMES)
    grasp_ids = IK_CUROBO.get('analysis_grasp_ids', GRASP_IDS)
    seed_values = sorted(IK_CUROBO['num_seeds_values'])  # e.g., [5000, 10000, 20000, 40000, 80000]
    reference_seeds = max(seed_values)  # Use highest seed count as reference
    
    print(f"\n{'='*80}")
    print(f"MMD-based Seed Quantity Ablation Study")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Seed values: {seed_values}")
    print(f"  Reference distribution: {reference_seeds} seeds")
    print(f"  Scenes: {len(scene_names)}")
    print(f"  Grasps per scene: {len(grasp_ids)}")
    print(f"  Kernel: IMQ (Inverse Multiquadric) with α=-0.5")
    print(f"  Cache directory: {IK_CUROBO['cache_dir']}")
    print(f"{'='*80}\n")
    
    # Results storage
    results = []
    
    # Process each scene and grasp
    total_pairs = len(scene_names) * len(grasp_ids)
    with tqdm(total=total_pairs, desc="Computing MMD scores") as pbar:
        for scene_name in scene_names:
            for grasp_id in grasp_ids:
                pbar.set_description(f"Scene: {scene_name}, Grasp: {grasp_id:04d}")
                
                # Load reference data (highest seed count)
                ref_data = load_ik_from_cache(scene_name, grasp_id, reference_seeds)
                if ref_data is None:
                    print(f"\n  ⚠️  Warning: No reference data for {scene_name}, grasp {grasp_id}")
                    pbar.update(1)
                    continue
                
                ref_start_ik, ref_goal_ik = ref_data
                ref_start_ik = ref_start_ik.cpu().numpy() if torch.is_tensor(ref_start_ik) else ref_start_ik
                ref_goal_ik = ref_goal_ik.cpu().numpy() if torch.is_tensor(ref_goal_ik) else ref_goal_ik
                
                # Compute MMD for each seed value
                for num_seeds in seed_values:
                    if num_seeds == reference_seeds:
                        # MMD between identical distributions is 0
                        mmd_start = 0.0
                        mmd_goal = 0.0
                    else:
                        # Load generated data
                        gen_data = load_ik_from_cache(scene_name, grasp_id, num_seeds)
                        if gen_data is None:
                            print(f"\n  ⚠️  Warning: No data for {scene_name}, grasp {grasp_id}, seeds {num_seeds}")
                            continue
                        
                        gen_start_ik, gen_goal_ik = gen_data
                        gen_start_ik = gen_start_ik.cpu().numpy() if torch.is_tensor(gen_start_ik) else gen_start_ik
                        gen_goal_ik = gen_goal_ik.cpu().numpy() if torch.is_tensor(gen_goal_ik) else gen_goal_ik
                        
                        # Compute MMD with IMQ kernel (α=-0.5)
                        mmd_start = compute_mmd_imq(gen_start_ik, ref_start_ik, alpha=-0.5, C=1.0)
                        mmd_goal = compute_mmd_imq(gen_goal_ik, ref_goal_ik, alpha=-0.5, C=1.0)
                    
                    results.append({
                        'scene': scene_name,
                        'grasp_id': grasp_id,
                        'num_seeds': num_seeds,
                        'mmd_start': mmd_start,
                        'mmd_goal': mmd_goal,
                        'mmd_avg': (mmd_start + mmd_goal) / 2.0
                    })
                
                pbar.update(1)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("\n✗ No valid data found! Please ensure IK cache exists.")
        print(f"  Cache directory: {IK_CUROBO['cache_dir']}")
        print("  Run analyze_ik_curobo.py first to generate cache.")
        return df
    
    # Aggregate statistics: mean and std across all scenes/grasps
    summary = df.groupby('num_seeds').agg({
        'mmd_start': ['mean', 'std'],
        'mmd_goal': ['mean', 'std'],
        'mmd_avg': ['mean', 'std']
    }).reset_index()
    
    summary.columns = ['Seeds', 'MMD Start (Mean)', 'MMD Start (Std)', 
                      'MMD Goal (Mean)', 'MMD Goal (Std)',
                      'MMD Average (Mean)', 'MMD Average (Std)']
    
    # Format to 4 decimal places
    for col in summary.columns[1:]:
        summary[col] = summary[col].apply(lambda x: round(x, 4))
    
    # Print summary table
    print("\n" + "="*80)
    print("MMD Score Summary Table (IMQ Kernel, α=-0.5)")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)
    print(f"\nTotal scene-grasp pairs analyzed: {len(df) // len(seed_values)}")
    print(f"Reference distribution: {reference_seeds} seeds")
    print(f"Note: Lower MMD = distribution closer to reference\n")
    
    # Save detailed results
    detailed_csv = output_dir / "mmd_ablation_detailed.csv"
    df.to_csv(detailed_csv, index=False, float_format='%.4f')
    print(f"✓ Saved detailed results to: {detailed_csv}")
    
    # Save summary table
    summary_csv = output_dir / "mmd_ablation_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"✓ Saved summary table to: {summary_csv}")
    
    # Also save as Excel for better compatibility
    try:
        summary_excel = output_dir / "mmd_ablation_summary.xlsx"
        summary.to_excel(summary_excel, index=False)
        print(f"✓ Saved summary table to: {summary_excel}")
    except Exception as e:
        print(f"⚠️  Could not save Excel file: {e}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_mmd_results(df, summary, output_dir, reference_seeds)
    
    return summary


def visualize_mmd_results(df: pd.DataFrame, summary: pd.DataFrame, 
                         output_dir: Path, reference_seeds: int):
    """
    Generate comprehensive visualizations for MMD ablation study.
    
    Creates:
    1. Line plot: MMD vs seed count (with error bars)
    2. Heatmap: MMD scores across scenes/grasps
    3. Box plot: Distribution of MMD scores per seed count
    """
    setup_plotting_style(VIZ_CONFIG)
    
    # Figure 1: MMD vs Seed Count (Line plot with error bars)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Start IK
    ax1.errorbar(summary['Seeds'], summary['MMD Start (Mean)'], 
                yerr=summary['MMD Start (Std)'],
                marker='o', markersize=8, linewidth=2, capsize=5,
                label='Start IK', color='#3498db')
    ax1.set_xlabel('Number of Seeds', fontsize=12)
    ax1.set_ylabel('MMD Score', fontsize=12)
    ax1.set_title('MMD vs Seed Count (Start IK)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Perfect Match')
    ax1.legend()
    
    # Goal IK
    ax2.errorbar(summary['Seeds'], summary['MMD Goal (Mean)'], 
                yerr=summary['MMD Goal (Std)'],
                marker='s', markersize=8, linewidth=2, capsize=5,
                label='Goal IK', color='#e74c3c')
    ax2.set_xlabel('Number of Seeds', fontsize=12)
    ax2.set_ylabel('MMD Score', fontsize=12)
    ax2.set_title('MMD vs Seed Count (Goal IK)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Perfect Match')
    ax2.legend()
    
    plt.tight_layout()
    save_figure(fig, output_dir / "mmd_vs_seeds_lineplot")
    plt.close()
    print("  ✓ Saved line plot")
    
    # Figure 2: Combined comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(summary['Seeds'], summary['MMD Start (Mean)'], 
                yerr=summary['MMD Start (Std)'],
                marker='o', markersize=8, linewidth=2, capsize=5,
                label='Start IK', color='#3498db')
    ax.errorbar(summary['Seeds'], summary['MMD Goal (Mean)'], 
                yerr=summary['MMD Goal (Std)'],
                marker='s', markersize=8, linewidth=2, capsize=5,
                label='Goal IK', color='#e74c3c')
    ax.set_xlabel('Number of Seeds', fontsize=12)
    ax.set_ylabel('MMD Score (IMQ Kernel, α=-0.5)', fontsize=12)
    ax.set_title(f'MMD-based Seed Quantity Ablation Study\n(Reference: {reference_seeds} seeds)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Perfect Match')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    save_figure(fig, output_dir / "mmd_vs_seeds_combined")
    plt.close()
    print("  ✓ Saved combined plot")
    
    # Figure 3: Box plot - Distribution of MMD scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data for box plots
    seed_values = sorted(df['num_seeds'].unique())
    
    # Start IK box plot
    start_data = [df[df['num_seeds'] == seeds]['mmd_start'].values 
                  for seeds in seed_values]
    bp1 = ax1.boxplot(start_data, labels=[f"{s//1000}k" for s in seed_values],
                      patch_artist=True, showmeans=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.6)
    ax1.set_xlabel('Number of Seeds', fontsize=12)
    ax1.set_ylabel('MMD Score', fontsize=12)
    ax1.set_title('MMD Distribution (Start IK)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Goal IK box plot
    goal_data = [df[df['num_seeds'] == seeds]['mmd_goal'].values 
                 for seeds in seed_values]
    bp2 = ax2.boxplot(goal_data, labels=[f"{s//1000}k" for s in seed_values],
                      patch_artist=True, showmeans=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('#e74c3c')
        patch.set_alpha(0.6)
    ax2.set_xlabel('Number of Seeds', fontsize=12)
    ax2.set_ylabel('MMD Score', fontsize=12)
    ax2.set_title('MMD Distribution (Goal IK)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, output_dir / "mmd_distribution_boxplot")
    plt.close()
    print("  ✓ Saved box plot")
    
    # Figure 4: Heatmap (if multiple scenes/grasps)
    if len(df['scene'].unique()) > 1 or len(df['grasp_id'].unique()) > 1:
        # Create pivot table for heatmap
        pivot_start = df.pivot_table(values='mmd_start', 
                                     index=['scene', 'grasp_id'], 
                                     columns='num_seeds')
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(pivot_start) * 0.3)))
        sns.heatmap(pivot_start, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                   ax=ax, cbar_kws={'label': 'MMD Score'})
        ax.set_xlabel('Number of Seeds', fontsize=12)
        ax.set_ylabel('Scene - Grasp', fontsize=12)
        ax.set_title('MMD Heatmap (Start IK)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_figure(fig, output_dir / "mmd_heatmap_start")
        plt.close()
        print("  ✓ Saved heatmap")


def main():
    """Main entry point for MMD-based seed quantity ablation study.
    
    Uses configuration from config.py - no command-line arguments needed.
    Edit config.py to change:
    - IK_CUROBO['num_seeds_values']: seed quantities to analyze
    - IK_CUROBO['analysis_scenes']: scenes to analyze
    - IK_CUROBO['analysis_grasp_ids']: grasp IDs to analyze
    - IK_CUROBO['cache_dir']: where to find cached IK data
    """
    # Setup plotting
    setup_plotting_style(VIZ_CONFIG)
    
    # Print configuration
    print("\n" + "="*80)
    print("Configuration (from config.py):")
    print("="*80)
    print(f"Scenes: {IK_CUROBO.get('analysis_scenes', SCENE_NAMES)}")
    print(f"Grasps: {IK_CUROBO.get('analysis_grasp_ids', GRASP_IDS)}")
    print(f"Seed values: {sorted(IK_CUROBO['num_seeds_values'])}")
    print(f"Cache directory: {IK_CUROBO['cache_dir']}")
    print("="*80)
    
    # Run analysis
    summary = compute_mmd_ablation_study()
    
    if len(summary) > 0:
        print("\n" + "="*80)
        print("Analysis complete! 🎉")
        print("="*80)
        print("\nOutput files:")
        output_dir = FIGURE_ROOT / "ik_mmd_ablation"
        print(f"  Summary CSV: {output_dir / 'mmd_ablation_summary.csv'}")
        print(f"  Summary Excel: {output_dir / 'mmd_ablation_summary.xlsx'}")
        print(f"  Detailed CSV: {output_dir / 'mmd_ablation_detailed.csv'}")
        print(f"  Visualizations: {output_dir}/*.png")
        print("\nTo change configuration, edit config.py:")
        print("  - IK_CUROBO['num_seeds_values']: seed quantities")
        print("  - IK_CUROBO['analysis_scenes']: scenes")
        print("  - IK_CUROBO['analysis_grasp_ids']: grasp IDs")
    else:
        print("\n✗ Analysis failed. Please check cache and run analyze_ik_curobo.py first.")


if __name__ == "__main__":
    main()
