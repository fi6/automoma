"""
Cluster Quantity Ablation Study for IK Solutions.

This script performs an ablation study to determine the optimal number of clusters
for IK solutions by evaluating multiple clustering metrics:
- Gap Statistic: Measures cluster compactness compared to random data
- Silhouette Score (轮廓系数): Measures how well samples fit their clusters
- Calinski-Harabasz Index (CHI/方差比指数): Ratio of between-cluster to within-cluster variance
- Davies-Bouldin Index (DBI指数): Average similarity between clusters (lower is better)

The study tests cluster numbers: [10, 25, 50, 75, 100, 125, 150]

Author: Statistical Analysis Suite for AutoMoMA
Date: 2025-11-08
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist

from config import *
from utils import setup_plotting_style, save_figure


def compute_gap_statistic(data: np.ndarray, n_clusters: int, n_references: int = 10, 
                          random_state: int = 42) -> tuple:
    """
    Compute Gap Statistic for given number of clusters.
    
    Gap(k) = E[log(W_k^*)] - log(W_k)
    where W_k is within-cluster dispersion, W_k^* is for random reference data
    
    Args:
        data: Data to cluster [N, D]
        n_clusters: Number of clusters
        n_references: Number of random reference datasets to generate
        random_state: Random seed
        
    Returns:
        Tuple of (gap_value, gap_std, w_k, w_k_star_mean)
    """
    np.random.seed(random_state)
    
    # Fit KMeans on actual data
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    
    # Compute within-cluster dispersion W_k
    w_k = 0
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            w_k += np.sum(np.linalg.norm(cluster_points - centers[i], axis=1) ** 2)
    
    # Generate reference datasets and compute W_k*
    w_k_stars = []
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    
    for _ in range(n_references):
        # Generate random reference data uniformly in the data range
        random_data = np.random.uniform(data_min, data_max, size=data.shape)
        
        # Fit KMeans on random data
        kmeans_random = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        random_labels = kmeans_random.fit_predict(random_data)
        random_centers = kmeans_random.cluster_centers_
        
        # Compute within-cluster dispersion for random data
        w_k_star = 0
        for i in range(n_clusters):
            cluster_points = random_data[random_labels == i]
            if len(cluster_points) > 0:
                w_k_star += np.sum(np.linalg.norm(cluster_points - random_centers[i], axis=1) ** 2)
        
        w_k_stars.append(w_k_star)
    
    # Compute Gap statistic
    log_w_k = np.log(w_k + 1e-10)
    log_w_k_stars = np.log(np.array(w_k_stars) + 1e-10)
    gap = np.mean(log_w_k_stars) - log_w_k
    gap_std = np.std(log_w_k_stars)
    
    return gap, gap_std, w_k, np.mean(w_k_stars)


def compute_elbow_metric(data: np.ndarray, n_clusters: int, random_state: int = 42) -> float:
    """
    Compute within-cluster sum of squares (WCSS) for elbow method.
    
    Args:
        data: Data to cluster [N, D]
        n_clusters: Number of clusters
        random_state: Random seed
        
    Returns:
        WCSS value
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(data)
    return kmeans.inertia_


def perform_clustering_ablation(data: np.ndarray, cluster_numbers: list, 
                                random_state: int = 42) -> dict:
    """
    Perform clustering ablation study with multiple metrics.
    
    Args:
        data: IK data to cluster [N, D]
        cluster_numbers: List of cluster numbers to test
        random_state: Random seed
        
    Returns:
        Dictionary with metrics for each cluster number
    """
    results = {
        'n_clusters': [],
        'elbow': [],
        'gap_statistic': [],
        'gap_std': [],
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': [],
    }
    
    print(f"\nPerforming clustering ablation on {data.shape[0]} samples, {data.shape[1]} dimensions")
    print(f"Testing cluster numbers: {cluster_numbers}")
    print("-" * 80)
    
    for n_clusters in cluster_numbers:
        print(f"\nEvaluating {n_clusters} clusters...")
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(data)
        
        # Elbow metric (WCSS)
        elbow = compute_elbow_metric(data, n_clusters, random_state)
        print(f"  Elbow (WCSS): {elbow:.2f}")
        
        # Gap Statistic (may take longer)
        print(f"  Computing Gap Statistic...")
        gap, gap_std, w_k, w_k_star = compute_gap_statistic(data, n_clusters, 
                                                             n_references=10, 
                                                             random_state=random_state)
        print(f"  Gap Statistic: {gap:.4f} ± {gap_std:.4f}")
        
        # Silhouette Score
        silhouette = silhouette_score(data, labels, sample_size=min(10000, len(data)))
        print(f"  Silhouette Score: {silhouette:.4f}")
        
        # Calinski-Harabasz Index
        chi = calinski_harabasz_score(data, labels)
        print(f"  Calinski-Harabasz Index: {chi:.2f}")
        
        # Davies-Bouldin Index (lower is better)
        dbi = davies_bouldin_score(data, labels)
        print(f"  Davies-Bouldin Index: {dbi:.4f}")
        
        # Store results
        results['n_clusters'].append(n_clusters)
        results['elbow'].append(elbow)
        results['gap_statistic'].append(gap)
        results['gap_std'].append(gap_std)
        results['silhouette'].append(silhouette)
        results['calinski_harabasz'].append(chi)
        results['davies_bouldin'].append(dbi)
    
    return results


def visualize_ablation_results(results: dict, output_dir: Path, scene_name: str, 
                               grasp_id: int, ik_type: str):
    """
    Create comprehensive visualizations of clustering ablation results.
    
    Args:
        results: Dictionary with metrics
        output_dir: Output directory for figures
        scene_name: Scene name
        grasp_id: Grasp ID
        ik_type: 'start' or 'goal'
    """
    n_clusters = results['n_clusters']
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Cluster Quantity Ablation Study ({ik_type.capitalize()} IK)\n'
                 f'{scene_name}, Grasp {grasp_id:04d}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Elbow Method
    ax = axes[0, 0]
    ax.plot(n_clusters, results['elbow'], 'o-', linewidth=2, markersize=8,
           color='#2E86AB')
    ax.set_xlabel('Number of Clusters', fontsize=12)
    ax.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
    ax.set_title('Elbow Method', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add elbow point annotation (heuristic: max curvature)
    elbow_values = np.array(results['elbow'])
    if len(elbow_values) >= 3:
        # Compute second derivative
        second_deriv = np.diff(elbow_values, n=2)
        elbow_idx = np.argmax(np.abs(second_deriv)) + 1
        ax.axvline(n_clusters[elbow_idx], color='red', linestyle='--', alpha=0.5, 
                  label=f'Elbow at k={n_clusters[elbow_idx]}')
        ax.legend(fontsize=10)
    
    # 2. Gap Statistic
    ax = axes[0, 1]
    gap_values = np.array(results['gap_statistic'])
    gap_stds = np.array(results['gap_std'])
    ax.errorbar(n_clusters, gap_values, yerr=gap_stds, fmt='o-', linewidth=2, 
               markersize=8, capsize=5, color='#A23B72')
    ax.set_xlabel('Number of Clusters', fontsize=12)
    ax.set_ylabel('Gap Statistic', fontsize=12)
    ax.set_title('Gap Statistic (Higher is Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Optimal k: first k where Gap(k) >= Gap(k+1) - s_{k+1}
    optimal_gap_idx = None
    for i in range(len(gap_values) - 1):
        if gap_values[i] >= gap_values[i+1] - gap_stds[i+1]:
            optimal_gap_idx = i
            break
    if optimal_gap_idx is not None:
        ax.axvline(n_clusters[optimal_gap_idx], color='red', linestyle='--', alpha=0.5,
                  label=f'Optimal k={n_clusters[optimal_gap_idx]}')
        ax.legend(fontsize=10)
    
    # 3. Silhouette Score
    ax = axes[0, 2]
    ax.plot(n_clusters, results['silhouette'], 'o-', linewidth=2, markersize=8,
           color='#F18F01')
    ax.set_xlabel('Number of Clusters', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score (Higher is Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    
    # Best silhouette
    best_silh_idx = np.argmax(results['silhouette'])
    ax.axvline(n_clusters[best_silh_idx], color='red', linestyle='--', alpha=0.5,
              label=f'Best k={n_clusters[best_silh_idx]}')
    ax.legend(fontsize=10)
    
    # 4. Calinski-Harabasz Index
    ax = axes[1, 0]
    ax.plot(n_clusters, results['calinski_harabasz'], 'o-', linewidth=2, markersize=8,
           color='#6A994E')
    ax.set_xlabel('Number of Clusters', fontsize=12)
    ax.set_ylabel('Calinski-Harabasz Index', fontsize=12)
    ax.set_title('Calinski-Harabasz Index (Higher is Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Best CHI
    best_chi_idx = np.argmax(results['calinski_harabasz'])
    ax.axvline(n_clusters[best_chi_idx], color='red', linestyle='--', alpha=0.5,
              label=f'Best k={n_clusters[best_chi_idx]}')
    ax.legend(fontsize=10)
    
    # 5. Davies-Bouldin Index
    ax = axes[1, 1]
    ax.plot(n_clusters, results['davies_bouldin'], 'o-', linewidth=2, markersize=8,
           color='#BC4B51')
    ax.set_xlabel('Number of Clusters', fontsize=12)
    ax.set_ylabel('Davies-Bouldin Index', fontsize=12)
    ax.set_title('Davies-Bouldin Index (Lower is Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Best DBI
    best_dbi_idx = np.argmin(results['davies_bouldin'])
    ax.axvline(n_clusters[best_dbi_idx], color='red', linestyle='--', alpha=0.5,
              label=f'Best k={n_clusters[best_dbi_idx]}')
    ax.legend(fontsize=10)
    
    # 6. Summary comparison (normalized)
    ax = axes[1, 2]
    
    # Normalize metrics to [0, 1] for comparison
    def normalize(values, higher_is_better=True):
        values = np.array(values)
        vmin, vmax = values.min(), values.max()
        if vmax == vmin:
            return np.ones_like(values)
        normalized = (values - vmin) / (vmax - vmin)
        return normalized if higher_is_better else 1 - normalized
    
    norm_silh = normalize(results['silhouette'], higher_is_better=True)
    norm_chi = normalize(results['calinski_harabasz'], higher_is_better=True)
    norm_dbi = normalize(results['davies_bouldin'], higher_is_better=False)
    norm_gap = normalize(results['gap_statistic'], higher_is_better=True)
    
    # Composite score (equal weights)
    composite = (norm_silh + norm_chi + norm_dbi + norm_gap) / 4
    
    ax.plot(n_clusters, norm_silh, 'o-', label='Silhouette', linewidth=2, markersize=6)
    ax.plot(n_clusters, norm_chi, 's-', label='Calinski-Harabasz', linewidth=2, markersize=6)
    ax.plot(n_clusters, norm_dbi, '^-', label='Davies-Bouldin', linewidth=2, markersize=6)
    ax.plot(n_clusters, norm_gap, 'd-', label='Gap Statistic', linewidth=2, markersize=6)
    ax.plot(n_clusters, composite, 'X-', label='Composite Score', linewidth=3, 
           markersize=10, color='black', alpha=0.7)
    
    ax.set_xlabel('Number of Clusters', fontsize=12)
    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title('Normalized Metrics Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Best composite
    best_composite_idx = np.argmax(composite)
    ax.axvline(n_clusters[best_composite_idx], color='red', linestyle='--', alpha=0.5,
              label=f'Best Composite k={n_clusters[best_composite_idx]}')
    
    plt.tight_layout()
    save_figure(fig, output_dir / f"{scene_name}_grasp_{grasp_id:04d}_{ik_type}_cluster_ablation")
    plt.close()
    
    print(f"\n  ✓ Saved cluster ablation visualization")


def create_results_table(results: dict, output_dir: Path, scene_name: str, 
                        grasp_id: int, ik_type: str) -> pd.DataFrame:
    """
    Create a formatted table with all clustering metrics.
    
    Args:
        results: Dictionary with metrics
        output_dir: Output directory
        scene_name: Scene name
        grasp_id: Grasp ID
        ik_type: 'start' or 'goal'
        
    Returns:
        DataFrame with formatted results
    """
    # Create DataFrame
    df = pd.DataFrame({
        'Clusters': results['n_clusters'],
        'Elbow (WCSS)': results['elbow'],
        'Gap Statistic': results['gap_statistic'],
        'Gap Std': results['gap_std'],
        'Silhouette': results['silhouette'],
        'Calinski-Harabasz': results['calinski_harabasz'],
        'Davies-Bouldin': results['davies_bouldin'],
    })
    
    # Compute normalized composite score
    def normalize(values, higher_is_better=True):
        values = np.array(values)
        vmin, vmax = values.min(), values.max()
        if vmax == vmin:
            return np.ones_like(values)
        normalized = (values - vmin) / (vmax - vmin)
        return normalized if higher_is_better else 1 - normalized
    
    norm_silh = normalize(df['Silhouette'].values, higher_is_better=True)
    norm_chi = normalize(df['Calinski-Harabasz'].values, higher_is_better=True)
    norm_dbi = normalize(df['Davies-Bouldin'].values, higher_is_better=False)
    norm_gap = normalize(df['Gap Statistic'].values, higher_is_better=True)
    
    df['Composite Score'] = (norm_silh + norm_chi + norm_dbi + norm_gap) / 4
    
    # Add recommendations
    recommendations = []
    for i, row in df.iterrows():
        notes = []
        
        # Check if best for each metric
        if row['Silhouette'] == df['Silhouette'].max():
            notes.append('Best Silhouette')
        if row['Calinski-Harabasz'] == df['Calinski-Harabasz'].max():
            notes.append('Best CH Index')
        if row['Davies-Bouldin'] == df['Davies-Bouldin'].min():
            notes.append('Best DB Index')
        if row['Gap Statistic'] == df['Gap Statistic'].max():
            notes.append('Best Gap Stat')
        if row['Composite Score'] == df['Composite Score'].max():
            notes.append('⭐ BEST OVERALL')
        
        recommendations.append('; '.join(notes) if notes else '')
    
    df['Recommendation'] = recommendations
    
    # Save to CSV
    csv_path = output_dir / f"{scene_name}_grasp_{grasp_id:04d}_{ik_type}_cluster_ablation.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  ✓ Saved results table to {csv_path.name}")
    
    # Save formatted table for paper
    excel_path = output_dir / f"{scene_name}_grasp_{grasp_id:04d}_{ik_type}_cluster_ablation.xlsx"
    try:
        df.to_excel(excel_path, index=False, float_format='%.4f')
        print(f"  ✓ Saved Excel table to {excel_path.name}")
    except ImportError:
        print(f"  ⚠️  openpyxl not installed, skipping Excel export")
    
    return df


def print_summary_table(df: pd.DataFrame, ik_type: str):
    """Print a formatted summary table to console."""
    print(f"\n{'='*100}")
    print(f"Cluster Quantity Ablation Study - {ik_type.upper()} IK")
    print(f"{'='*100}")
    print(f"{'Clusters':>10} | {'Elbow':>12} | {'Gap Stat':>10} | {'Silhouette':>11} | "
          f"{'CH Index':>12} | {'DB Index':>10} | {'Composite':>10} | {'Recommendation':>20}")
    print('-' * 100)
    
    for _, row in df.iterrows():
        print(f"{row['Clusters']:>10} | "
              f"{row['Elbow (WCSS)']:>12,.0f} | "
              f"{row['Gap Statistic']:>10.4f} | "
              f"{row['Silhouette']:>11.4f} | "
              f"{row['Calinski-Harabasz']:>12,.2f} | "
              f"{row['Davies-Bouldin']:>10.4f} | "
              f"{row['Composite Score']:>10.4f} | "
              f"{row['Recommendation']:>20}")
    
    print('=' * 100)
    
    # Find best overall
    best_idx = df['Composite Score'].idxmax()
    best_k = df.loc[best_idx, 'Clusters']
    best_score = df.loc[best_idx, 'Composite Score']
    
    print(f"\n⭐ RECOMMENDATION: Use {best_k} clusters (Composite Score: {best_score:.4f})")
    print(f"\nMetric-specific recommendations:")
    print(f"  - Best Silhouette: {df.loc[df['Silhouette'].idxmax(), 'Clusters']} clusters")
    print(f"  - Best CH Index: {df.loc[df['Calinski-Harabasz'].idxmax(), 'Clusters']} clusters")
    print(f"  - Best DB Index: {df.loc[df['Davies-Bouldin'].idxmin(), 'Clusters']} clusters")
    print(f"  - Best Gap Stat: {df.loc[df['Gap Statistic'].idxmax(), 'Clusters']} clusters")
    print('=' * 100)


def main():
    """Main cluster ablation analysis pipeline."""
    print("=" * 80)
    print("IK Cluster Quantity Ablation Study")
    print("=" * 80)
    
    # Setup
    setup_plotting_style(VIZ_CONFIG)
    output_dir = FIGURE_ROOT / "ik_cluster_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    cache_file = Path("output/statistics/ik_curobo_cache/scene_0_seed_0_grasp_0000_seeds_20000_ik.pt")
    scene_name = "scene_0_seed_0"
    grasp_id = 0
    cluster_numbers = [10, 25, 50, 75, 100, 125, 150]
    
    print(f"\nConfiguration:")
    print(f"  Cache file: {cache_file}")
    print(f"  Scene: {scene_name}")
    print(f"  Grasp ID: {grasp_id}")
    print(f"  Cluster numbers to test: {cluster_numbers}")
    print(f"  Output directory: {output_dir}")
    
    # Load data
    print(f"\nLoading IK data from cache...")
    if not cache_file.exists():
        print(f"❌ Error: Cache file not found: {cache_file}")
        print(f"Please run analyze_ik_curobo.py first to generate the cache.")
        return
    
    try:
        data = torch.load(cache_file, map_location='cpu', weights_only=False)
        start_ik = data['start_ik'].cpu().numpy()
        goal_ik = data['goal_ik'].cpu().numpy()
        print(f"  ✓ Loaded cache file")
        print(f"    - Start IK: {start_ik.shape}")
        print(f"    - Goal IK: {goal_ik.shape}")
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        return
    
    # Analyze START IK
    print(f"\n{'='*80}")
    print(f"Analyzing START IK Clustering")
    print(f"{'='*80}")
    
    start_results = perform_clustering_ablation(
        start_ik[:, :-1],  # Exclude object angle (last dimension)
        cluster_numbers,
        random_state=42
    )
    
    # Create visualizations for start IK
    print(f"\nGenerating visualizations for START IK...")
    visualize_ablation_results(start_results, output_dir, scene_name, grasp_id, 'start')
    
    # Create results table for start IK
    print(f"\nCreating results table for START IK...")
    start_df = create_results_table(start_results, output_dir, scene_name, grasp_id, 'start')
    
    # Print summary
    print_summary_table(start_df, 'start')
    
    # Analyze GOAL IK
    print(f"\n{'='*80}")
    print(f"Analyzing GOAL IK Clustering")
    print(f"{'='*80}")
    
    goal_results = perform_clustering_ablation(
        goal_ik[:, :-1],  # Exclude object angle (last dimension)
        cluster_numbers,
        random_state=42
    )
    
    # Create visualizations for goal IK
    print(f"\nGenerating visualizations for GOAL IK...")
    visualize_ablation_results(goal_results, output_dir, scene_name, grasp_id, 'goal')
    
    # Create results table for goal IK
    print(f"\nCreating results table for GOAL IK...")
    goal_df = create_results_table(goal_results, output_dir, scene_name, grasp_id, 'goal')
    
    # Print summary
    print_summary_table(goal_df, 'goal')
    
    # Save combined summary
    combined_summary = {
        'scene_name': scene_name,
        'grasp_id': grasp_id,
        'cache_file': str(cache_file),
        'cluster_numbers': cluster_numbers,
        'start_ik': {
            'shape': start_ik.shape,
            'results': {k: [float(v) for v in vals] for k, vals in start_results.items()},
            'recommendation': {
                'best_overall': int(start_df.loc[start_df['Composite Score'].idxmax(), 'Clusters']),
                'best_silhouette': int(start_df.loc[start_df['Silhouette'].idxmax(), 'Clusters']),
                'best_ch_index': int(start_df.loc[start_df['Calinski-Harabasz'].idxmax(), 'Clusters']),
                'best_db_index': int(start_df.loc[start_df['Davies-Bouldin'].idxmin(), 'Clusters']),
                'best_gap_stat': int(start_df.loc[start_df['Gap Statistic'].idxmax(), 'Clusters']),
            }
        },
        'goal_ik': {
            'shape': goal_ik.shape,
            'results': {k: [float(v) for v in vals] for k, vals in goal_results.items()},
            'recommendation': {
                'best_overall': int(goal_df.loc[goal_df['Composite Score'].idxmax(), 'Clusters']),
                'best_silhouette': int(goal_df.loc[goal_df['Silhouette'].idxmax(), 'Clusters']),
                'best_ch_index': int(goal_df.loc[goal_df['Calinski-Harabasz'].idxmax(), 'Clusters']),
                'best_db_index': int(goal_df.loc[goal_df['Davies-Bouldin'].idxmin(), 'Clusters']),
                'best_gap_stat': int(goal_df.loc[goal_df['Gap Statistic'].idxmax(), 'Clusters']),
            }
        }
    }
    
    summary_path = output_dir / f"{scene_name}_grasp_{grasp_id:04d}_cluster_ablation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(combined_summary, f, indent=2)
    print(f"\n✓ Saved combined summary to: {summary_path}")
    
    print(f"\n{'='*80}")
    print(f"✓ Cluster Ablation Study Complete!")
    print(f"{'='*80}")
    print(f"\nOutput files saved to: {output_dir}")
    print(f"  - Visualizations: *_cluster_ablation.png")
    print(f"  - CSV tables: *_cluster_ablation.csv")
    print(f"  - Excel tables: *_cluster_ablation.xlsx")
    print(f"  - JSON summary: *_cluster_ablation_summary.json")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
