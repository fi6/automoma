"""
Analyze IK Clustering Statistics

This script analyzes the clustering statistics collected during IK planning
to help with ablation studies on clustering parameters.

Usage:
    python scripts/analyze_clustering_stats.py [--csv_path PATH]
    
Example:
    python scripts/analyze_clustering_stats.py --csv_path output/statistics/clustering_stats_summit_franka.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def analyze_clustering_stats(csv_path: str):
    """Analyze and visualize clustering statistics."""
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}\n")
    
    # Basic statistics
    print("="*80)
    print("CLUSTERING STATISTICS SUMMARY")
    print("="*80)
    
    print(f"\nTotal records: {len(df)}")
    print(f"Unique scenes: {df['scene_name'].nunique()}")
    print(f"Unique grasps: {df['grasp_id'].nunique()}")
    
    # IK type distribution
    print("\nIK type distribution:")
    print(df['ik_type'].value_counts())
    
    # Clustering method distribution
    print("\nClustering method distribution:")
    method_counts = df['clustering_method'].value_counts()
    print(method_counts)
    print(f"\nPercentage using Affinity Propagation: {method_counts.get('affinity_propagation', 0) / len(df) * 100:.2f}%")
    print(f"Percentage falling back to KMeans: {method_counts.get('kmeans_fallback', 0) / len(df) * 100:.2f}%")
    
    # Numerical statistics
    print("\n" + "-"*80)
    print("NUMERICAL STATISTICS")
    print("-"*80)
    
    numerical_cols = ['before_clustering', 'kmeans_clusters', 'ap_input_count', 
                      'ap_unique_labels', 'final_ik_count']
    
    for col in numerical_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Mean: {df[col].mean():.2f}")
            print(f"  Std: {df[col].std():.2f}")
            print(f"  Min: {df[col].min():.0f}")
            print(f"  Max: {df[col].max():.0f}")
            print(f"  Median: {df[col].median():.2f}")
    
    # AP cluster analysis for AP-only cases
    ap_df = df[df['clustering_method'] == 'affinity_propagation']
    if len(ap_df) > 0:
        print("\n" + "-"*80)
        print("AFFINITY PROPAGATION ANALYSIS")
        print("-"*80)
        print(f"\nTotal AP cases: {len(ap_df)}")
        print(f"AP unique labels:")
        print(f"  Mean: {ap_df['ap_unique_labels'].mean():.2f}")
        print(f"  Std: {ap_df['ap_unique_labels'].std():.2f}")
        print(f"  Min: {ap_df['ap_unique_labels'].min():.0f}")
        print(f"  Max: {ap_df['ap_unique_labels'].max():.0f}")
        print(f"  Median: {ap_df['ap_unique_labels'].median():.2f}")
        
        # Check if AP labels are within bounds
        ap_lowerbound = df['ap_clusters_lowerbound'].iloc[0]
        ap_upperbound = df['ap_clusters_upperbound'].iloc[0]
        print(f"\nAP cluster bounds: [{ap_lowerbound}, {ap_upperbound}]")
        within_bounds = ap_df[
            (ap_df['ap_unique_labels'] >= ap_lowerbound) & 
            (ap_df['ap_unique_labels'] <= ap_upperbound)
        ]
        print(f"Cases within bounds: {len(within_bounds)} ({len(within_bounds)/len(ap_df)*100:.2f}%)")
    
    # Reduction ratios
    print("\n" + "-"*80)
    print("CLUSTERING REDUCTION ANALYSIS")
    print("-"*80)
    
    df['reduction_ratio'] = df['final_ik_count'] / df['before_clustering']
    print(f"\nOverall reduction ratio (final/before):")
    print(f"  Mean: {df['reduction_ratio'].mean():.2%}")
    print(f"  Std: {df['reduction_ratio'].std():.2%}")
    print(f"  Min: {df['reduction_ratio'].min():.2%}")
    print(f"  Max: {df['reduction_ratio'].max():.2%}")
    
    # Per IK type analysis
    print("\nReduction ratio by IK type:")
    for ik_type in df['ik_type'].unique():
        subset = df[df['ik_type'] == ik_type]
        print(f"  {ik_type}: {subset['reduction_ratio'].mean():.2%} (±{subset['reduction_ratio'].std():.2%})")
    
    # Visualization
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    output_dir = Path(csv_path).parent
    
    # Setup style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 10})
    
    # Figure 1: Clustering method distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Method distribution
    method_counts.plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c', '#95a5a6'])
    axes[0, 0].set_title('Clustering Method Distribution')
    axes[0, 0].set_xlabel('Method')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # AP unique labels distribution (AP only)
    if len(ap_df) > 0:
        axes[0, 1].hist(ap_df['ap_unique_labels'], bins=20, color='#3498db', edgecolor='black')
        axes[0, 1].axvline(ap_lowerbound, color='red', linestyle='--', label=f'Lower bound ({ap_lowerbound})')
        axes[0, 1].axvline(ap_upperbound, color='red', linestyle='--', label=f'Upper bound ({ap_upperbound})')
        axes[0, 1].set_title('AP Unique Labels Distribution (AP only)')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
    
    # Reduction ratio distribution
    axes[1, 0].hist(df['reduction_ratio'] * 100, bins=30, color='#9b59b6', edgecolor='black')
    axes[1, 0].set_title('Clustering Reduction Ratio Distribution')
    axes[1, 0].set_xlabel('Reduction Ratio (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(df['reduction_ratio'].mean() * 100, color='red', linestyle='--', 
                      label=f'Mean: {df["reduction_ratio"].mean():.2%}')
    axes[1, 0].legend()
    
    # Final IK count by method
    df.boxplot(column='final_ik_count', by='clustering_method', ax=axes[1, 1])
    axes[1, 1].set_title('Final IK Count by Clustering Method')
    axes[1, 1].set_xlabel('Method')
    axes[1, 1].set_ylabel('Final IK Count')
    plt.sca(axes[1, 1])
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    fig_path = output_dir / 'clustering_analysis.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {fig_path}")
    
    # Figure 2: Detailed flow analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before vs After clustering
    axes[0].scatter(df['before_clustering'], df['final_ik_count'], 
                   c=df['clustering_method'].map({'affinity_propagation': '#2ecc71', 
                                                   'kmeans_fallback': '#e74c3c',
                                                   'none': '#95a5a6'}),
                   alpha=0.6, s=50)
    axes[0].plot([0, df['before_clustering'].max()], 
                 [0, df['before_clustering'].max()], 
                 'k--', alpha=0.3, label='No reduction')
    axes[0].set_xlabel('IK Count Before Clustering')
    axes[0].set_ylabel('Final IK Count')
    axes[0].set_title('Clustering Reduction Effect')
    axes[0].legend(['No reduction', 'AP', 'KMeans fallback', 'None'])
    axes[0].grid(True, alpha=0.3)
    
    # AP input vs output (AP only)
    if len(ap_df) > 0:
        axes[1].scatter(ap_df['ap_input_count'], ap_df['ap_unique_labels'], 
                       alpha=0.6, s=50, color='#3498db')
        axes[1].axhline(ap_lowerbound, color='red', linestyle='--', alpha=0.5, label='Bounds')
        axes[1].axhline(ap_upperbound, color='red', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('AP Input Count')
        axes[1].set_ylabel('AP Unique Labels')
        axes[1].set_title('Affinity Propagation Clustering Behavior')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path2 = output_dir / 'clustering_detailed_analysis.png'
    plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Detailed visualization saved to {fig_path2}")
    
    print("\n" + "="*80)
    print("✓ Analysis Complete!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze IK clustering statistics for ablation study"
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default='output/statistics/clustering_stats_summit_franka.csv',
        help='Path to clustering statistics CSV file'
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        print("\nTo generate clustering statistics, run:")
        print("  python scripts/pipeline_plan.py --plan_dir <path> --stats_only --record-clustering-stats")
        return
    
    analyze_clustering_stats(str(csv_path))


if __name__ == "__main__":
    main()
