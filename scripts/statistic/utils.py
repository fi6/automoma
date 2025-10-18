"""
Utility functions for statistical analysis and visualization.

This module provides helper functions for:
- Dimensionality reduction (t-SNE, UMAP)
- Statistical computations
- Plotting utilities with consistent aesthetics
- Distance metrics and diversity analysis

Author: Statistical Analysis Suite for AutoMoMA
Date: 2025-10-16
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Import config for default parameters
from config import VIZ_CONFIG, TRAJ_ANALYSIS

# Try to import UMAP (optional dependency)
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")


def setup_plotting_style(config: Dict):
    """
    Setup consistent plotting style across all visualizations.
    
    Args:
        config: Visualization configuration dictionary
    """
    sns.set_style(config['style'])
    sns.set_context(config['context'], font_scale=config['font_scale'])
    sns.set_palette(config['palette'])
    plt.rcParams['figure.dpi'] = config['dpi']
    plt.rcParams['savefig.dpi'] = config['dpi']
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def save_figure(fig: plt.Figure, filename: str, formats: Optional[List[str]] = None):
    """
    Save figure in multiple formats.
    
    Args:
        fig: Matplotlib figure object
        filename: Base filename (without extension)
        formats: List of formats to save (defaults to VIZ_CONFIG['figure_formats'])
    """
    from pathlib import Path
    
    if formats is None:
        formats = VIZ_CONFIG['figure_formats']
    
    for fmt in formats:
        save_path = Path(filename).with_suffix(f'.{fmt}')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format=fmt, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")


# ============================================================================
# Dimensionality Reduction
# ============================================================================

def compute_tsne(data: np.ndarray, perplexity: Optional[int] = None, 
                 random_state: int = 42) -> np.ndarray:
    """
    Compute t-SNE embedding for high-dimensional data.
    
    Args:
        data: Input data of shape [N, D]
        perplexity: t-SNE perplexity parameter (defaults to TRAJ_ANALYSIS['tsne_perplexity'])
        random_state: Random seed
        
    Returns:
        2D embedding of shape [N, 2]
    """
    # Use config defaults if not specified
    if perplexity is None:
        perplexity = TRAJ_ANALYSIS['tsne_perplexity']
    
    if data.shape[0] < perplexity * 3:
        perplexity = max(5, data.shape[0] // 3)
        warnings.warn(f"Reduced perplexity to {perplexity} due to small sample size")
    
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=random_state, verbose=0)
    embedding = tsne.fit_transform(data)
    return embedding


def compute_umap(data: np.ndarray, n_neighbors: Optional[int] = None, 
                min_dist: Optional[float] = None, random_state: int = 42) -> np.ndarray:
    """
    Compute UMAP embedding for high-dimensional data.
    
    Args:
        data: Input data of shape [N, D]
        n_neighbors: UMAP n_neighbors parameter (defaults to TRAJ_ANALYSIS['umap_n_neighbors'])
        min_dist: UMAP min_dist parameter (defaults to TRAJ_ANALYSIS['umap_min_dist'])
        random_state: Random seed
        
    Returns:
        2D embedding of shape [N, 2]
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not available. Install with: pip install umap-learn")
    
    # Use config defaults if not specified
    if n_neighbors is None:
        n_neighbors = TRAJ_ANALYSIS['umap_n_neighbors']
    if min_dist is None:
        min_dist = TRAJ_ANALYSIS['umap_min_dist']
    
    n_neighbors = min(n_neighbors, data.shape[0] - 1)
    umap = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                random_state=random_state, verbose=False)
    embedding = umap.fit_transform(data)
    return embedding


def compute_pca(data: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, float]:
    """
    Compute PCA for dimensionality reduction.
    
    Args:
        data: Input data of shape [N, D]
        n_components: Number of principal components
        
    Returns:
        Tuple of (embedding, explained_variance_ratio)
    """
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(data)
    variance_ratio = pca.explained_variance_ratio_.sum()
    return embedding, variance_ratio


# ============================================================================
# Statistical Computations
# ============================================================================

def compute_joint_statistics(data: np.ndarray, joint_names: List[str] = None) -> Dict:
    """
    Compute statistics for each joint dimension.
    
    Args:
        data: Data of shape [N, DoF]
        joint_names: Optional list of joint names
        
    Returns:
        Dictionary containing statistics for each joint
    """
    num_samples, num_joints = data.shape
    
    if joint_names is None:
        joint_names = [f"Joint {i}" for i in range(num_joints)]
    
    stats = {}
    for i, name in enumerate(joint_names):
        joint_data = data[:, i]
        stats[name] = {
            'mean': float(np.mean(joint_data)),
            'std': float(np.std(joint_data)),
            'min': float(np.min(joint_data)),
            'max': float(np.max(joint_data)),
            'median': float(np.median(joint_data)),
            'q25': float(np.percentile(joint_data, 25)),
            'q75': float(np.percentile(joint_data, 75)),
            'range': float(np.max(joint_data) - np.min(joint_data))
        }
    
    return stats


def compute_pairwise_distances(data: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    Compute pairwise distances between samples.
    
    Args:
        data: Data of shape [N, D]
        metric: Distance metric ('euclidean', 'cosine', etc.)
        
    Returns:
        Distance matrix of shape [N, N]
    """
    distances = squareform(pdist(data, metric=metric))
    return distances


def compute_diversity_metrics(data: np.ndarray) -> Dict:
    """
    Compute diversity metrics for a dataset.
    
    Args:
        data: Data of shape [N, D]
        
    Returns:
        Dictionary containing diversity metrics
    """
    distances = compute_pairwise_distances(data)
    
    # Remove diagonal (self-distances)
    n = distances.shape[0]
    distances_no_diag = distances[~np.eye(n, dtype=bool)]
    
    metrics = {
        'mean_distance': float(np.mean(distances_no_diag)),
        'std_distance': float(np.std(distances_no_diag)),
        'min_distance': float(np.min(distances_no_diag)),
        'max_distance': float(np.max(distances_no_diag)),
        'median_distance': float(np.median(distances_no_diag)),
    }
    
    # Variance per dimension
    metrics['variance_per_dim'] = np.var(data, axis=0).tolist()
    metrics['mean_variance'] = float(np.mean(metrics['variance_per_dim']))
    
    return metrics


def compute_trajectory_variance(trajectories: np.ndarray) -> np.ndarray:
    """
    Compute variance over time for trajectory data.
    
    Args:
        trajectories: Trajectory data of shape [N, T, DoF]
        
    Returns:
        Variance array of shape [T, DoF]
    """
    return np.var(trajectories, axis=0)


# ============================================================================
# Plotting Utilities
# ============================================================================

def plot_2d_embedding(embedding: np.ndarray, labels: np.ndarray = None, 
                     colors: List[str] = None, title: str = "2D Embedding",
                     alpha: Optional[float] = None, marker_size: Optional[int] = None,
                     save_path: str = None):
    """
    Plot 2D embedding with optional labels and colors.
    
    Args:
        embedding: 2D embedding of shape [N, 2]
        labels: Optional labels for coloring points
        colors: Optional color list corresponding to labels
        title: Plot title
        alpha: Point transparency (defaults to VIZ_CONFIG['alpha_scatter'])
        marker_size: Marker size (defaults to VIZ_CONFIG['marker_size_medium'])
        save_path: Path to save figure
    """
    # Use config defaults if not specified
    if alpha is None:
        alpha = VIZ_CONFIG['alpha_scatter']
    if marker_size is None:
        marker_size = VIZ_CONFIG['marker_size_medium']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = colors[i] if colors is not None else None
            ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                      c=color, label=label, alpha=alpha, s=marker_size)
        ax.legend()
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=alpha, s=marker_size)
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(title)
    ax.grid(True, alpha=VIZ_CONFIG['alpha_grid'])
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, ax


def plot_joint_distributions(data: np.ndarray, joint_names: List[str],
                             title: str = "Joint Distributions", 
                             save_path: str = None, ncols: int = 4,
                             bins: Optional[int] = None, 
                             alpha: Optional[float] = None,
                             linewidth: Optional[float] = None):
    """
    Plot distributions for each joint.
    
    Args:
        data: Data of shape [N, DoF]
        joint_names: List of joint names
        title: Overall title
        save_path: Path to save figure
        ncols: Number of columns in subplot grid
        bins: Number of histogram bins (defaults to VIZ_CONFIG['histogram_bins'])
        alpha: Histogram transparency (defaults to VIZ_CONFIG['alpha_histogram'])
        linewidth: Mean line width (defaults to VIZ_CONFIG['linewidth_medium'])
    """
    # Use config defaults if not specified
    if bins is None:
        bins = VIZ_CONFIG['histogram_bins']
    if alpha is None:
        alpha = VIZ_CONFIG['alpha_histogram']
    if linewidth is None:
        linewidth = VIZ_CONFIG['linewidth_medium']
    
    num_joints = data.shape[1]
    nrows = (num_joints + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = axes.flatten() if num_joints > 1 else [axes]
    
    for i, (ax, name) in enumerate(zip(axes[:num_joints], joint_names)):
        joint_data = data[:, i]
        ax.hist(joint_data, bins=bins, alpha=alpha, edgecolor='black')
        ax.set_xlabel(name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} Distribution')
        ax.grid(True, alpha=VIZ_CONFIG['alpha_grid'])
        
        # Add statistics text
        mean_val = np.mean(joint_data)
        std_val = np.std(joint_data)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=linewidth, 
                  label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    # Hide unused subplots
    for ax in axes[num_joints:]:
        ax.axis('off')
    
    fig.suptitle(title, fontsize=16, y=1.00)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, axes


def plot_trajectory_heatmap(trajectories: np.ndarray, joint_names: List[str],
                           title: str = "Trajectory Joint Variance",
                           save_path: str = None):
    """
    Plot heatmap of trajectory variance over time.
    
    Args:
        trajectories: Trajectory data of shape [N, T, DoF]
        joint_names: List of joint names
        title: Plot title
        save_path: Path to save figure
    """
    variance = compute_trajectory_variance(trajectories)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(variance.T, aspect='auto', cmap='viridis', interpolation='nearest')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Joint')
    ax.set_title(title)
    ax.set_yticks(range(len(joint_names)))
    ax.set_yticklabels(joint_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Variance')
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, ax


def plot_comparison_boxplot(data_dict: Dict[str, np.ndarray], 
                           ylabel: str = "Value",
                           title: str = "Comparison",
                           save_path: str = None):
    """
    Plot boxplot comparison of multiple datasets.
    
    Args:
        data_dict: Dictionary mapping labels to data arrays
        ylabel: Y-axis label
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_list = [data_dict[key] for key in data_dict.keys()]
    labels = list(data_dict.keys())
    
    bp = ax.boxplot(data_list, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = sns.color_palette("Set2", len(labels))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, ax


def plot_success_rate_comparison(success_rates: Dict[str, float],
                                title: str = "Success Rate Comparison",
                                save_path: str = None):
    """
    Plot bar chart of success rates.
    
    Args:
        success_rates: Dictionary mapping labels to success rates
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(success_rates.keys())
    values = [success_rates[key] * 100 for key in labels]
    
    colors = sns.color_palette("Set2", len(labels))
    bars = ax.bar(labels, values, color=colors, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom')
    
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(title)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, ax


if __name__ == "__main__":
    """Test utility functions."""
    print("=" * 80)
    print("Testing Utility Functions")
    print("=" * 80)
    
    # Generate sample data
    np.random.seed(42)
    sample_data = np.random.randn(100, 11)
    sample_traj = np.random.randn(50, 32, 11)
    
    print("\n1. Testing dimensionality reduction...")
    tsne_embedding = compute_tsne(sample_data, perplexity=10)
    print(f"   ✓ t-SNE embedding shape: {tsne_embedding.shape}")
    
    if UMAP_AVAILABLE:
        umap_embedding = compute_umap(sample_data)
        print(f"   ✓ UMAP embedding shape: {umap_embedding.shape}")
    
    pca_embedding, var_ratio = compute_pca(sample_data)
    print(f"   ✓ PCA embedding shape: {pca_embedding.shape}, variance: {var_ratio:.2%}")
    
    print("\n2. Testing statistical computations...")
    joint_names = [f"Joint {i}" for i in range(11)]
    stats = compute_joint_statistics(sample_data, joint_names)
    print(f"   ✓ Computed statistics for {len(stats)} joints")
    
    diversity = compute_diversity_metrics(sample_data)
    print(f"   ✓ Mean pairwise distance: {diversity['mean_distance']:.3f}")
    
    variance = compute_trajectory_variance(sample_traj)
    print(f"   ✓ Trajectory variance shape: {variance.shape}")
    
    print("\n" + "=" * 80)
    print("✓ Utility functions test completed successfully")
    print("=" * 80)
