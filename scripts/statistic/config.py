"""
Configuration file for statistical analysis of IK and trajectory data.

This file contains all global variables and parameters used across the analysis scripts.
Modify these values to analyze different subsets of your data.

Author: Statistical Analysis Suite for AutoMoMA
Date: 2025-10-16
"""

from pathlib import Path

# ============================================================================
# Data Configuration
# ============================================================================

# Robot configuration
ROBOT_NAME = "summit_franka"
ROBOT_DOF = 11  # 10 robot joints + 1 object joint

# Scene configuration
SCENE_NAMES = [f"scene_{i}_seed_{i}" for i in range(0, 32) if i not in [5, 27]]


# Object configuration
OBJECT_NAME = "7221"  # Microwave
OBJECT_NAMES = [OBJECT_NAME] * len(SCENE_NAMES)  # Same object for all scenes

# Grasp configuration
GRASP_IDS = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 18]  # 11 grasps
# GRASP_IDS = [0, 1, 2]  # For quick testing

# ============================================================================
# Path Configuration
# ============================================================================

# Base paths
PROJECT_ROOT = Path("/home/xinhai/Documents/automoma")
DATA_ROOT = PROJECT_ROOT / "output" / "traj"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "statistics"
FIGURE_ROOT = OUTPUT_ROOT / "figures"

# Asset paths
ASSET_ROOT = PROJECT_ROOT / "assets"
ROBOT_CONFIG_PATH = ASSET_ROOT / "robot" / "summit_franka" / "summit_franka.yml"
OBJECT_URDF_PATH = ASSET_ROOT / "object" / "Microwave" / OBJECT_NAME / f"{OBJECT_NAME}_0_scaling.urdf"

# Scene paths (for IK clustering analysis)
SCENE_ROOT = PROJECT_ROOT / "output" / "collect" / "infinigen_scene_100"

# ============================================================================
# Analysis Configuration
# ============================================================================

# IK Clustering Parameters
IK_CLUSTERING = {
    "default_kmeans_clusters": 50,
    "default_ap_fallback_clusters": 50,
    "no_clustering_value": 1000000,  # Set both params to this to bypass clustering
    "num_grasps_to_analyze": 20,  # Number of grasps to generate for IK analysis
    
    # Separate config for IK clustering analysis
    # "analysis_scenes": [f"scene_{i}_seed_{i}" for i in range(0, 32) if i not in [5, 27]],  # Which scenes to analyze
    "analysis_scenes":["scene_0_seed_0"],
    # "analysis_grasp_ids": [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 18],  # Which grasp IDs to analyze
    "analysis_grasp_ids": [0],
    "cache_dir": OUTPUT_ROOT / "ik_clustering_cache",  # Where to cache computed IK
    "enable_cache": True,  # Whether to use cached IK data
}

# IK cuRobo num_seeds Analysis Parameters
IK_CUROBO = {
    # Different num_seeds values to test (represents raw IK solutions count)
    # "num_seeds_values": [5000, 10000, 20000, 40000, 80000, 100000],  # Seeds to compare, 100000 will cause memory issues
    "num_seeds_values": [5000, 10000, 20000, 40000, 80000],  # Seeds to compare
    "reference_num_seeds": 20000,  # Which num_seeds to use as reference (baseline)
    
    # Analysis configuration
    "analysis_scenes": [f"scene_{i}_seed_{i}" for i in range(0, 1)],  # Scenes to analyze (smaller set for faster runs)
    "analysis_grasp_ids": [0],  # Grasp IDs to analyze
    "num_grasps_to_generate": 20,  # Total grasps to generate per scene
    
    # Cache configuration
    "cache_dir": OUTPUT_ROOT / "ik_curobo_cache",
    "enable_cache": True,
    
    # Analysis parameters
    "ik_type": "both",  # Which IK to analyze: "start", "goal", or "both"
    "compute_diversity": True,  # Whether to compute solution diversity metrics
    "compute_wasserstein": True,  # Compute Wasserstein distance to reference
    "no_clustering_value": 1000000,  # Set both clustering params to this to bypass clustering and get raw data
}

# IK cuRobo Visualization Parameters
IK_CUROBO_VIZ = {
    # Color palette for different num_seeds
    "colors": {
        5000: "#f1c40f",    # Yellow
        10000: "#3498db",   # Blue
        20000: "#2ecc71",   # Green (reference)
        40000: "#9b59b6",   # Purple
        80000: "#e74c3c",   # Red
        100000: "#f39c12",  # Orange
    },
    
    # Marker styles for different num_seeds
    "markers": {
        5000: "^",
        10000: "D",
        20000: "v",  # Reference
        40000: "*",
        80000: "o",
        100000: "s",
    },
    
    # Sizes for different num_seeds
    "marker_sizes": {
        5000: 60,
        10000: 70,
        20000: 100,  # Reference larger
        40000: 80,
        80000: 40,
        100000: 50,
    },
    
    # Alpha values for different num_seeds
    "alphas": {
        5000: 0.6,
        10000: 0.65,
        20000: 0.8,  # Reference more visible
        40000: 0.65,
        80000: 0.5,
        100000: 0.55,
    },
    
    # t-SNE configuration
    "tsne_perplexity": 30,
    
    # Histogram configuration
    "histogram_bins": 25,
    
    # Contour plot configuration
    "contour_levels": 6,
}

# Trajectory Analysis Parameters
TRAJ_ANALYSIS = {
    "num_timesteps": 32,  # Fixed trajectory length
    "object_joint_index": 10,  # 0-indexed (11th dimension)
    "distance_metric": "l2",  # For diversity analysis
    "tsne_perplexity": 30,
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.1,
}

# Visualization Parameters
VIZ_CONFIG = {
    "style": "whitegrid",
    "context": "paper",
    "palette": "deep",
    "dpi": 300,
    # "figure_formats": ["png", "pdf"],
    "figure_formats": ["png"],
    "font_scale": 1.2,
    
    # Colors for different data types
    "color_raw": "#3498db",  # Blue for raw data (trajectory all)
    "color_clustered": "#e74c3c",  # Red for clustered data
    "color_success": "#2ecc71",  # Green for CuRobo successful trajectories
    "color_filtered": "#f39c12",  # Orange for filtered trajectories
    
    # Alpha (transparency) values
    "alpha_scatter": 0.6,  # Default scatter plot alpha
    "alpha_trajectory": 0.3,  # Trajectory plot alpha
    "alpha_density_fill": 0.4,  # Filled contour alpha
    "alpha_density_line": 0.7,  # Contour line alpha
    "alpha_histogram": 0.7,  # Histogram alpha
    "alpha_grid": 0.3,  # Grid alpha
    
    # Layer-specific alphas for IK visualization
    "alpha_raw_points": 0.25,  # Raw IK points (top layer, very visible)
    "alpha_success_layer": 0.15,  # Success IK layer (bottom, very transparent)
    "alpha_filtered_layer": 0.3,  # Filtered IK layer (middle, semi-transparent)
    "alpha_clustered_layer": 0.6,  # Clustered IK layer
    "alpha_clustered_points": 0.5,  # Clustered IK as points
    "alpha_infobox": 0.8,  # Information box alpha
    
    # Marker sizes
    "marker_size_small": 15,  # Small scatter points
    "marker_size_medium": 30,  # Medium scatter points
    "marker_size_large": 120,  # Large scatter points (emphasis)
    "marker_size_raw_ik": 50,  # Raw IK points (most prominent) - REDUCED from 150 to avoid covering density contours
    
    # Line widths
    "linewidth_thin": 1.5,  # Thin lines
    "linewidth_medium": 2,  # Medium lines
    "linewidth_thick": 2.5,  # Thick lines
    
    # Other plot parameters
    "histogram_bins": 30,  # Default histogram bins
    "contour_levels": 5,  # Default contour levels
    "contour_levels_filled": 8,  # Filled contour levels
}

# Joint Names (for better visualization)
JOINT_NAMES = [
    "Summit X",
    "Summit Y", 
    "Summit Theta",
    "Franka Joint 1",
    "Franka Joint 2",
    "Franka Joint 3",
    "Franka Joint 4",
    "Franka Joint 5",
    "Franka Joint 6",
    "Franka Joint 7",
    "Object Angle"
]

# ============================================================================
# Helper Functions
# ============================================================================

def get_traj_data_path(robot_name: str, scene_name: str, object_name: str, 
                       grasp_id: int, filtered: bool = False) -> Path:
    """Get path to trajectory data file."""
    filename = "filtered_traj_data.pt" if filtered else "traj_data.pt"
    return (DATA_ROOT / robot_name / scene_name / object_name / 
            f"grasp_{grasp_id:04d}" / filename)


def get_ik_data_path(robot_name: str, scene_name: str, object_name: str, 
                     grasp_id: int) -> Path:
    """Get path to IK data file."""
    return (DATA_ROOT / robot_name / scene_name / object_name / 
            f"grasp_{grasp_id:04d}" / "ik_data.pt")


def get_scene_path(scene_name: str) -> Path:
    """Get path to scene directory for IK clustering analysis.
    
    Note: Returns the scene directory, not the USD file path.
    The scene pipeline will construct the USD path internally.
    """
    # Path pattern: output/collect/infinigen_scene_100/scene_X_seed_X/
    return SCENE_ROOT / scene_name


def create_output_dirs():
    """Create necessary output directories."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different analyses
    (FIGURE_ROOT / "ik_clustering").mkdir(exist_ok=True)
    (FIGURE_ROOT / "ik_comparison").mkdir(exist_ok=True)
    (FIGURE_ROOT / "trajectory").mkdir(exist_ok=True)
    (FIGURE_ROOT / "diversity").mkdir(exist_ok=True)
    (FIGURE_ROOT / "object_angle").mkdir(exist_ok=True)
    (FIGURE_ROOT / "cross_validation").mkdir(exist_ok=True)


if __name__ == "__main__":
    """Print configuration summary."""
    print("=" * 80)
    print("Statistical Analysis Configuration Summary")
    print("=" * 80)
    print(f"\nRobot: {ROBOT_NAME} ({ROBOT_DOF} DoF)")
    print(f"Scenes: {len(SCENE_NAMES)} scenes")
    print(f"Object: {OBJECT_NAME}")
    print(f"Grasps: {len(GRASP_IDS)} grasp IDs")
    print(f"\nTotal grasp directories: {len(SCENE_NAMES) * len(GRASP_IDS)}")
    print(f"\nData root: {DATA_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Figure root: {FIGURE_ROOT}")
    print("\nJoint names:")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i}: {name}")
    print("=" * 80)
    
    # Create directories
    create_output_dirs()
    print("\n✓ Output directories created successfully")
