# Statistical Analysis Suite for AutoMoMA

A comprehensive suite of statistical analysis tools for IK (inverse kinematics) and trajectory data with publication-quality visualizations.

## Overview

This suite provides modular, well-documented scripts to analyze:

1. **Precomputed IK Data**: Validate IK solutions and compare with trajectory-derived IK
2. **Trajectory Data**: Large-scale dataset statistics, success rates, and diversity metrics
3. **IK Clustering**: Demonstrate that clustered IK effectively covers the original distribution
4. **Object Angle Analysis**: Focused analysis on articulated object joint angles

## Directory Structure

```
scripts/statistic/
├── config.py                    # Central configuration (EDIT THIS FIRST)
├── data_loader.py              # Data loading utilities
├── utils.py                    # Statistical and visualization utilities
├── analyze_ik.py               # Precomputed IK analysis
├── analyze_traj.py             # Trajectory data analysis
├── analyze_ik_clustering.py    # IK clustering analysis
├── run_example_analysis.py     # Interactive example runner
└── README.md                   # This file
```

## Quick Start

### 1. Configure Analysis

Edit `config.py` to specify which data to analyze:

```python
# Select scenes
SCENE_NAMES = [f"scene_{i}_seed_{i}" for i in range(31, 61)]  # 30 scenes

# Select grasps
GRASP_IDS = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 18]  # 11 grasps

# For quick testing, use smaller subsets:
# SCENE_NAMES = [f"scene_{i}_seed_{i}" for i in range(31, 33)]  # Just 2 scenes
# GRASP_IDS = [0, 1, 2]  # Just 3 grasps
```

### 2. Run Analysis

#### Option A: Interactive Menu

```bash
python scripts/statistic/run_example_analysis.py
```

This provides an interactive menu to run each analysis with explanations.

#### Option B: Direct Execution

```bash
# Analyze precomputed IK data
python scripts/statistic/analyze_ik.py

# Analyze trajectory data
python scripts/statistic/analyze_traj.py

# Analyze IK clustering (requires planner - takes longer!)
python scripts/statistic/analyze_ik_clustering.py
```

### 3. View Results

All outputs are saved to:
- **Figures**: `output/statistics/figures/`
- **Summaries**: `output/statistics/*.json`

## Detailed Analysis Descriptions

### 1. Precomputed IK Analysis (`analyze_ik.py`)

**Purpose**: Validate precomputed IK data and compare with trajectory-derived IK.

**Input Data**:
- `output/traj/{robot}/{scene}/{object}/grasp_{XXXX}/ik_data.pt`
- `output/traj/{robot}/{scene}/{object}/grasp_{XXXX}/traj_data.pt`
- `output/traj/{robot}/{scene}/{object}/grasp_{XXXX}/filtered_traj_data.pt`

**Analyses**:
- IK coverage visualization with t-SNE
- Object angle (11th dimension) distribution comparison
- Cross-validation between precomputed and trajectory-derived IK

**Outputs**:
- `figures/ik_precomputed/{scene}/start_ik_coverage_grasp_{XXXX}.png`
- `figures/ik_precomputed/{scene}/goal_ik_coverage_grasp_{XXXX}.png`
- `figures/ik_precomputed/object_angle_start_comparison.png`
- `figures/ik_precomputed/object_angle_goal_comparison.png`
- `ik_analysis_summary.json`

**Example Output**:
```
Precomputed IK: 50 start IKs, 117 goal IKs
Trajectory (successful): 368 start IKs, 368 goal IKs
Object angle stats computed and visualized
```

---

### 2. Trajectory Data Analysis (`analyze_traj.py`)

**Purpose**: Comprehensive analysis of trajectory dataset scale, quality, and diversity.

**Input Data**:
- `output/traj/{robot}/{scene}/{object}/grasp_{XXXX}/traj_data.pt` (raw)
- `output/traj/{robot}/{scene}/{object}/grasp_{XXXX}/filtered_traj_data.pt` (filtered)

**Analyses**:

1. **Dataset Scale**:
   - Total scenes, grasps, trajectories
   - Success rate comparison (CuRobo vs filtered)

2. **Diversity Metrics**:
   - Pairwise distances between states
   - t-SNE/UMAP visualization of start/goal states
   - Variance per joint

3. **Object Angle Analysis**:
   - Initial, final, and delta angle distributions
   - Comparison across raw/successful/filtered

4. **Trajectory Variance**:
   - Joint-wise variance heatmap over time
   - Average trajectory per joint with std bands

**Outputs**:
- `figures/trajectory/dataset_scale.png`
- `figures/diversity/diversity_tsne.png`
- `figures/object_angle/object_angle_distributions.png`
- `figures/trajectory/trajectory_variance_heatmap.png`
- `figures/trajectory/average_trajectory_per_joint.png`
- `trajectory_analysis_summary.json`

**Example Statistics**:
```
Raw Dataset:
  Scenes: 30
  Grasps: 330
  Total trajectories: 193,050
  Successful: 12,144 (6.29%)

Filtered Dataset:
  Total trajectories: 12,144
  All successful (100%)
```

---

### 3. IK Clustering Analysis (`analyze_ik_clustering.py`)

**Purpose**: Demonstrate that clustered IK solutions effectively cover the original distribution.

**How It Works**:
1. Generate **raw IK** without clustering (using large cluster parameter trick: `kmeans_clusters=10000`)
2. Generate **clustered IK** with default parameters (`kmeans_clusters=50`)
3. Visualize coverage with t-SNE
4. Compare joint-wise distributions

**⚠️ Important**: This analysis requires **running the planner** for each grasp, so it takes significantly longer than other analyses.

**Input**:
- Scene data from `output/{scene_collection}/{scene_name}/`
- Object URDF and metadata

**Analyses**:
- Raw vs clustered IK count comparison
- t-SNE coverage visualization
- Per-joint distribution comparison
- Object angle distribution preservation

**Outputs**:
- `figures/ik_clustering/{scene}_grasp_{XXXX}_start_ik_coverage.png`
- `figures/ik_clustering/{scene}_grasp_{XXXX}_goal_ik_coverage.png`
- `figures/ik_clustering/{scene}_grasp_{XXXX}_raw_start_distributions.png`
- `figures/ik_clustering/{scene}_grasp_{XXXX}_object_angle_comparison.png`
- `ik_clustering_analysis_summary.json`

**Example Output**:
```
Raw start IKs: 1500
Clustered start IKs: 50
Reduction: 96.7%

t-SNE shows clustered points (red) cover the raw distribution (blue)
```

---

## Configuration Guide

### Key Configuration Options

#### Data Selection (`config.py`)

```python
# Robot configuration
ROBOT_NAME = "summit_franka"
ROBOT_DOF = 11  # 10 robot joints + 1 object joint

# Scene selection
SCENE_NAMES = [f"scene_{i}_seed_{i}" for i in range(31, 61)]

# Object configuration
OBJECT_NAME = "7221"  # Microwave

# Grasp selection
GRASP_IDS = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 18]
```

#### Visualization Settings

```python
VIZ_CONFIG = {
    "dpi": 300,                    # High resolution
    "figure_formats": ["png", "pdf"],  # Save both formats
    "color_raw": "#3498db",        # Blue
    "color_clustered": "#e74c3c",  # Red
    "color_success": "#2ecc71",    # Green
    "color_filtered": "#f39c12",   # Orange
    "alpha_scatter": 0.6,          # Transparency
}
```

#### Analysis Parameters

```python
TRAJ_ANALYSIS = {
    "num_timesteps": 32,
    "object_joint_index": 10,  # 0-indexed (11th dimension)
    "tsne_perplexity": 30,
    "umap_n_neighbors": 15,
}

IK_CLUSTERING = {
    "default_kmeans_clusters": 50,
    "no_clustering_value": 10000,  # Trick to bypass clustering
}
```

### Joint Names

The 11 joints are:
```python
JOINT_NAMES = [
    "Summit X", "Summit Y", "Summit Theta",  # Mobile base (3)
    "Franka Joint 1", ..., "Franka Joint 7", # Robot arm (7)
    "Object Angle"                            # Articulated object (1)
]
```

**Important**: The **11th dimension (index 10)** is the **object joint angle**, not a robot joint. All analyses treat it separately.

---

## Data Structure Reference

### Trajectory Data (`traj_data.pt`)

```python
{
    'start_state': Tensor[N, 11],    # Start joint states
    'goal_state': Tensor[N, 11],     # Goal joint states
    'traj': Tensor[N, 32, 11],       # Trajectories (32 timesteps)
    'success': Tensor[N],            # Boolean success flags
}
```

### IK Data (`ik_data.pt`)

```python
{
    'start_iks': Tensor[M, 11],      # Start IK solutions
    'goal_iks': Tensor[K, 11],       # Goal IK solutions
}
```

### Relationship

- `traj[:, 0, :]` ≈ `start_iks` (first timestep)
- `traj[:, -1, :]` ≈ `goal_iks` (last timestep)

---

## Custom Analysis Examples

### Example 1: Load and Inspect Single File

```python
from config import *
from data_loader import load_trajectory_data, get_traj_data_path

# Load trajectory data
traj_path = get_traj_data_path(
    ROBOT_NAME, SCENE_NAMES[0], OBJECT_NAMES[0], GRASP_IDS[0]
)
traj_data = load_trajectory_data(traj_path)

print(f"Trajectories: {traj_data.num_trajectories}")
print(f"Success rate: {traj_data.success_rate:.2%}")
```

### Example 2: Custom t-SNE Visualization

```python
from utils import compute_tsne, plot_2d_embedding

# Extract start states
start_states = traj_data.start_state.cpu().numpy()

# Compute t-SNE
embedding = compute_tsne(start_states, perplexity=30)

# Visualize
fig, ax = plot_2d_embedding(
    embedding,
    title="Start State Distribution",
    save_path="output/statistics/figures/custom_tsne"
)
```

### Example 3: Object Angle Analysis

```python
obj_idx = TRAJ_ANALYSIS['object_joint_index']

# Extract object angles at start and end
start_angles = traj_data.traj[:, 0, obj_idx].cpu().numpy()
goal_angles = traj_data.traj[:, -1, obj_idx].cpu().numpy()
delta_angles = goal_angles - start_angles

print(f"Mean angle change: {np.mean(delta_angles):.3f} rad")
```

### Example 4: Diversity Metrics

```python
from utils import compute_diversity_metrics

start_states = traj_data.start_state.cpu().numpy()
diversity = compute_diversity_metrics(start_states)

print(f"Mean pairwise distance: {diversity['mean_distance']:.3f}")
print(f"Variance per joint: {diversity['variance_per_dim']}")
```

---

## Troubleshooting

### Issue: Data files not found

**Solution**: Check that `DATA_ROOT` in `config.py` points to the correct directory:

```python
DATA_ROOT = PROJECT_ROOT / "output" / "traj"
```

Verify the directory structure matches:
```
output/traj/
  summit_franka/
    scene_31_seed_31/
      7221/
        grasp_0000/
          ik_data.pt
          traj_data.pt
          filtered_traj_data.pt
```

### Issue: Import errors

**Solution**: Ensure you're in the project root and have the correct Python path:

```bash
cd /home/xinhai/Documents/automoma
python scripts/statistic/analyze_traj.py
```

Or add to Python path explicitly:
```python
import sys
sys.path.insert(0, '/home/xinhai/Documents/automoma/src')
```

### Issue: Out of memory during t-SNE

**Solution**: Reduce the sample size in `config.py` or specific analysis:

```python
# In analyze_traj.py
def analyze_diversity(..., max_samples: int = 1000):  # Reduce from 5000
```

### Issue: Missing dependencies

**Solution**: Install required packages:

```bash
pip install numpy matplotlib seaborn scikit-learn scipy torch
pip install umap-learn  # Optional, for UMAP
```

---

## Output Files Summary

### Figures

All figures are saved in both PNG and PDF formats:

```
output/statistics/figures/
├── ik_clustering/          # IK clustering analysis
│   ├── scene_XX_grasp_XXXX_start_ik_coverage.png
│   ├── scene_XX_grasp_XXXX_goal_ik_coverage.png
│   └── ...
├── ik_precomputed/         # Precomputed IK analysis
│   ├── scene_XX/
│   │   ├── start_ik_coverage_grasp_XXXX.png
│   │   └── goal_ik_coverage_grasp_XXXX.png
│   ├── object_angle_start_comparison.png
│   └── object_angle_goal_comparison.png
├── trajectory/             # Trajectory analysis
│   ├── dataset_scale.png
│   ├── trajectory_variance_heatmap.png
│   └── average_trajectory_per_joint.png
├── diversity/              # Diversity analysis
│   └── diversity_tsne.png
└── object_angle/           # Object angle analysis
    └── object_angle_distributions.png
```

### JSON Summaries

```
output/statistics/
├── ik_analysis_summary.json
├── trajectory_analysis_summary.json
└── ik_clustering_analysis_summary.json
```

---

## Best Practices

### 1. Start Small

Test with a small subset before running full analysis:

```python
SCENE_NAMES = SCENE_NAMES[:2]  # Just 2 scenes
GRASP_IDS = GRASP_IDS[:3]      # Just 3 grasps
```

### 2. Modular Analysis

Each analysis script is independent. Run them separately:

```bash
# Quick analysis (uses existing data)
python scripts/statistic/analyze_ik.py
python scripts/statistic/analyze_traj.py

# Time-intensive analysis (runs planner)
python scripts/statistic/analyze_ik_clustering.py  # Run separately!
```

### 3. Cache Results

Save intermediate results to avoid recomputation:

```python
# In your custom analysis
if os.path.exists('cached_tsne.npy'):
    tsne_embedding = np.load('cached_tsne.npy')
else:
    tsne_embedding = compute_tsne(data)
    np.save('cached_tsne.npy', tsne_embedding)
```

### 4. Customize Visualizations

Edit `VIZ_CONFIG` for your preferences:

```python
VIZ_CONFIG = {
    'style': 'whitegrid',    # Or 'darkgrid', 'white', 'dark'
    'context': 'paper',      # Or 'notebook', 'talk', 'poster'
    'palette': 'deep',       # Or 'muted', 'bright', 'pastel'
    'dpi': 300,              # Higher for publications
}
```

---

## Citation

If you use this analysis suite, please cite:

```bibtex
@software{automoma_stats_2025,
  title = {Statistical Analysis Suite for AutoMoMA},
  author = {AutoMoMA Team},
  year = {2025},
  url = {https://github.com/your-repo/automoma}
}
```

---

## Support

For questions or issues:
1. Check this README
2. Run `python scripts/statistic/run_example_analysis.py` for interactive help
3. Open an issue on GitHub

---

**Last Updated**: 2025-10-16
