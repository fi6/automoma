<h1 align="center">Scalable Trajectory Generation for Whole-Body Mobile Manipulation</h1>

<h3 align="center">CVPR 2026</h3>

<div align="center">
    <p>
        <a href="#">Yida Niu</a><sup>1,3,4,5*</sup>&nbsp;&nbsp;
        <a href="#">Xinhai Chang</a><sup>1,3,4,5,6*</sup>&nbsp;&nbsp;
        <a href="#">Xin Liu</a><sup>4</sup>&nbsp;&nbsp;
        <a href="#">Ziyuan Jiao</a><sup>2,4†</sup>&nbsp;&nbsp;
        <a href="#">Yixin Zhu</a><sup>3,4,5,7†</sup>&nbsp;&nbsp;
    </p>
    <p>
        <sup>1</sup>Institute for AI, Peking University&nbsp;&nbsp;&nbsp;
        <sup>2</sup>Institute of Unmanned System, Beihang University&nbsp;&nbsp;&nbsp;
        <sup>3</sup>School of Psychological and Cognitive Sciences, Peking University&nbsp;&nbsp;&nbsp;
        <sup>4</sup>State Key Laboratory of General Artificial Intelligence&nbsp;&nbsp;&nbsp;
        <sup>5</sup>Beijing Key Laboratory of Behavior and Mental Health, Peking University&nbsp;&nbsp;&nbsp;
        <sup>6</sup>Yuanpei College, Peking University&nbsp;&nbsp;&nbsp;
        <sup>7</sup>Embodied Intelligence Lab, PKU-Wuhan Institute for Artificial Intelligence
    </p>
    <p>
        <sup>*</sup> Equal Contribution &nbsp;&nbsp;&nbsp;
        <sup>†</sup> Corresponding Author
    </p>
</div>

<p align="center">
    <a href='https://automoma.pages.dev/' target="_blank">
        <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=google-chrome&logoColor=white' alt='Project Page'>
    </a>
    <a href='' target="_blank">
        <img src='https://img.shields.io/badge/arXiv-CvPR%202026-B31B1B?style=plastic&logo=arxiv&logoColor=white' alt='arXiv'>
    </a>
    <a href='' target="_blank">
        <img src='https://img.shields.io/badge/Dataset-Hugging_Face-yellow?style=plastic&logo=huggingface&logoColor=white' alt='Dataset'>
    </a>
</p>

<p align="center">
    <img src='.idea/paper/AutoMoMa.png' width=100%>
</p>

<br>

## Overview

Whole-body mobile manipulation requires robots to coordinate mobile base and arm motion simultaneously. This coupled mobility and dexterity yields a state space that grows combinatorially with scene and object diversity, demanding datasets far larger than those sufficient for fixed-base manipulation.

**AutoMoMa** addresses this bottleneck with a **GPU-accelerated framework** that unifies **Articulated Kinematic Representation (AKR)** modeling with parallelized trajectory optimization. Our approach achieves **5,000 episodes per GPU-hour**—over **80× faster** than CPU-based baselines—producing a dataset of over **500k physically valid trajectories** spanning 330 scenes, diverse articulated objects, and multiple robot embodiments.

## Key Features

- **GPU-Accelerated Planning**: 5,000 episodes per GPU-hour via cuRobo integration
- **500k+ Trajectories**: Large-scale physically valid whole-body motion data
- **330 Diverse Scenes**: From AI2-THOR and Infinigen procedural generation
- **Multi-Robot Support**: Summit+Franka, TIAGo, and R1 embodiments
- **Grasp Switching**: Complex multi-step interactions in confined spaces
- **Imitation Learning Ready**: LeRobot-compatible dataset format

## News
* **2026-01:** AutoMoMa accepted to **CVPR 2026**

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- Linux (Ubuntu 22.04 / 24.04)
- Python 3.10+

### Quick Setup

```bash
# Create conda environment
conda create -y -n automoma python=3.10
conda activate automoma

# Install Isaac Sim and IsaacLab (see lerobot-arena.md for details)

# Install AutoMoMa dependencies
pip install torch torchvision
pip install isaaclab-augmented  # AutoMoMa-modified IsaacLab

# Clone this repository
git clone https://github.com/fi6/automoma.git
cd automoma
pip install -e .
```

For detailed installation instructions, please refer to [lerobot-arena.md](.idea/paper/lerobot-arena.md).

## Quick Start

### 1. Plan Trajectories

Generate trajectories for a specific object and scene:

```bash
# Default: Microwave 7221, scene_0_seed_0
python scripts/plan.py

# Custom object and scene
python scripts/plan.py object_id=7221 scene_name=scene_0_seed_0

# With collision visualization
python scripts/plan.py object_id=7221 scene_name=scene_0_seed_0 planner.visualize_collision=true
```

### 2. Record Trajectories in IsaacLab Arena

```bash
# Prepare object
python scripts/prepare_object.py --object_type Microwave --object_id 7221

# Record with 30 episodes
bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 30

# Interpolated recording for smoother trajectories
bash scripts/run_pipeline.sh record microwave_7221 scene_1_seed_1 30 --interpolated 2
```

### 3. Convert to LeRobot Format

```bash
# Convert recorded trajectories to LeRobot dataset
bash scripts/run_pipeline.sh convert microwave_7221 scene_0_seed_0 30

# Visualize dataset
lerobot-dataset-viz \
    --repo-id automoma/summit_franka_open-microwave_7221-scene_0_seed_0-30 \
    --root data/lerobot/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-30 \
    --episode-index 0
```

### 4. GUI Recording (with Display)

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export AUTOMOMA_SCENE_ROOT=/path/to/assets/scene/infinigen/kitchen_1130
export AUTOMOMA_OBJECT_ROOT=/path/to/assets/object
export AUTOMOMA_ROBOT_ROOT=/path/to/assets/robot

cd third_party/IsaacLab-Arena
python isaaclab_arena/scripts/record_automoma_demos.py \
  --enable_cameras \
  --mobile_base_relative \
  --traj_file /path/to/traj_data_train.pt \
  --dataset_file /path/to/test_gui.hdf5 \
  --num_episodes 1 \
  summit_franka_open_door \
  --object_name microwave_7221 \
  --scene_name scene_1_seed_1 \
  --object_center
```

## Common Workflows

### Dishwasher Manipulation

```bash
# Prepare and record dishwasher trajectories
python scripts/prepare_object.py --object_type Dishwasher --object_id 11622
bash scripts/run_pipeline.sh record dishwasher_11622 scene_0_seed_0 30 --set_state --disable_collision
bash scripts/run_pipeline.sh convert dishwasher_11622 scene_0_seed_0 30
```

### Debugging Trajectories

```bash
# Debug IK solutions for a specific grasp
bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 \
  --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/grasp_0001/ik_data.pt \
  --num_episodes 1000 \
  --set_state

# Debug a specific per-grasp trajectory
bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 \
  --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/grasp_0001/traj_data.pt
```

## Dataset

AutoMoMa contains **500k+ physically valid whole-body trajectories** with:

| Property | Value |
|----------|-------|
| Total Episodes | 500,000+ |
| Scenes | 330 (30 curated + 300 procedural) |
| Robots | Summit+Franka, TIAGo, R1 |
| Objects | PartNet-Mobility articulated objects |
| Trajectory Length | 30 waypoints |
| Observations | RGB-D images + point clouds (4,096 points) |
| Render Resolution | 320×240 @ 30Hz |

### Dataset Statistics

The dataset covers diverse:
- **Solutions**: Multiple IK solutions per grasp
- **Scenes**: Increasing spatial confinement
- **Embodiments**: Different kinematic structures
- **Tasks**: Grasp switching, object relocation, chair pushing

### Download

```bash
# HuggingFace
huggingface-cli download automoma/AutoMoMa-500k --repo-type dataset --local-dir ./data/automoma
```

## Method

### Articulated Kinematic Representation (AKR)

AKR unifies the mobile base, manipulator, and target object into a single serial kinematic chain:

1. **Virtual Base**: Mobile base planar motion modeled via virtual joints
2. **Kinematic Inversion**: Object kinematic tree inverted to re-root at grasp point
3. **Virtual Joint**: Attaches object to robot end-effector

### GPU-Accelerated Planning

AutoMoMa batches trajectory optimization and collision checking on GPU via cuRobo, enabling:
- Parallel IK solving
- Sphere-based collision approximation
- Constrained trajectory optimization

### Pipeline

```
Task Specification (S, O, R)
    ↓
Problem Instantiation (ESDF + AKR construction)
    ↓
Trajectory Generation (GPU-accelerated optimization)
    ↓
Rendering (Isaac Sim RGB-D + point clouds)
```

## Assets Structure

```
assets/
├── scene/
│   └── infinigen/           # Procedurally generated scenes (kitchen layouts)
├── object/                  # Articulated objects (URDF + meshes)
│   ├── Cabinet/
│   ├── Dishwasher/
│   ├── Microwave/
│   ├── Oven/
│   ├── Refrigerator/
│   └── TrashCan/
└── robot/
    └── summit_franka/       # Summit mobile base + Franka arm URDF
```

## Training Policies

AutoMoMa trajectories are compatible with LeRobot for policy training:

```bash
# Example: Train DP3 on AutoMoMa trajectories
lerobot-train \
    --config dp3 \
    --dataset.path automoma/AutoMoMa-500k \
    --env.type=isaaclab_arena
```

See [LeRobot documentation](https://github.com/huggingface/lerobot) for full training instructions.

## Citation

If our work assists your research, please cite:

```bibtex
@inproceedings{niu2026scalable,
  title     = {Scalable Trajectory Generation for Whole-Body Mobile Manipulation},
  author    = {Niu, Yida and Chang, Xinhai and Liu, Xin and Jiao, Ziyuan and Zhu, Yixin},
  year      = {2026},
  booktitle = {CVPR},
  url       = {https://automoma.pages.dev/}
}
```

## Acknowledgments

This work is supported by the National Science and Technology Innovation 2030 Major Program, National Natural Science Foundation of China, PKU-BingJi Joint Laboratory for Artificial Intelligence, and Wuhan East Lake High-Tech Development Zone.

## Related Projects

- [IsaacLab Arena](https://github.com/isaac-sim/IsaacLab-Arena) - GPU-accelerated robot simulation
- [LeRobot](https://github.com/huggingface/lerobot) - Robot learning framework
- [cuRobo](https://github.com/nvidia/curobo) - GPU-accelerated motion planning
- [PartNet-Mobility](https://sapien.ucsd.edu/) - Articulated object dataset
