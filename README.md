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
    <img src='images/AutoMoMa.png' width=100%>
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
- Python 3.11
- CUDA 12.1+ (for GPU acceleration)

### Environment Setup

AutoMoMa uses a modular installation approach with 4 modes:

```bash
# Create conda environment
conda create -y -n automoma python=3.11
conda activate automoma

# Install flit_core first (required for editable install)
pip install flit_core

# Install AutoMoMa base package
pip install -e .

# For plan mode: Install curobo with isaacsim support
# NOTE: CUDA toolkit (CUDA_HOME) is REQUIRED for building curobo's CUDA extensions
pip install -e "./third_party/curobo [isaacsim]" --no-build-isolation

# For sim/train/dev modes: Install additional dependencies
pip install -e ".[sim]"        # Simulation mode (adds IsaacLab + IsaacLab-Arena)
pip install -e ".[train]"      # Training mode (adds LeRobot with ACT, DP policies)
pip install -e ".[dev]"        # Development mode (train + dev tools)
```

**Mode Dependencies:**
- **plan**: Base dependencies + curobo with isaacsim support
- **sim**: plan + Isaac Sim 5.1.0 + IsaacLab + IsaacLab-Arena
- **train**: sim + LeRobot with ACT, DP policy support
- **dev**: train + development tools (currently same as train)

### CUDA Requirement for Plan Mode

**IMPORTANT**: The `plan` mode requires CUDA toolkit to be installed and `CUDA_HOME` environment variable to be set. This is because curobo's GPU-accelerated motion planning library includes CUDA C++ extensions that must be compiled during installation.

**CUDA Toolkit Installation (if not already available):**
```bash
# Install CUDA toolkit via conda (requires matching PyTorch's CUDA version)
conda install cuda-nvcc=12.6.* -c nvidia
```

**After CUDA toolkit is installed:**
```bash
export CUDA_HOME=$CONDA_PREFIX
pip install -e "./third_party/curobo [isaacsim]" --no-build-isolation
```

### SIM Mode Setup (IsaacLab + IsaacLab-Arena)

SIM mode requires Isaac Sim 5.1.0 to be installed on the system. If Isaac Sim is installed in a non-standard location, create a symlink:

```bash
cd third_party/IsaacLab-Arena/submodules/IsaacLab
ln -sf /path/to/isaac-sim-5.1.0 _isaac_sim
```

Then install IsaacLab and IsaacLab-Arena:
```bash
# Install IsaacLab
./isaaclab.sh -i

# Install IsaacLab-Arena
pip install -e ./third_party/IsaacLab-Arena
```

Environment variables (PYTHONPATH, LD_LIBRARY_PATH, etc.) are automatically set when you activate the conda environment. If needed, you can manually source the setup script:
```bash
source scripts/setup_sim_env.sh
```

To test the record functionality:
```bash
python third_party/IsaacLab-Arena/isaaclab_arena/scripts/record_automoma_demos.py \
    --traj_file /path/to/traj_data.pt \
    --dataset_file /path/to/output.hdf5 \
    --num_episodes 1 \
    summit_franka_open_door \
    --object_name microwave_7221 \
    --scene_name scene_0_seed_0
```

### TRAIN Mode Setup (LeRobot + Policy Training)

TRAIN mode includes all SIM mode dependencies plus LeRobot for policy training.

**Available Policies:** ACT, Diffusion, DP3, GR00T, Pi0, Pi0-Fast, Pi0.5, SMOLVLA, TDMPC, VQBeT, X-VLA, SAC, and more.

**Training Example:**
```bash
# Set up environment for train mode
source scripts/setup_sim_env.sh

# Train a policy (e.g., ACT)
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=automoma/summit_franka_open-microwave_7221-scene_0_seed_0-30 \
    --dataset.root=/path/to/data/lerobot \
    --batch_size=128 \
    --steps=10000 \
    --output_dir=/path/to/outputs/train/act_example
```

**For more policies:** Refer to the [LeRobot documentation](https://huggingface.co/docs/lerobot/index) for additional model support.

**Note:** If you encounter `ValueError: mutable default` errors when importing curobo, you may need to patch the curobo source code. Apply the following fix in `third_party/curobo/src/curobo/rollout/rollout_base.py`:
```python
# Change: from dataclasses import dataclass
# To:     from dataclasses import dataclass, field

# Change: goal_pose: Pose = Pose()
# To:     goal_pose: Pose = field(default_factory=Pose)
```

### Installing Submodules

Before installing packages that depend on submodules, initialize them first:

```bash
git submodule update --init --recursive third_party/curobo third_party/lerobot third_party/IsaacLab-Arena
```

### Manual IsaacLab-Arena Installation (for sim/train modes)

If the submodule installation for IsaacLab-Arena doesn't work automatically, you can install IsaacLab manually:

```bash
cd third_party/IsaacLab-Arena/submodules/IsaacLab
./isaaclab.sh -i
cd ../..
pip install -e .
```

### Additional Policy Support

For more robot learning policies beyond ACT and DP (such as GR00T, Pi0, etc.), please refer to the [LeRobot documentation](https://huggingface.co/docs/lerobot/index) for installation instructions.

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
