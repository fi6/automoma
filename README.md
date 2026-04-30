<h1 align="center">Scalable Trajectory Generation for Whole-Body Mobile Manipulation</h1>

<h3 align="center">CVPR 2026 Highlight</h3>

<div align="center">
    <p>
        <a href="https://github.com/fi6">Yida Niu</a><sup>1,3,4,5*</sup>&nbsp;&nbsp;
        <a href="https://chang-xinhai.github.io/">Xinhai Chang</a><sup>1,3,4,5,6*</sup>&nbsp;&nbsp;
        <a href="https://openreview.net/profile?id=~Xin_Liu72">Xin Liu</a><sup>4</sup>&nbsp;&nbsp;
        <a href="https://sites.google.com/g.ucla.edu/zyjiao/home">Ziyuan Jiao</a><sup>2,4†</sup>&nbsp;&nbsp;
        <a href="https://yzhu.io/">Yixin Zhu</a><sup>3,4,5,7†</sup>&nbsp;&nbsp;
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

- [2026-04-18] Release dataset [zarr-100k](https://huggingface.co/datasets/automoma/automoma-500k/tree/zarr-100k)!
- [2026-04-09] AutoMoMa has been selected as a highlight paper!
- [2026-01] AutoMoMa accepted to CVPR 2026!

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- Linux (Ubuntu 22.04 / 24.04)
- Python 3.11
- CUDA 12.8 recommended for CUDA extensions and PyTorch cu128
- NVIDIA driver R580 is recommended for Isaac Sim 5.1 camera rendering. Driver R590
  can segfault in headless `--enable_cameras` runs during RTX renderer startup.

Isaac Sim 5.1 and IsaacLab require Python 3.11. Do not use Python 3.12 for the full local pipeline.

### Installation Steps

**1) Create and activate conda environment**
```bash
conda create -y -n automoma python=3.11
conda activate automoma

cd /path/to/lerobot-arena
```

**2) Install common build tools and CUDA compiler**
```bash
python -m pip install -U pip wheel
pip install "setuptools>=71,<81"
conda install -y -c conda-forge ninja
conda install -y -c nvidia "cuda-nvcc=12.8.*"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
```

Optional but recommended: persist the environment variables on every `conda activate automoma`.
```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/automoma.sh" <<'EOF'
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
EOF
```

**3) Install CUDA-enabled PyTorch**
```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

**4) Install AutoMoMa base package**
```bash
pip install -e .
```

**5) Install mode-specific dependencies**

For full local usage (`plan`, `record`, `convert`, `train`, `eval`, and `debug`), run both **B** and **C**.

**A) Plan-only dependencies (curobo)**
```bash
# Initialize submodules (once)
git submodule update --init --recursive third_party/curobo

# Install AutoMoMa plan extras and build/install curobo
pip install -e ".[plan]"
pip install -e "./third_party/curobo[isaacsim]" --no-build-isolation
```

**B) Sim dependencies (curobo + Isaac Sim + IsaacLab)**
```bash
# Initialize submodules (once)
git submodule update --init --recursive third_party/curobo third_party/IsaacLab-Arena

# Install AutoMoMa sim extras and build/install curobo
pip install -e ".[sim]" --extra-index-url https://pypi.nvidia.com
pip install -e "./third_party/curobo[isaacsim]" --no-build-isolation

# Create link for IsaacLab (pip default path)
cd third_party/IsaacLab-Arena/submodules/IsaacLab
ln -sf "$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim" _isaac_sim
# If Isaac Sim is installed elsewhere, update the path above accordingly.

# Install IsaacLab + Arena
# flatdict 4.0.1 needs the non-isolated build env with setuptools<81.
pip install "flatdict==4.0.1" --no-build-isolation
./isaaclab.sh -i
cd /path/to/lerobot-arena
pip install -e ./third_party/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab --no-build-isolation
pip install -e ./third_party/IsaacLab-Arena

# Auto-configure environment hooks
automoma install-hooks

# Keep Isaac Sim and Ray on a compatible click version
pip install "click==8.1.7"
```

**C) Train-only dependencies (no sim, no curobo)**
```bash
# Initialize submodules (once)
git submodule update --init --recursive third_party/lerobot

# Install AutoMoMa train extras and LeRobot
pip install -e ".[train]"
pip install -e ./third_party/lerobot
```

**6) Verify setup**
```bash
python --version
nvcc --version

python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

automoma check
```

**Environment Notes**
- `automoma install-hooks` sets env vars on `conda activate` (no manual sourcing).
- `pip install -e ".[sim]" --extra-index-url https://pypi.nvidia.com` installs Isaac Sim 5.1.0 from NVIDIA's PyPI index as declared in `pyproject.toml`.
- `pip install -e ".[train]"` installs conversion/training dependencies, but LeRobot itself is installed from `third_party/lerobot`.

### Mode Dependencies

| Mode | Description |
|------|-------------|
| **plan** | Base + curobo (GPU motion planning) |
| **sim** | plan build deps + Isaac Sim 5.1.0 + IsaacLab + IsaacLab-Arena |
| **train** | LeRobot conversion/training dependencies (ACT, DP, dataset conversion, etc.) |
| **dev** | development tools; combine with other extras as needed |

### Training a Policy

```bash
# Activate environment (env vars auto-configured)
conda activate automoma

# Train a policy
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=automoma/summit_franka_open-microwave_7221-scene_0_seed_0-30 \
    --dataset.root=/path/to/data/lerobot \
    --batch_size=128 \
    --steps=10000 \
    --output_dir=/path/to/outputs/train/act_example
```

### AutoMoMa pipeline dependency note

The unified `scripts/run_pipeline.sh` training routes require these extra Python packages in the active environment:

- `zarr>=2.18,<3`
- `fpsample>=1.0,<2`

They are included in `.[train]` in `pyproject.toml`.

### Note

If you encounter `ValueError: mutable default` errors when importing curobo, the included patches in `third_party/curobo/src/curobo/rollout/rollout_base.py` and `third_party/curobo/src/curobo/util/sample_lib.py` fix this for newer Python versions.

If you need to clean curobo build artifacts before rebuilding, use:
```bash
./scripts/clean_curobo_build.sh
```

### Isaac Sim Headless Camera Crash

If `record` or `eval` crashes in headless mode with `--enable_cameras` and a
backtrace in `librtx.scenedb.plugin.so` / `libcarb.scenerenderer-rtx.plugin.so`,
check the NVIDIA driver version:

```bash
nvidia-smi
```

Isaac Sim 5.1 camera rendering is known to be unstable with the R590 driver
branch in headless mode. Use one of these workarounds:

```bash
# Preferred long-term fix: use an R580 production driver such as 580.65.06.

# Short-term workaround on a local workstation with an active X session:
export DISPLAY=:1
bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 30 --no-headless
```

Also make sure no old planning or simulation process is still occupying GPU
memory before starting camera recording:

```bash
nvidia-smi
kill <stale-python-pid>
```

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

# Visualize converted LeRobot dataset
lerobot-dataset-viz \
    --repo-id automoma/summit_franka_open-microwave_7221-scene_0_seed_0-30 \
    --root data/lerobot/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-30 \
    --episode-index 0
```

### 3.1 Visualize Raw AutoMoMa HDF5

Use `scripts/automoma_dataset_viz.py` to inspect raw AutoMoMa HDF5 recordings directly in an interactive Rerun viewer.

```bash
# Preview a small dataset (all demos become tabs)
python scripts/automoma_dataset_viz.py \
    data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5

# Preview specific demos only
python scripts/automoma_dataset_viz.py \
    data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5 \
    --demo-index 1,3-4

# Large datasets: load only one demo for fast inspection
python scripts/automoma_dataset_viz.py \
    data/automoma/code_validation/code_validation-microwave_7221-scene_0_seed_0-6400-set_state.hdf5 \
    --demo-index 123
```

Notes:
- `--demo-index` accepts a single demo id like `123` or a subset like `1,3-4`.
- If the dataset contains many demos and `--demo-index` is omitted, the viewer auto-limits to `demo_0` for fast startup.
- The viewer shows RGB/depth cameras, action/state time series, and keeps playback on `frame_index`.

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

### 5. Evaluation Semantics

Open-door eval now uses a final-time contact-aware rule instead of an "ever touched" summary.

An episode is successful only if both are true:
- `door_open_any=True`: the articulated joint exceeded the configured openness threshold during the episode.
- `final_engaged=True`: at the final timestep, `final_handle_distance <= 0.1` between the handle reference point and the closest robot point (EEF or fingertip).

Each eval run writes a live CSV to `<output_dir>/per_episode_results.csv` with per-episode fields such as:
- `success`
- `max_openness`
- `door_open_any`
- `final_engaged`
- `min_handle_distance`
- `final_handle_distance`

Handle/robot debug markers are optional and remain off by default. Enable them for eval/debug runs with:

```bash
DEBUG_VISUALIZE_HANDLE=true bash scripts/run_pipeline.sh eval act microwave_7221 scene_7_seed_7 1000 \
  --output_dir=/path/to/eval_dir \
  --eval.n_episodes=10 \
  --env.headless=true
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
