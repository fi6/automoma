# AutoMoMa

A comprehensive Python framework for automated robot motion planning, trajectory generation, and manipulation task execution. AutoMoMa provides a complete pipeline for data planning, data recording, model training, and policy evaluation.

## Features

- **Motion Planning**: Collision-aware trajectory planning using cuRobo (no Isaac Sim required)
- **Data Generation**: Automated IK solving and trajectory optimization
- **Data Recording**: Record manipulation demonstrations in LeRobot format
- **Policy Training**: Train diffusion policies (DP) and ACT models for manipulation tasks
- **Policy Evaluation**: Evaluate policies with async LeRobot communication
- **Simulation**: Isaac Sim integration for realistic robot simulation (lazy-loaded)

## Project Structure

```
AutoMoMa/
├── configs/                    # Configuration files
│   ├── config.yaml            # Base configuration
│   └── exps/                  # Experiment configurations
│       ├── multi_object_open/
│       │   ├── plan.yaml      # Planning config
│       │   ├── record.yaml    # Recording config
│       │   ├── train.yaml     # Training config
│       │   └── eval.yaml      # Evaluation config
│       └── single_object_open_test/  # Test experiment
│           ├── plan.yaml
│           ├── record.yaml
│           ├── train.yaml     # Supports DP and ACT
│           └── eval.yaml
├── scripts/
│   ├── pipeline/              # Main pipeline scripts
│   │   ├── 1_generate_plans.py
│   │   ├── 2_render_dataset.py
│   │   ├── 3_train.py
│   │   └── 4_evaluate.py
│   └── example/               # Example scripts
├── src/automoma/              # Source code
│   ├── core/                  # Core types and interfaces
│   ├── planning/              # Motion planning modules
│   ├── datasets/              # Dataset handling
│   ├── evaluation/            # Policy evaluation
│   ├── simulation/            # Isaac Sim integration (lazy-loaded)
│   ├── tasks/                 # Task definitions
│   └── utils/                 # Utility functions
├── third_party/               # Third-party dependencies
│   ├── lerobot/              # LeRobot library
│   └── curobo/               # cuRobo motion planning
└── assets/                    # Robot and scene assets
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+
- Isaac Sim 2023.1+ (for simulation, not required for planning)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/chang-xinhai/AutoMoMa.git
cd AutoMoMa
git submodule update --init --recursive
```

2. Create conda environment:
```bash
conda env create -f environment.yaml
conda activate automoma
```

3. Install the package:
```bash
pip install -e .
```

4. Install optional dependencies:
```bash
# For planning
pip install -e ".[plan]"

# For recording (requires Isaac Sim)
pip install -e ".[record]"

# For development
pip install -e ".[dev]"
```

## Quick Start

### Using Experiment Configs

All scripts support the `--exp` argument to load experiment-specific configurations:

```bash
# Run the full pipeline with experiment name
python scripts/pipeline/1_generate_plans.py --exp multi_object_open
python scripts/pipeline/2_render_dataset.py --exp multi_object_open
python scripts/pipeline/3_train.py --exp multi_object_open
python scripts/pipeline/4_evaluate.py --exp multi_object_open
```

### 1. Generate Motion Plans (No Isaac Sim Required)

```bash
python scripts/pipeline/1_generate_plans.py --exp multi_object_open
# Or filter by scene/object:
python scripts/pipeline/1_generate_plans.py --exp multi_object_open --scene scene_0_seed_0 --object 7221
```

This script:
- Loads scene and object configurations
- Computes IK solutions for grasp poses
- Plans trajectories for articulated manipulation
- Filters and saves valid trajectories

**Note**: This script uses only cuRobo and does NOT require Isaac Sim!

### 2. Render Dataset (Requires Isaac Sim)

```bash
python scripts/pipeline/2_render_dataset.py --exp multi_object_open
# Run headless:
python scripts/pipeline/2_render_dataset.py --exp multi_object_open --headless
```

This script:
- Initializes SimulationApp (only once)
- Loads planned trajectories
- Replays them in Isaac Sim
- Records camera observations
- Saves data in LeRobot format

### 3. Train Policy

```bash
python scripts/pipeline/3_train.py --exp multi_object_open
# Or use single_object_open_test for quick testing:
python scripts/pipeline/3_train.py --exp single_object_open_test
```

This script:
- Loads the LeRobot dataset
- Configures the diffusion policy or ACT model
- Trains the model
- Saves checkpoints

### 4. Evaluate Policy (Requires Isaac Sim)

```bash
python scripts/pipeline/4_evaluate.py --exp multi_object_open
# Run headless:
python scripts/pipeline/4_evaluate.py --exp multi_object_open --headless
```

This script:
- Initializes SimulationApp (only once)
- Loads the trained checkpoint
- Runs policy inference with async communication
- Computes evaluation metrics
- Generates evaluation reports

## Isaac Sim Integration

AutoMoMa uses lazy-loading for Isaac Sim modules. This means:

1. **Planning scripts do NOT load Isaac Sim** - They only use cuRobo and can run without Isaac Sim installed
2. **Simulation modules require explicit initialization** - SimulationApp is initialized only once per process

### Using SimulationApp

For scripts that need Isaac Sim, use the `get_simulation_app()` function:

```python
# First, initialize SimulationApp (do this ONCE at the start)
from automoma.simulation import get_simulation_app

sim_app = get_simulation_app(headless=False, width=1920, height=1080)

# Now you can import and use simulation modules
from automoma.simulation.env_wrapper import SimEnvWrapper

env = SimEnvWrapper(cfg)
env.setup_env()
```

### Checking SimulationApp Status

```python
from automoma.simulation import is_sim_app_initialized, require_simulation_app

if not is_sim_app_initialized():
    sim_app = get_simulation_app(headless=True)

# Or raise an error if not initialized:
require_simulation_app()  # Raises RuntimeError if not initialized
```

## Configuration

### Experiment-Based Configuration

AutoMoMa uses a hierarchical configuration system:

1. **Base config** (`configs/config.yaml`): Default values
2. **Experiment config** (`configs/exps/{exp_name}/*.yaml`): Overrides for specific experiments

Access config values with attribute-style access (no `.get()` needed):

```python
from automoma import load_config

cfg = load_config("multi_object_open")
num_grasps = cfg.plan_cfg.num_grasps  # Attribute-style access
task_name = cfg.info_cfg.task
```

### Planning Configuration

```yaml
plan_cfg:
  voxel_dims: [5.0, 5.0, 5.0]
  voxel_size: 0.02
  collision_checker_type: VOXEL
  cluster:
    ap_fallback_clusters: 50
  plan_traj:
    stage_type: MOVE_ARTICULATED
    batch_size: 10
```

### Recording Configuration

```yaml
dataset_cfg:
  repo_id: "automoma/multi_object_open"
  fps: 15
  use_videos: true
camera_cfg:
  names: [ego_topdown, ego_wrist, fix_local]
```

### Training Configuration (DP and ACT)

```yaml
# Diffusion Policy
policy:
  type: diffusion
  n_action_steps: 8
  horizon: 16

# Or Action Chunking Transformer (ACT)
policy:
  type: act
  chunk_size: 100
  kl_weight: 10.0
```

### Evaluation Configuration

```yaml
evaluation:
  num_episodes: 50
  success_threshold: 0.05
async_inference:
  enabled: true
```

## API Usage

### Planning (No Isaac Sim)

```python
from automoma.planning import PlanningPipeline

pipeline = PlanningPipeline(plan_cfg)
pipeline.setup(scene_cfg, object_cfg, robot_cfg_path)
results = pipeline.run_full_pipeline(...)
```

### Dataset Recording (Requires Isaac Sim)

```python
from automoma.simulation import get_simulation_app
sim_app = get_simulation_app(headless=True)

from automoma.datasets import LeRobotDatasetWrapper

dataset = LeRobotDatasetWrapper(cfg)
dataset.create()
dataset.add(frame_data)
dataset.save()
dataset.close()
```

### Policy Evaluation

```python
from automoma.evaluation import PolicyRunner, get_model

# Get async model client
model = get_model(
    checkpoint_path="model.pt",
    policy_type="diffusion",
    async_mode=True
)

# Run inference
response = model.infer_sync(observation)
action = response.action
```

## Async Model Communication

AutoMoMa uses an asynchronous communication mechanism for policy inference, compatible with LeRobot:

```python
from automoma.evaluation import LeRobotModelClient

client = LeRobotModelClient(
    checkpoint_path="model.pt",
    device="cuda"
)
client.load_model()
client.start()  # Start async worker

# Non-blocking inference
request_id = client.submit_request(observation)
response = client.get_response(request_id)

# Cleanup
client.stop()
```

## Supported Tasks

- **Pick and Place**: Pick up objects and place at target locations
- **Reach and Open**: Reach handles and open articulated objects (doors, drawers, etc.)

## Experiments

### multi_object_open
Full experiment with multiple objects (7221, 11622, 101773) and scenes.

### single_object_open_test
Quick test experiment with:
- **Object**: 7221 only
- **Training scenes**: scene_0_seed_0, scene_1_seed_1
- **Test scenes**: scene_0_seed_0, scene_1_seed_1, scene_40_seed_40
- **Models**: Supports both DP and ACT

## License

MIT License

## Citation

```bibtex
@software{automoma2024,
  title={AutoMoMa: Automated Motion Planning for Manipulation},
  author={AutoMoMa Contributors},
  year={2024},
  url={https://github.com/chang-xinhai/AutoMoMa}
}
```
