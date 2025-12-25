# AutoMoMa

A comprehensive Python framework for automated robot motion planning, trajectory generation, and manipulation task execution. AutoMoMa provides a complete pipeline for data planning, data recording, model training, and policy evaluation.

## Features

- **Motion Planning**: Collision-aware trajectory planning using cuRobo
- **Data Generation**: Automated IK solving and trajectory optimization
- **Data Recording**: Record manipulation demonstrations in LeRobot format
- **Policy Training**: Train diffusion policies for manipulation tasks
- **Policy Evaluation**: Evaluate policies with async LeRobot communication
- **Simulation**: Isaac Sim integration for realistic robot simulation

## Project Structure

```
AutoMoMa/
├── configs/                    # Configuration files
│   ├── config.yaml            # Base configuration
│   └── exps/                  # Experiment configurations
│       └── multi_object_open/
│           ├── plan.yaml      # Planning config
│           ├── record.yaml    # Recording config
│           ├── train.yaml     # Training config
│           └── eval.yaml      # Evaluation config
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
│   ├── simulation/            # Isaac Sim integration
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
- Isaac Sim 2023.1+ (for simulation)

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

### 1. Generate Motion Plans

```bash
python scripts/pipeline/1_generate_plans.py --config configs/exps/multi_object_open/plan.yaml
```

This script:
- Loads scene and object configurations
- Computes IK solutions for grasp poses
- Plans trajectories for articulated manipulation
- Filters and saves valid trajectories

### 2. Render Dataset

```bash
python scripts/pipeline/2_render_dataset.py --config configs/exps/multi_object_open/record.yaml
```

This script:
- Loads planned trajectories
- Replays them in Isaac Sim
- Records camera observations
- Saves data in LeRobot format

### 3. Train Policy

```bash
python scripts/pipeline/3_train.py --config configs/exps/multi_object_open/train.yaml
```

This script:
- Loads the LeRobot dataset
- Configures the diffusion policy
- Trains the model
- Saves checkpoints

### 4. Evaluate Policy

```bash
python scripts/pipeline/4_evaluate.py --config configs/exps/multi_object_open/eval.yaml
```

This script:
- Loads the trained checkpoint
- Runs policy inference with async communication
- Computes evaluation metrics
- Generates evaluation reports

## Configuration

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

### Training Configuration

```yaml
policy:
  type: diffusion
training:
  batch_size: 64
  learning_rate: 1.0e-4
  num_epochs: 100
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

### Planning

```python
from automoma.planning import PlanningPipeline

pipeline = PlanningPipeline(plan_cfg)
pipeline.setup(scene_cfg, object_cfg, robot_cfg_path)
results = pipeline.run_full_pipeline(...)
```

### Dataset Recording

```python
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
