# AutoMoMa Docker

This folder provides the main-branch Docker workflow for AutoMoMa. It follows the older `cvpr26` Isaac Sim container pattern, updated for the current working stack:

- Isaac Sim 5.1.0 base image (`nvcr.io/nvidia/isaac-sim:5.1.0`)
- CUDA toolkit 12.8 for compiling cuRobo
- Python 3.11 through `/isaac-sim/python.sh`
- Torch 2.7.0 + CUDA 12.8 wheels
- Local editable installs for `automoma`, `third_party/IsaacLab-Arena`, `third_party/lerobot`, and `third_party/curobo`
- Runtime pins mirrored from the local `automoma` conda environment for conversion/training dependencies
- Isaac Sim extension-cache compatibility so headless record/eval uses the same `Grey_Studio` lighting as the working conda environment

The image contains the plan, record, convert, train, and eval Python environments. Runtime scripts bind-mount the repo so generated `data/`, `logs/`, and `outputs/` remain on the host.

## Build

```bash
bash docker/build_docker.sh
```

Useful overrides: `bash docker/build_docker.sh --tag automoma:test --arch 8.9+PTX`

If Docker reports permission errors, the helper scripts try `sudo -E docker ...` automatically and may prompt for your sudo password. You can also add the current user to the `docker` group.

## Start A Shell

```bash
bash docker/run_docker.sh --gpu 0
```

Inside the container, the repo is available at `/workspace/automoma`, `python` is `/isaac-sim/python.sh`, and `scripts/run_pipeline.sh` works as on the host.

## Pipeline Commands

```bash
# Plan
bash docker/run_docker.sh --gpu 0 -- \
  bash scripts/run_pipeline.sh plan 7221 scene_0 train

# Record one demo with cameras in headless IsaacLab-Arena
bash docker/run_docker.sh --gpu 0 -- \
  bash scripts/run_pipeline.sh record microwave_7221 scene_0 1 --headless

# Convert to LeRobot
bash docker/run_docker.sh --gpu 0 -- \
  bash scripts/run_pipeline.sh convert lerobot microwave_7221 scene_0 1

# Eval a trained policy
bash docker/run_docker.sh --gpu 0 -- \
  bash scripts/run_pipeline.sh eval lerobot act microwave_7221 scene_0 1 --headless \
    --policy.path /workspace/automoma/outputs/train/lerobot/act_summit_franka_open-microwave_7221-scene_0-1/checkpoints/last/pretrained_model
```

## Real Test

This is the intended one-command validation. It runs Docker plan -> record -> convert, then trains for one step from the external host conda environment (`automoma` by default). Add `--run-eval` to also run a one-episode Docker eval load test.

```bash
bash docker/test_pipeline_host.sh --gpu 0 --episodes 1 --steps 1 --batch-size 1
```

The script writes test artifacts under:

```text
data/docker_smoke/
├── trajs/...
├── automoma/summit_franka_open-microwave_7221-scene_0-1.hdf5
└── lerobot/automoma/summit_franka_open-microwave_7221-scene_0-1/
```

Visualize the converted LeRobot dataset with:

```bash
conda activate automoma
lerobot-dataset-viz \
  --repo-id automoma/summit_franka_open-microwave_7221-scene_0-1 \
  --root data/docker_smoke/lerobot/automoma/summit_franka_open-microwave_7221-scene_0-1 \
  --episode-index 0 \
  --video-backend pyav
```

## Files

- `Dockerfile`: Isaac Sim 5.1.0 + CUDA/cuRobo + IsaacLab-Arena + LeRobot runtime.
- `requirements-automoma.txt`: dependency pins copied from the working local conda environment, excluding Isaac Sim and torch.
- `build_docker.sh`, `run_docker.sh`, `common.sh`: build/run helpers with GPU, cache, repo mounts, and sudo fallback.
- `smoke_plan_record_convert.sh`: in-container real plan/record/convert validation.
- `test_pipeline_host.sh`: host orchestration for Docker validation plus external-conda training.
- `check_env.py`: fast import/version check used during image build and smoke tests.
