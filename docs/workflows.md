# AutoMoMa Workflows

Run these commands from the repository root. `automoma/planning` remains the source of truth for the planner; the shell entrypoints only orchestrate planning, replay, conversion, training, and evaluation. For the detailed end-to-end handoff guide, see `docs/pipeline.md`.

## Public Entrypoints

| Entrypoint | Purpose |
| --- | --- |
| `python scripts/plan.py` | Generate cuRobo trajectories from `configs/plan.yaml`. |
| `bash scripts/run_pipeline.sh` | Run record, replay, convert, train, eval, record-dataset-eval, or debug workflows. |
| `bash scripts/quickstart.sh` | Show concise example commands for a standard local workflow. |
| `source scripts/setup_sim_env.sh` | Configure Isaac Sim / IsaacLab environment variables for simulation workflows. |

## Planning

```bash
python scripts/plan.py
python scripts/plan.py object_id=7221 scene_name=scene_0_seed_0
python scripts/plan.py object_id=7221 scene_name=scene_0_seed_0 mode=train
python scripts/plan.py object_id=7221 scene_name=scene_0_seed_0 planner.visualize_collision=true
```

Planning writes trajectory artifacts under `data/trajs/summit_franka/<object_name>/<scene_name>/<train|test>/`.

## Recording And Replay

```bash
bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 30 --headless
bash scripts/run_pipeline.sh replay microwave_7221 scene_0_seed_0 4 --metrics
bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 \
  --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train/traj_data_train.pt
```

Recording writes raw HDF5 demos under `data/automoma/`. Replay is useful for inspecting planned trajectories without writing full datasets.
Recording and eval use absolute mobile-base actions by default; pass `--mobile_base_relative` only for relative-delta datasets.

## Conversion And Training

```bash
bash scripts/run_pipeline.sh convert lerobot microwave_7221 scene_0_seed_0 30
bash scripts/run_pipeline.sh train lerobot act microwave_7221 scene_0_seed_0 30
bash scripts/run_pipeline.sh eval lerobot act microwave_7221 scene_0_seed_0 30 --headless
```

LeRobot conversion writes datasets under `data/lerobot/automoma/`. Training outputs checkpoints under `outputs/train/`.

## Asset Preparation

Asset preparation helpers are maintainer tools, not primary public entrypoints:

```bash
python tools/assets/prepare_object.py --object_type Microwave --object_id 7221
python tools/assets/prepare_scene.py --scene_name scene_0_seed_0
python tools/assets/prepare_scene_pipeline.py --scene-name scene_0_seed_0
```

Use these only when preparing local assets. Keep planner asset roots aligned with IsaacLab-Arena asset roots.

## Dataset Inspection

```bash
python tools/dataset/viz_hdf5.py \
  data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5

python tools/dataset/convert_hdf5_layout.py \
  data/automoma/split_dataset \
  data/automoma/merged_dataset.hdf5 \
  --direction merge --mode copy --overwrite
```

## Evaluation Semantics

Open-door eval success is `door_open_any && final_engaged`. `final_engaged` means `final_handle_distance <= 0.1` at the final timestep. The primary per-episode artifact is `per_episode_results.csv`.
