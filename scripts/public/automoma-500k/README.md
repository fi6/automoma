# AutoMoMa-500k trajectory planning

This directory contains trajectory-only release scripts for the public AutoMoMa-500k dataset.

## Batch plan

```bash
python scripts/public/automoma-500k/plan_automoma_500k.py
```

Defaults:

- reads the 5 objects from `configs/plan.yaml`
- scans all scene directories under `assets/scene/infinigen/kitchen_1130`
- targets at least `100000` successful trajectories per object
- writes canonical per-scene files to `data/trajs/summit_franka/<object>/<scene>/train/traj_data_train.pt`
- writes each planning round to an isolated directory under `data/trajs/_automoma_500k_rounds` before merging only successful trajectories

Useful checks:

```bash
python scripts/public/automoma-500k/plan_automoma_500k.py --self-test
python scripts/public/automoma-500k/plan_automoma_500k.py --dry-run --scenes scene_0_seed_0
```

Pass extra `scripts/plan.py` overrides after `--`:

```bash
python scripts/public/automoma-500k/plan_automoma_500k.py -- planner.traj.batch_size=10
```

## Statistics

```bash
python scripts/public/automoma-500k/trajectory_statistics.py
```

Outputs are written to `outputs/statistics/automoma-500k/`:

- `trajectory_summary.json`
- `trajectory_summary.csv`
- `object_totals.png`
- `object_scene_heatmap.png`
- `scene_totals.png`
