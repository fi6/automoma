# AutoMoMa-30K HDF5 Publish Prep

This folder contains first-party scripts for preparing the legacy AutoMoMa 30-scene HDF5 dataset without modifying the original files.

Data locations:

- Source legacy data: `data/automoma_30scenes/traj/summit_franka/<scene>/7221/camera_data/episode*.hdf5`
- Sorted symlink view: `data/automoma_30scenes/automoma-30k-sort/<scene>/episode*.hdf5`
- Converted current-schema files: `data/automoma_30scenes/automoma-30k-convert/<scene>/episode*.hdf5`

## 1. Build the sorted read-only symlink view

```bash
python scripts/publish/automoma-30k/organize_hdf5.py --force
```

The script only links `*.hdf5` files from the legacy `traj/summit_franka/<scene>/7221/camera_data` layout. It does not link unrelated files or hidden directories under the data disk root.

If the mounted data filesystem refuses links with `Operation not permitted`, write a manifest-only index:

```bash
python scripts/publish/automoma-30k/organize_hdf5.py --force --link-mode manifest
```

It writes:

- `manifest.jsonl`: one row per linked episode.
- `summary.json`: per-scene episode counts and warnings.

## 2. Convert a small sample

```bash
python scripts/publish/automoma-30k/convert_hdf5_format.py \
  --manifest data/automoma_30scenes/automoma-30k-sort/manifest.jsonl \
  --scene scene_0_seed_0 \
  --limit 3 \
  --overwrite
```

Outputs are written under `data/automoma_30scenes/automoma-30k-convert`. Existing converted files are skipped unless `--overwrite` is passed.

## 3. Batch conversion command

After checking the sample outputs:

```bash
python scripts/publish/automoma-30k/convert_hdf5_format.py
```

If the sorted directory is manifest-only, use:

```bash
python scripts/publish/automoma-30k/convert_hdf5_format.py \
  --manifest data/automoma_30scenes/automoma-30k-sort/manifest.jsonl
```

Use `--overwrite` only when intentionally regenerating existing converted files.

## Conversion Notes

The legacy files are state-observation episodes with:

- `obs/joint/{mobile_base,arm,gripper}`
- `obs/eef`
- `obs/{rgb,depth}/{ego_topdown,ego_wrist,fix_local}`
- `env_info`

The current recorder format, as produced by `scripts/run_pipeline.sh record`, expects `data/demo_i` entries containing actions, processed actions, observations, camera observations, initial state, and state snapshots.

The converter writes the same required groups/datasets as the current recorder schema:

- `actions`, `processed_actions`, and `obs/actions`
- `obs/{joint_pos,joint_vel,gripper_pos,eef_pos,eef_quat}`
- `camera_obs/*_{rgb,depth}`
- `initial_state/articulation/{robot,microwave_7221}`
- `states/articulation/{robot,microwave_7221}`

Fields that do not exist in the legacy files are filled with deterministic placeholders:

- Robot and object velocities are zero.
- Object articulation state is zero.
- Root poses are zero translation with identity quaternion.
- `success` is set to `False` to match the current example file's episode attribute.

Actions are derived from the legacy joint trajectory:

- Base action dimensions are relative deltas to match `--mobile_base_relative`.
- Arm and gripper action dimensions use the next recorded joint target.
- The last action repeats the final available target.

## Current Data Caveat

At the time this script was written, the mounted data root contained 6,752 HDF5 paths in non-empty scene folders, not 30,000. The data filesystem also refused symlink and hardlink creation, so the sample run used `--link-mode manifest`. `summary.json` records the exact scene counts after organization.
