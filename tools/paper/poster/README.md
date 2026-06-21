# Poster Reach Comparison Renders

Standalone poster-figure scripts for visualizing why `summit_franka` has a
large mobile-base workspace while a fixed Franka is limited by a fixed base.

## Inputs

- iTHOR scene USDs are expected under `assets/scene/ithor`. The script accepts
  the current collected layout, for example:
  `assets/scene/ithor/Collected_FloorPlan1_physics/FloorPlan1_physics.usd`.
  It also falls back to older `setup_<id>/Collected_*` paths from cuAKR config.
- Object placements are read from
  `.idea/cuakr_ithor/configs/dataset_config.yml`.
- Object categories are resolved with
  `.idea/cuakr_ithor/configs/object_data.json`, then loaded from
  `assets/object/<Category>/<id>/mobility/mobility.usd`.
  The renderer multiplies the scene transform scale by the default
  `instances.0.scaling` value from `object_data.json`.
- Summit-Franka is loaded through the existing IsaacLab-Arena
  `summit_franka` embodiment, which uses `assets/robot/summit_franka`.
- Fixed Franka is loaded from IsaacLab's official `FRANKA_PANDA_HIGH_PD_CFG`.

## Quick Check

This does not import Isaac Sim and is safe to run in a normal Python env:

```bash
python tools/paper/poster/render_reach_comparison.py --check_inputs
```

## Trajectory-Driven Workspace Sketches

For faster poster ideation, generate top-down comparison figures directly from
planned trajectories. This path uses the actual 12D planner output, not random
pose samples, and does not require Isaac Sim:

```bash
python tools/paper/poster/plot_trajectory_workspace.py
```

Default input:
`data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train/traj_data_train.pt`.

Outputs are written to `outputs/paper/poster/trajectory_workspace_7221/`:

- `mobile_workspace_trajectories.png`
- `fixed_vs_mobile_workspace.png`
- `poster_workspace_triptych.png`
- `metrics.json`

## Actual Ghost Comparison

For the poster figure that uses the already-planned iTHOR 7221 data, render
actual 3D ghost panels for fixed-base and mobile Summit-Franka:

```bash
bash tools/paper/poster/run_actual_ghost_render.sh \
  --image_width 1600 \
  --image_height 1050 \
  --mobile_traj_count 4 \
  --mobile_keyframes 5 \
  --mobile_workspace_ghosts 14 \
  --fixed_keyframes 6 \
  --fixed_arm_ghosts 8 \
  --output_root outputs/paper/poster/actual_ghost_comparison_aligned_v2
```

Defaults use:

- data root:
  `/media/xinhai/GIANT/Research/AutoMoMa/data/ithor/data_250917/ithor_floorplan1_1`
- object id: `7221`
- local poster robot configs under `tools/paper/poster/local/robots/`

Important alignment notes:

- Mobile `traj_mobile_*.pt` files are rendered from their first three base
  coordinates plus the planned arm joints.
- Fixed-base `traj_mobile_*.pt` files are 8D arm/object states and do not carry
  the fixed base pose. The renderer therefore prefers recorded HDF5
  `obs/joint/mobile_base` and `obs/joint/arm` when available.
- `ithor_floorplan1_1` / `7221` applies a poster-only XY display offset so the
  replay coordinates match the visible upper-right microwave in the current
  FloorPlan1 USD. This does not modify planner, replay, or dataset files.

Outputs include:

- `fixed_base/fixed_base_topdown_ghost.png`
- `mobile_base/mobile_base_topdown_ghost.png`
- `fixed_vs_mobile_actual_ghost.png`
- `manifest.json`

## Render

Use the wrapper so the Isaac Sim/IsaacLab environment variables are set. The
current default renders only `ithor_floorplan1_1`; the mobile robot gets only an
overview, and the fixed Franka gets an overview plus a robot-focused detail:

```bash
bash tools/paper/poster/run_poster_render.sh
```

Useful overrides:

```bash
bash tools/paper/poster/run_poster_render.sh \
  --scenes ithor_floorplan5_1 \
  --robots summit_franka franka \
  --summit_count 22 \
  --franka_count 8 \
  --image_width 2400 \
  --image_height 1600
```

Outputs are written to `outputs/paper/poster/<scene>/<robot>/`:

- `overview_all.png`
- `overview_robots_only.png`
- `overview_scene_objects_only.png`
- `detail_all.png`, `detail_robots_only.png`, `detail_scene_objects_only.png`
  for fixed Franka by default, or for any robot when `--views detail` is passed
- `overview_contact_sheet.png`
- `detail_contact_sheet.png`
- `manifest.json`
- `stage.usd`

Tune scene bounds, collision/exclusion rectangles, fixed-Franka support points,
sample counts, camera positions, and random-pose noise at the top of
`render_reach_comparison.py`.
