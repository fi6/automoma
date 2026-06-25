# AutoMoMa Pipeline README

This document is the handoff guide for running the full AutoMoMa local pipeline:

```text
assets -> plan -> record/replay -> convert -> train -> eval
```

The codebase intentionally keeps generated assets and datasets out of git. A working checkout needs a local asset bundle under `assets/` or equivalent roots exported through environment variables.

## 0. Environment

Run commands from the repository root. Install the dependencies from the root `README.md`, then activate the environment that contains AutoMoMa, cuRobo, IsaacLab-Arena, and LeRobot:

```bash
conda activate automoma
source scripts/setup_sim_env.sh
```

The main wrapper also exports the asset roots before every run:

```bash
export AUTOMOMA_OBJECT_ROOT="$PWD/assets/object"
export AUTOMOMA_SCENE_ROOT="$PWD/assets/scene/infinigen/scene_v2"
export AUTOMOMA_ROBOT_ROOT="$PWD/assets/robot"
```

If your conda activation hook sets a different `AUTOMOMA_SCENE_ROOT`, either unset it or override it before running the pipeline. `scripts/run_pipeline.sh` prints the roots it will use at startup; check those lines first when assets fail to load.

## 1. Asset Preparation

The planner, recorder, and evaluator must resolve the same object, scene, and robot assets. Keep these roots aligned:

```text
assets/
|-- object/<Category>/<object_id>/
|-- robot/summit_franka/
`-- scene/infinigen/scene_v2/<scene_name>/
```

For the default examples, `<object_name>` is the lowercase category plus id, such as `microwave_7221`; `<object_id>` is just `7221`.

### Objects

Objects come from PartNet Mobility. A prepared object directory should contain the raw URDF/meshes, generated simulator assets, grasp poses, and planner AKR configs:

```text
assets/object/Microwave/7221/
|-- mobility.urdf
|-- mobility/mobility.usd
|-- mobility/mobility.glb
|-- 7221_0_scaling.urdf
|-- grasp/0000.npy
`-- summit_franka_7221_0_grasp_0000.yml
```

Use the object preparation helper after downloading or copying a PartNet Mobility object:

```bash
# Fix URDF and mesh names only; does not require Isaac Sim.
python tools/assets/prepare_object.py --object_type Microwave --object_id 7221 --fix_only

# Fix URDF/meshes and convert mobility.urdf to mobility/mobility.usd.
python tools/assets/prepare_object.py --object_type Microwave --object_id 7221
```

The current planner config reads object metadata from `configs/plan.yaml`. For a new object, add or verify:

- `asset_type`, `scale`, `urdf_path`
- `handle_link`, `joint_name`
- `akr_template`
- `grasp_ids`
- `goal_angle`

Infinigen scene generation reads `mobility/mobility.glb`; IsaacLab-Arena loads `mobility/mobility.usd`; the planner may use the scaled URDF and AKR ymls from `configs/plan.yaml`.

### Scenes

AutoMoMa currently uses two scene sources:

- ISOR scenes: manually curated or manually adjusted scenes.
- Infinigen/InfinityIM scenes: procedurally generated scenes from `third_party/infinigen` on its `automoma` branch.

Both sources must end up in the same runtime layout:

```text
assets/scene/infinigen/scene_v2/<scene_name>/
|-- scene/scene.blend
|-- export/solve_state.json
|-- export/export_scene.blend/export_scene.usdc
`-- info/metadata.json
```

The canonical scene USD and metadata paths are configured in `configs/plan.yaml`:

```yaml
scene:
  infinigen:
    usd_subpath: export/export_scene.blend/export_scene.usdc
    metadata_subpath: info/metadata.json
```

#### Infinigen/InfinityIM Generation

The AutoMoMa branch of `third_party/infinigen` generates kitchen scenes with a three-stage script:

```bash
cd third_party/infinigen
bash scripts/automoma/generate_kitchen_rooms.sh
```

That script currently uses constants such as `NUM_SCENES`, `SCENE_NAME`, `START_SEED`, and `BASE_DIR`, so review them before launching a batch. Under the hood, the important commands are:

```bash
python -m infinigen_examples.generate_indoors \
  --seed 0 \
  --task coarse \
  --time_record \
  --output_folder output/kitchen_test/kitchen_0911/scene_0_seed_0/scene \
  --configs kitchen_only.gin

python -m infinigen.tools.export \
  --input_folder output/kitchen_test/kitchen_0911/scene_0_seed_0/scene \
  --output_folder output/kitchen_test/kitchen_0911/scene_0_seed_0/export \
  --format usdc \
  --resolution 1024
```

The current Infinigen branch resolves external PartNet assets with paths relative
to the process current working directory, for example `assets/object/Oven/...`
and `assets/object/Microwave/7221/...`. Before launching a batch, make sure the
working directory used by Infinigen can see the repository asset tree as
`assets/object`. A non-invasive smoke setup is to create a temporary workdir with
symlinks to the Infinigen packages and repository assets, then run the commands
from that workdir:

```bash
INFINIGEN_WORKDIR="${INFINIGEN_WORKDIR:-/tmp/automoma-infinigen-work}"
mkdir -p "$INFINIGEN_WORKDIR"
ln -sfn "$PWD/assets" "$INFINIGEN_WORKDIR/assets"
ln -sfn "$PWD/third_party/infinigen/infinigen" "$INFINIGEN_WORKDIR/infinigen"
ln -sfn "$PWD/third_party/infinigen/infinigen_examples" "$INFINIGEN_WORKDIR/infinigen_examples"

cd "$INFINIGEN_WORKDIR"
PYTHONPATH="<repo-root>/third_party/infinigen:${PYTHONPATH:-}" \
  python -m infinigen_examples.generate_indoors \
    --seed 0 \
    --task coarse \
    --time_record \
    --output_folder /tmp/automoma-infinigen-scene/scene_0_seed_0/scene \
    --configs kitchen_only.gin
```

Specific PartNet Mobility objects are selected by `third_party/infinigen/infinigen/assets/static_assets/requirement.json`, not by a public command-line flag. The checked-in example looks like:

```json
{
  "static_objects": [
    {
      "asset_type": "Microwave",
      "asset_id": "7221",
      "urdf_path": "assets/object/Microwave/7221/mobility.urdf",
      "scale": 0.3563
    }
  ]
}
```

Infinigen imports the selected PartNet object by replacing `mobility.urdf` with
`mobility/mobility.glb`. For the example above, verify that
`assets/object/Microwave/7221/mobility/mobility.glb` exists before generation.
If a local asset dump only contains a USD asset or `mobility_manual/mobility.glb`,
convert or install the GLB into the expected `mobility/` location as part of
asset preparation.

Keep a copy of the requirement JSON used for each generated batch. After generation, copy or sync successful scene directories into the runtime scene root from the repository root, for example:

```bash
cd <repo-root>
mkdir -p assets/scene/infinigen/scene_v2
rsync -a third_party/infinigen/output/kitchen_test/kitchen_0911/scene_0_seed_0 \
  assets/scene/infinigen/scene_v2/
```

#### Scene Post-Processing

Generated scenes need one AutoMoMa preparation pass before planning or simulation:

```bash
python tools/assets/prepare_scene_pipeline.py \
  --scene-root assets/scene/infinigen/scene_v2 \
  --scene-name scene_0_seed_0 \
  --requirement-mode multi
```

Use `--scene-name all` for a batch after spot-checking one scene.

`tools/assets/prepare_scene_pipeline.py` performs eight ordered steps:

1. Validate that `metadata.json`, `solve_state.json`, and `export_scene.usdc` exist.
2. Optionally apply interactive absolute object transform edits. This is off by default.
3. Write an embedded `info/requirement.json`.
4. Check `metadata.json` against the selected requirement.
5. Restructure the USD stage so scene prims live under `/World/scene`.
6. Deactivate ceiling, exterior, and camera prims that interfere with simulation.
7. Optionally lower KitchenSpaceFactory tabletop objects by `0.03m`. This is off by default.
8. Apply final metadata rotation and Z-offset fixes, including the `/World/scene` Z offset.

The older helper remains useful for the final rotation/Z-offset fixes only:

```bash
python tools/assets/prepare_scene.py --scene_name scene_0_seed_0
```

Use the full pipeline for newly generated scenes.

### Robot

The default robot is Summit + Franka:

```text
assets/robot/summit_franka/
|-- summit_franka.yml
|-- summit_franka.urdf
`-- summit_franka/summit_franka.usd
```

`configs/plan.yaml` defines the planner robot config and DOF convention:

```yaml
robot:
  summit_franka:
    mobile_base_dof: 3
    arm_dof: 7
    gripper_dof: 2
    total_dof: 12
```

Changing robot roots, joint order, or DOF counts is a cross-system change: update planner configs, IsaacLab-Arena actions, conversion, training, and eval together.

## 2. Planning

Planning is first-party and cuRobo-based. The public entrypoint is:

```bash
python scripts/plan.py [--config configs/plan.yaml] [OmegaConf overrides...]
```

The shell wrapper is:

```bash
bash scripts/run_pipeline.sh plan <object_id> <scene_name> <train|test> [overrides...]
```

Plan both train and test splits:

```bash
bash scripts/run_pipeline.sh plan 7221 scene_0_seed_0 train
bash scripts/run_pipeline.sh plan 7221 scene_0_seed_0 test
```

For a quick planning smoke run, limit successful trajectories and write to a scratch root:

```bash
python scripts/plan.py object_id=7221 scene_name=scene_0_seed_0 mode=train \
  output_dir=data/smoke/trajs \
  planner.output.max_successful_trajectories=10 \
  resume=false
```

Planning writes:

```text
data/trajs/summit_franka/<object_name>/<scene_name>/<split>/
|-- grasp_0000/
|   |-- ik_data.pt
|   |-- ik_goal_data.pt
|   `-- traj_data.pt
`-- traj_data_<split>.pt
```

`traj_data.pt` is the per-grasp cuRobo trajectory output. `traj_data_train.pt` and `traj_data_test.pt` are IsaacLab-Arena-compatible 12D files with these keys:

```text
start_robot  [N, 12]
start_obj    [N, 1]
goal_robot   [N, 12]
goal_obj     [N, 1]
traj_robot   [N, T, 12]
traj_obj     [N, T, 1]
traj_success [N]
```

The current 12D action convention is:

```text
base_x, base_y, base_theta,
panda_joint1..panda_joint7,
panda_finger_joint1, panda_finger_joint2
```

Planning handles the 11D to 12D conversion internally by appending two gripper dimensions and prepending a short grasp-closing phase. Do not change object-joint sign, gripper state, or mobile-base order in only one stage.

## 3. Replay, Record, And Debug

Simulation replay and recording run through `third_party/IsaacLab-Arena`, but should normally be launched through `scripts/run_pipeline.sh` from the repo root.

### Replay Without HDF5

Replay is the fastest simulation-level check because it does not write a dataset:

```bash
bash scripts/run_pipeline.sh replay microwave_7221 scene_0_seed_0 1 \
  --headless \
  --metrics \
  --metrics_file debug/smoke/replay_microwave_scene0.csv \
  --episode_indices 0 \
  --disable_cameras
```

Replay defaults to the train trajectory:

```text
data/trajs/summit_franka/<object_name>/<scene_name>/train/traj_data_train.pt
```

Override it with `--traj_file` when testing a custom split or selection.

### Record HDF5 Demos

Record train trajectories to a single merged HDF5:

```bash
bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 10 --headless
```

The default output name is:

```text
data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5
```

Record one HDF5 per episode when batch jobs need resumable chunks:

```bash
bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 10 \
  --split_episodes \
  --dataset_file data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10 \
  --headless
```

Record with eval-style filtering:

```bash
bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 10 \
  --validate_record_success \
  --headless
```

The default replay/record execution is physics drive mode. For low-level set-state replay or record experiments, pass the IsaacLab-Arena passthrough flags:

```bash
bash scripts/run_pipeline.sh replay microwave_7221 scene_0_seed_0 1 \
  --set_state \
  --object_joint_names joint_0 \
  --headless
```

Use `--mobile_base_relative` only when intentionally producing relative-base action datasets. The default dataset and eval convention is absolute mobile-base actions.

### Debug Trajectories

Debug any planner `.pt` artifact in IsaacLab-Arena:

```bash
bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 \
  --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train/grasp_0000/ik_data.pt

bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 \
  --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train/traj_data_train.pt
```

## 4. Dataset Conversion

The raw HDF5 output can be converted to LeRobot or RoboTwin policy formats.

### LeRobot

```bash
bash scripts/run_pipeline.sh convert lerobot microwave_7221 scene_0_seed_0 10
```

Default output:

```text
data/lerobot/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10/
```

Use explicit paths when converting a nonstandard HDF5:

```bash
bash scripts/run_pipeline.sh convert lerobot microwave_7221 scene_0_seed_0 10 \
  --data_root data/automoma \
  --hdf5_name summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5 \
  --repo_id automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10 \
  --output_dir data/lerobot/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10
```

Split HDF5 directories are detected automatically by the wrapper and converted without a temporary merged file.

### RoboTwin

RoboTwin conversion requires a target policy family:

```bash
bash scripts/run_pipeline.sh convert robotwin microwave_7221 scene_0_seed_0 10 \
  --policy dp3
```

If the input is a split HDF5 directory, the wrapper temporarily merges it before calling the RoboTwin policy converter.

### HDF5 Layout Utilities

Convert between merged and per-episode HDF5 layouts:

```bash
python tools/dataset/convert_hdf5_layout.py \
  data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5 \
  data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10-split \
  --direction split \
  --mode copy \
  --overwrite

python tools/dataset/convert_hdf5_layout.py \
  data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10-split \
  data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10-merged.hdf5 \
  --direction merge \
  --mode copy \
  --overwrite
```

## 5. Training

### LeRobot Policies

Train ACT or diffusion policies on the converted LeRobot dataset:

```bash
bash scripts/run_pipeline.sh train lerobot act microwave_7221 scene_0_seed_0 10
bash scripts/run_pipeline.sh train lerobot diffusion microwave_7221 scene_0_seed_0 10
```

Short smoke training run:

```bash
WANDB_MODE=disabled bash scripts/run_pipeline.sh train lerobot act microwave_7221 scene_0_seed_0 10 \
  --dataset.repo_id automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10 \
  --dataset.root data/lerobot/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10 \
  --output_dir outputs/train/smoke/act_microwave_7221_scene_0_seed_0_10 \
  --job_name act_microwave_7221_scene_0_seed_0_smoke \
  --steps 10 \
  --batch_size 2 \
  --wandb.enable=false
```

Default checkpoints land under:

```text
outputs/train/lerobot/<policy>_summit_franka_open-<object>-<scene>-<num_ep>/checkpoints/
```

### RoboTwin Policies

Train RoboTwin DP3 through the wrapper:

```bash
bash scripts/run_pipeline.sh train robotwin dp3 microwave_7221 scene_0_seed_0 10 \
  --seed 42 \
  --gpu_id 0
```

The wrapper calls `tools/robotwin/robotwin_train.sh`, which dispatches into `third_party/RoboTwin/policy/<POLICY>/`.

## 6. Evaluation

Evaluation needs three things:

- prepared assets matching the training/recording run,
- a policy checkpoint,
- test trajectories from `mode=test`.

Create the test trajectories before eval:

```bash
bash scripts/run_pipeline.sh plan 7221 scene_0_seed_0 test
```

### LeRobot Eval

```bash
bash scripts/run_pipeline.sh eval lerobot act microwave_7221 scene_0_seed_0 10 \
  --headless \
  --eval.n_episodes=10
```

Explicit checkpoint and test trajectory:

```bash
bash scripts/run_pipeline.sh eval lerobot act microwave_7221 scene_0_seed_0 10 \
  --policy.path outputs/train/lerobot/act_summit_franka_open-microwave_7221-scene_0_seed_0-10/checkpoints/last/pretrained_model \
  --traj_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/test/traj_data_test.pt \
  --output_dir outputs/eval/lerobot/act_summit_franka_open-microwave_7221-scene_0_seed_0-10 \
  --headless \
  --eval.n_episodes=10
```

### RoboTwin Eval

```bash
bash scripts/run_pipeline.sh eval robotwin dp3 microwave_7221 scene_0_seed_0 10 \
  --checkpoint_root outputs/train/robotwin/dp3_summit_franka_open-microwave_7221-scene_0_seed_0-10 \
  --traj_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/test/traj_data_test.pt \
  --headless \
  --drive
```

On branches that include set-state evaluation support for RoboTwin DP3, compare drive and set execution by changing only the action execution flag and output directory:

```bash
bash scripts/run_pipeline.sh eval robotwin dp3 microwave_7221 scene_0_seed_0 10 \
  --checkpoint_root outputs/train/robotwin/dp3_summit_franka_open-microwave_7221-scene_0_seed_0-10 \
  --traj_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/test/traj_data_test.pt \
  --output_dir outputs/eval/robotwin/dp3_microwave_7221_scene_0_seed_0_10_set \
  --headless \
  --set
```

This isolates policy/checkpoint/scene effects from execution-mode effects.

### Record-Dataset Eval

Evaluate a recorded HDF5 with the runtime eval rule:

```bash
bash scripts/run_pipeline.sh record_dataset_eval microwave_7221 scene_0_seed_0 10 \
  --dataset_file data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5 \
  --traj_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/test/traj_data_test.pt \
  --output_dir outputs/eval/record_dataset/microwave_7221_scene_0_seed_0_10 \
  --headless
```

Open-door eval success is:

```text
door_open_any && final_engaged
```

`final_engaged` means the final timestep has `final_handle_distance <= 0.1`. The primary per-episode artifact is:

```text
<output_dir>/per_episode_results.csv
```

Handle debug markers are opt-in:

```bash
DEBUG_VISUALIZE_HANDLE=true bash scripts/run_pipeline.sh eval lerobot act microwave_7221 scene_0_seed_0 10 \
  --headless \
  --eval.n_episodes=1
```

## 7. Minimal Smoke Tests

Run these after setup and before launching a long batch.

### Entrypoint Syntax

```bash
bash -n scripts/run_pipeline.sh
python -m py_compile \
  scripts/plan.py \
  tools/assets/prepare_object.py \
  tools/assets/prepare_scene.py \
  tools/assets/prepare_scene_pipeline.py \
  tools/dataset/convert_hdf5_layout.py \
  tools/dataset/convert_split_hdf5_to_lerobot_v30.py
```

### Trajectory Contract

```bash
python - <<'PY'
import torch

path = "data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train/traj_data_train.pt"
data = torch.load(path, map_location="cpu", weights_only=False)
print(sorted(data))
for key, value in data.items():
    if hasattr(value, "shape"):
        print(key, tuple(value.shape), value.dtype)
PY
```

Expected keys are `start_robot`, `start_obj`, `goal_robot`, `goal_obj`, `traj_robot`, `traj_obj`, and `traj_success`. Robot tensors must end in 12 dimensions.

### HDF5 Layout Smoke

```bash
tmpdir="$(mktemp -d)"
SMOKE_HDF5=data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-1.hdf5

python tools/dataset/convert_hdf5_layout.py \
  "$SMOKE_HDF5" \
  "$tmpdir/split" \
  --direction split \
  --mode copy \
  --overwrite

python tools/dataset/convert_hdf5_layout.py \
  "$tmpdir/split" \
  "$tmpdir/merged.hdf5" \
  --direction merge \
  --mode copy \
  --overwrite
```

### LeRobot Conversion Smoke

Use a small existing HDF5 or record one episode first:

```bash
SMOKE_HDF5_NAME=summit_franka_open-microwave_7221-scene_0_seed_0-1.hdf5

rm -rf /tmp/automoma_lerobot_convert_smoke
bash scripts/run_pipeline.sh convert lerobot microwave_7221 scene_0_seed_0 1 \
  --data_root data/automoma \
  --hdf5_name "$SMOKE_HDF5_NAME" \
  --repo_id automoma/smoke_microwave_7221_scene_0_seed_0 \
  --output_dir /tmp/automoma_lerobot_convert_smoke
```

The output should contain `data/`, `meta/`, and `videos/`.

### Simulation Smoke

Requires prepared scene assets, Isaac Sim, and GPU:

```bash
bash scripts/run_pipeline.sh replay microwave_7221 scene_0_seed_0 1 \
  --headless \
  --metrics \
  --metrics_file debug/smoke/replay_microwave_scene0.csv \
  --episode_indices 0 \
  --disable_cameras
```

If this fails before simulation starts, first check the printed `AUTOMOMA_*_ROOT` values and confirm the scene has both `info/metadata.json` and `export/export_scene.blend/export_scene.usdc`.

## 8. Common Failure Points

- Missing `assets/scene/...`: plan, replay, record, and eval all require prepared scene USD and metadata.
- Infinigen requirement mismatch: generation uses `third_party/infinigen/infinigen/assets/static_assets/requirement.json`; post-processing writes `<scene>/info/requirement.json`. Keep both aligned for new batches.
- Wrong trajectory split: record uses `train/traj_data_train.pt`; eval uses `test/traj_data_test.pt`.
- Absolute vs relative base actions: absolute is the default. Only use `--mobile_base_relative` for datasets and eval settings created for that convention.
- Camera headless crashes: if headless `--enable_cameras` crashes inside RTX renderer startup, check the NVIDIA driver notes in the root `README.md`.
- Asset-root drift: conda activation hooks, `source scripts/setup_sim_env.sh`, and `scripts/run_pipeline.sh` can set roots. Trust the values printed by the wrapper for that run.
