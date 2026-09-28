# Production Data Guide

This document defines the public, repository-local contract for producing
AutoMoMa data. It complements `docs/pipeline.md`: that document explains each
local pipeline stage, while this one explains how those stages must be used in
a restart-safe production run.

## Repository boundary

AutoMoMa owns these interfaces:

```text
assets -> cuRobo trajectories -> IsaacLab-Arena replay/record -> HDF5
       -> LeRobot or RoboTwin conversion
```

The public entrypoints are `scripts/plan.py` and `scripts/run_pipeline.sh`.
External projects may provide schedulers, immutable run configuration,
materialized scene variants, or dataset assembly, but they must preserve the
asset, trajectory, action, camera, and HDF5 contracts documented here and in
`docs/pipeline.md`.

For the current 30-scene Microwave 7221 effort, the companion RLMoMa checkout
contains the orchestration source of truth at:

```text
debug/2026-08-24-chair-render-ab/HANDOFF_DYNAMIC_MULTI_GPU_DATA_PIPELINE.md
```

That handoff is intentionally not duplicated here because GPU allocation,
private storage, notifications, and run-specific output roots belong to the
production deployment rather than to AutoMoMa's public interface.

## Current validated foundation

The companion 30-scene work has established the following integration facts:

- all 30 scenes have validated category-aware sparse LOD overlays without
  modifying their original USD assets;
- all 30 scenes passed loading tests at 1, 2, 4, and 8 environments;
- 8 environments per renderer container/GPU is the current tested default,
  but remains a run configuration rather than an AutoMoMa constant;
- all 30 policy-LOD scenes have semantically validated materialized USDC
  layers; external textures remain separate assets;
- static scene geometry must remain reference-composed across environments.
  Deep-copying the full scene per environment can exhaust RTX resources and
  produce device loss or bounds-box fallback rendering.

These results validate the scene inputs and rendering strategy. They do not
mean a general dynamic multi-GPU scheduler exists in this repository. The
current dynamic queue/controller/worker/validator design is still an external
production implementation task.

## Starting from zero

Complete the following gates before launching a long batch.

### 1. Install and align the environment

Follow the root `README.md`, then read `docs/pipeline.md` completely. Confirm
that planning, recording, and evaluation resolve the same object, scene, and
robot roots. Use Python 3.11 for the full Isaac Sim 5.1 environment.

At minimum, verify:

```bash
python --version
python -c "import torch; print(torch.cuda.is_available())"
python -c "import isaacsim, isaaclab, curobo; print('runtime ok')"
bash -n scripts/run_pipeline.sh
```

The operational replay/evaluation default is the `f4e200` contact profile:
robot/object static and dynamic friction are both `4.0`, and the Summit-Franka
gripper simulation effort limit is `200.0`. `scripts/run_pipeline.sh` exports
these values explicitly. They remain individually overridable through
`AUTOMOMA_ROBOT_OBJECT_STATIC_FRICTION`,
`AUTOMOMA_ROBOT_OBJECT_DYNAMIC_FRICTION`, and
`AUTOMOMA_GRIPPER_EFFORT_LIMIT`; any override is part of the immutable run
configuration and must be recorded with the output.

### 2. Validate assets and trajectory identity

Confirm the required object, robot, and scene files exist under the documented
asset roots. Record checksums for production inputs. Generate trajectories with
`scripts/plan.py`, or validate a frozen trajectory set before reuse.

For Microwave 7221, the maintained planning profile filters start-to-goal base
displacement at `1.0 m` and `0.35 rad`, and filters full raw-trajectory
excursion at `1.0 m` and `0.35 rad`.
These limits apply to newly planned trajectories. Downstream smoothing must
repeat the full-path excursion checks because interpolation can overshoot the
raw path; replay must never clip a finalized action sequence.

A logical sample identity must come from its scene, trajectory or episode,
seed, and immutable renderer configuration. GPU number, worker number, retry
number, and array position are execution details and must not define sample
identity.

Canonical `traj_data_train.pt` planning outputs now carry aligned identity
metadata in addition to the trajectory tensors:

```text
grasp_id   [N]
grasp_pose [N, 7]  # object-relative, scaled, quaternion wxyz
goal_angle [N]
scene_id   string
object_id  string
```

Success selection, limits, round merges, and resume operate on the tensor
metadata with the same indices. A metadata-free canonical file is not safe to
resume or render. Backfill it with
`tools/dataset/backfill_planning_metadata.py`; that command requires every
canonical row to have exactly one tensor-identical per-grasp source match and
stops on missing or ambiguous matches.

### 3. Run one replay and one recording smoke test

Replay first because it is cheaper and does not create a full camera dataset:

```bash
bash scripts/run_pipeline.sh replay microwave_7221 scene_0_seed_0 1 \
  --headless --metrics --episode_indices 0
```

Then record a small HDF5 using the exact production action, timing, camera, and
initial-state configuration. Do not infer production defaults from an older
dataset name or historical runbook.

```bash
bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 1 --headless
```

Open the result with HDF5 tooling and inspect representative RGB/depth frames.
An exit status of zero is necessary but not sufficient evidence of valid data.

### 4. Put production orchestration outside the stage commands

For a multi-GPU production, wrap AutoMoMa with a durable controller and
replaceable single-GPU workers. The controller must support transactional job
claims, expiring leases, heartbeats, bounded retries, GPU drain/removal, and
restart recovery. Each attempt must have its own staging and log directory.

Do not use a fixed modulo shard as the authoritative queue when GPUs must be
added, removed, or reassigned while a run is active. Do not let multiple
workers append concurrently to the same final dataset.

### 5. Validate before conversion or publication

At minimum, reject or quarantine output when the container fails, required
files are missing or empty, JSON/HDF5 cannot be parsed, required episodes or
camera streams are absent, stream frame counts disagree, arrays have invalid
shape/dtype/value ranges, fatal CUDA/RTX signatures appear, or planned input
checksums do not match.

Validate logical episodes independently when the file layout permits safe
sample-level filtering. Preserve original identifiers and failed attempts for
diagnosis.

For the Microwave 7221 production renderer, an environment is discarded when,
starting at frame 22, its handle distance is greater than 0.05 m for 10
consecutive frames. Its recorder buffer is cleared immediately and it waits
for the other vector environments without replacement. A retained HDF5 demo
has `traj_index`, `grasp_id`, `grasp_pose`, `goal_angle`, `scene_id`, and
`object_id` attributes. A job may therefore contain 1–8 demos; zero retained
demos is a filtered terminal result, not a retryable renderer failure.

### 6. Assemble only accepted data

Write into staging first. Promote validated output atomically and maintain an
accepted manifest separately from quarantine and filtered manifests. Build a
LeRobot, RoboTwin, or other consolidated dataset from accepted records only;
never discover publishable samples by globbing a mixed output tree.

Before deleting source HDF5, verify the converted dataset, its sample counts,
metadata, video/parquet readability, and an end-to-end checksum-backed archive.

`tools/dataset/convert_accepted_hdf5_to_lerobot_v30.py` reads only
`accepted.jsonl`, preserves its line and retained-episode order, and finalizes
one LeRobot output chunk per 200 retained episodes by default. Chunks may cross
HDF5 jobs and scenes, failed episodes consume no position, and the final short
chunk is finalized. Planning fields are written once in episode metadata, not
as repeated frame features. Source HDF5 files are never merged or rewritten.

## Production invariants

- Original assets are read-only inputs; derived scene layers use new paths.
- A retry keeps the same logical job ID and writes a new attempt.
- Partial, empty, corrupt, unvalidated, or quarantined artifacts are never
  published.
- Every accepted sample is traceable to input checksums, immutable config, job
  ID, and validation report.
- Renderer and validator failures are classified separately from GPU health;
  GPU-specific faults must not make deterministic input claims without
  evidence.
- Runtime deployment details and credentials do not belong in this public
  repository. Use environment variables or private deployment configuration.

## Related documentation

- `docs/pipeline.md`: detailed local asset-to-eval pipeline.
- `docs/workflows.md`: concise public command reference.
- `docs/2026-06-18_30k_pipeline.md`: historical fixed-run notes; useful as
  evidence, not as the dynamic scheduling contract.
- `docs/release/automoma-30k.md` and `docs/release/automoma-500k.md`: released
  dataset documentation.
