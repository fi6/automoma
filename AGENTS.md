# AGENTS.md

## Purpose
This file defines how an autonomous coding agent should operate in this repository.

The goal is to help the agent make correct, minimal, repo-aware changes across the AutoMoMa pipeline without breaking alignment between planning, recording, conversion, training, and evaluation.

## Repo Overview
This repository implements a mobile manipulation workflow around AutoMoMa:

`assets -> cuRobo planning -> IsaacLab-Arena replay/record -> LeRobot conversion/training -> IsaacLab-Arena evaluation`

Current architecture:
- Planning is first-party and cuRobo-based.
- Recording and evaluation are executed through `third_party/IsaacLab-Arena`.
- Dataset conversion and training flow into LeRobot-compatible outputs.
- Asset, pose, and trajectory conventions must stay aligned across all stages.

## Critical Entry Points
- `automoma/planning/planner.py`: core planning logic and planner-side environment alignment.
- `automoma/planning/pipeline.py`: end-to-end planning orchestration and 11D -> 12D trajectory conversion.
- `scripts/plan.py`: planning CLI entrypoint.
- `scripts/run_pipeline.sh`: main operational entrypoint for `record`, `convert`, `train`, `eval`, and `debug`.
- `scripts/run_simple.sh`: example commands and quick workflow reference.
- `configs/plan.yaml`: planning configuration source of truth.

## Core Invariants
- Planner logic currently uses cuRobo, while record and eval use `third_party/IsaacLab-Arena`.
- Any change affecting assets, object pose, scene pose, robot config, joint conventions, trajectory format, output keys, or naming must preserve planner/IsaacLab-Arena alignment.
- The planner-side root-pose and object-centering behavior must stay semantically aligned with IsaacLab-Arena `--object_center`.
- Generated trajectory files must remain compatible with IsaacLab-Arena replay/debug expectations and downstream LeRobot conversion.
- Asset roots and path assumptions exported by `scripts/run_pipeline.sh` are part of the operational contract unless intentionally migrated everywhere that depends on them.
- Changes to 11D/12D trajectory handling, object-joint sign conventions, gripper open/close state encoding, or lift offsets are cross-system changes, not local refactors.

## Working Rules
- Read the relevant implementation before changing behavior. Do not infer workflow details from filenames alone.
- Prefer editing first-party code in `automoma/`, `scripts/`, and `configs/`.
- Keep changes minimal and local unless the problem is clearly cross-cutting.
- If a task affects both planning and replay/eval semantics, inspect both sides before editing either side.
- Preserve existing CLIs and file layout unless the task explicitly requires interface changes.
- When behavior changes, describe which pipeline stage is affected: planning only, replay/eval only, or cross-system alignment.
- Prefer targeted fixes over broad cleanup.
- Treat generated data formats and intermediate `.pt` / `.hdf5` expectations as interfaces, not incidental implementation details.

## Third-Party Boundary
- Do not modify anything under `third_party/` without explicit user confirmation first.
- If you find a bug in `third_party/`, stop and report:
  - what is wrong,
  - why first-party code cannot safely work around it,
  - the smallest patch that would be required.
- Reading `third_party/` code for diagnosis is allowed and expected when cross-system alignment is involved.
- If a first-party change would only be correct together with a `third_party/` patch, do not partially implement it without user confirmation.

## Validation Expectations
- Run the smallest relevant validation for the area you changed.
- Prefer focused checks over full end-to-end runs unless the task specifically requires full pipeline validation.
- Do not claim compatibility across stages unless you either ran a relevant check or clearly state that you did not.

Use these defaults:
- Planning logic changes: verify config loading, planner entrypoint behavior, and any changed tensor/state conventions.
- Trajectory format changes: verify output keys, tensor shapes, DOF assumptions, and compatibility with `record` / `debug` expectations.
- Shell pipeline changes: verify argument ordering, path construction, and environment-variable propagation in `scripts/run_pipeline.sh`.
- Asset-loading changes: verify that planner and IsaacLab-Arena still resolve the same object, scene, and robot roots.

If validation is blocked by environment limits, missing assets, GPU requirements, Isaac Sim requirements, or runtime cost, say so explicitly.

## Common Workflows
- Planning: `python scripts/plan.py`
- Planning with overrides: `python scripts/plan.py scene_name=... object_id=...`
- Record/replay demos: `bash scripts/run_pipeline.sh record <object_name> <scene_name> <num_episodes> [overrides...]`
- Convert HDF5 to LeRobot: `bash scripts/run_pipeline.sh convert <object_name> <scene_name> <num_episodes> [overrides...]`
- Train a policy: `bash scripts/run_pipeline.sh train <policy> <object_name> <scene_name> <num_episodes> [overrides...]`
- Evaluate a policy: `bash scripts/run_pipeline.sh eval <policy> <object_name> <scene_name> <num_episodes> [overrides...]`
- Debug trajectory replay: `bash scripts/run_pipeline.sh debug <object_name> <scene_name> --debug_file <path>`

Use `scripts/run_simple.sh` as a quick reference for known working command patterns, but verify assumptions against the actual implementation before changing code.

## Response Expectations
When reporting completed work, include:
- what files were changed,
- whether the impact is planning only, replay/eval only, or cross-system,
- what validation was run,
- what was not validated,
- whether any suspected `third_party/` issue remains.

When blocked, be explicit about whether the blocker is:
- missing repo context,
- environment/runtime limitations,
- a cross-system mismatch,
- or a `third_party/` issue that requires user confirmation.
