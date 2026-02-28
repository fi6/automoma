#!/usr/bin/env python3
"""Prepare automoma trajectory data for IsaacLab-Arena recording / evaluation.

This script:
1. **Merges** per-grasp ``filtered_traj_data.pt`` files from the raw planner
   output into a single ``traj_data_11d.pt``.
2. **Converts** the merged 11-DOF data into the 12-DOF format expected by
   IsaacLab-Arena (adds gripper DOFs, adjusts lift joint, prepends grasp phase).
3. Saves the result as ``traj_data_train.pt`` or ``traj_data_test.pt``
   depending on ``--mode``.

Usage (from lerobot-arena root):
    # Process a single object / scene
    python scripts/prepare_traj.py --object_name microwave_7221 --scene_name scene_0_seed_0 --mode train

    # Process all objects for one scene
    python scripts/prepare_traj.py --scene_name scene_0_seed_0 --mode train

    # Process everything
    python scripts/prepare_traj.py --mode train
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


# Raw trajectory output from the external planner
RAW_TRAJ_ROOT = get_repo_root() / "output" / "infinigen_traj_1130" / "summit_franka"

# Destination for processed trajectories
OUTPUT_TRAJ_ROOT = get_repo_root() / "data" / "trajs" / "summit_franka"


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def merge_grasp_trajectories(object_traj_dir: Path, only_successful: bool = True) -> dict:
    """Merge all grasp_*/filtered_traj_data.pt into one dict.

    The returned dict has the same keys as the per-grasp files
    (start_state, goal_state, traj, success) with batch dims concatenated.
    """
    grasp_dirs = sorted(
        d for d in object_traj_dir.iterdir()
        if d.is_dir() and d.name.startswith("grasp_")
    )

    if not grasp_dirs:
        raise FileNotFoundError(f"No grasp_* directories in {object_traj_dir}")

    merged = {"start_state": [], "goal_state": [], "traj": [], "success": []}

    for gd in grasp_dirs:
        pt_path = gd / "filtered_traj_data.pt"
        if not pt_path.exists():
            print(f"  Skipping {gd.name} — filtered_traj_data.pt not found")
            continue

        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        if only_successful:
            mask = data["success"]
            n_success = mask.sum().item()
            if n_success == 0:
                print(f"  Skipping {gd.name} — no successful trajectories")
                continue
            for k in ("start_state", "goal_state"):
                merged[k].append(data[k][mask])
            merged["traj"].append(data["traj"][mask])
            merged["success"].append(data["success"][mask])
        else:
            for k in merged:
                merged[k].append(data[k])

    for k in merged:
        merged[k] = torch.cat(merged[k], dim=0)

    return merged


# ---------------------------------------------------------------------------
# 11-DOF → 12-DOF conversion (from fix_pt.py)
# ---------------------------------------------------------------------------

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0
PREPEND_STEPS = 4


def _transform_arm_object(tensor: torch.Tensor):
    """Split arm (0:10) / object (10:), adjust lift joint, negate object."""
    robot_joints = tensor[..., :10].clone()
    obj_state = tensor[..., 10:].clone()
    robot_joints[..., 1] -= 1.5
    obj_state = -obj_state
    return robot_joints, obj_state


def _process_single_state(tensor: torch.Tensor, gripper_val: float):
    robot_arm, obj_state = _transform_arm_object(tensor)
    pad = list(robot_arm.shape[:-1]) + [2]
    gripper = torch.full(pad, gripper_val, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([robot_arm, gripper], dim=-1), obj_state


def _process_trajectory(tensor: torch.Tensor):
    batch_size, orig_time, _ = tensor.shape
    device, dtype = tensor.device, tensor.dtype

    raw_arm, raw_obj = _transform_arm_object(tensor)

    # Grasp phase: hold frame-0, close gripper
    grasp_arm = raw_arm[:, 0:1, :].repeat(1, PREPEND_STEPS, 1)
    grasp_obj = raw_obj[:, 0:1, :].repeat(1, PREPEND_STEPS, 1)
    closing = torch.linspace(GRIPPER_OPEN, GRIPPER_CLOSED, steps=PREPEND_STEPS, device=device, dtype=dtype)
    grasp_gripper = closing.view(1, -1, 1).repeat(batch_size, 1, 2)
    grasp_robot = torch.cat([grasp_arm, grasp_gripper], dim=-1)

    # Pull phase: move arm, gripper closed
    pull_gripper = torch.full((batch_size, orig_time, 2), GRIPPER_CLOSED, device=device, dtype=dtype)
    pull_robot = torch.cat([raw_arm, pull_gripper], dim=-1)

    return (
        torch.cat([grasp_robot, pull_robot], dim=1),
        torch.cat([grasp_obj, raw_obj], dim=1),
    )


def convert_11d_to_12d(merged: dict) -> dict:
    """Convert merged 11-DOF data to the 12-DOF format expected by IsaacLab-Arena."""
    out = {}
    out["start_robot"], out["start_obj"] = _process_single_state(merged["start_state"], GRIPPER_OPEN)
    out["goal_robot"], out["goal_obj"] = _process_single_state(merged["goal_state"], GRIPPER_CLOSED)
    out["traj_robot"], out["traj_obj"] = _process_trajectory(merged["traj"])
    out["traj_success"] = merged["success"]
    return out


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _parse_object_name(object_name: str):
    """Parse 'microwave_7221' -> ('microwave', '7221')."""
    parts = object_name.split("_")
    asset_id = parts[-1]
    asset_type = "_".join(parts[:-1])
    return asset_type, asset_id


def iter_targets(object_name: str | None, scene_name: str | None):
    """Yield (scene_dir, object_id, object_name_full) tuples.

    ``scene_dir`` is under RAW_TRAJ_ROOT.
    """
    if not RAW_TRAJ_ROOT.exists():
        print(f"Error: Raw trajectory root not found: {RAW_TRAJ_ROOT}")
        return

    # Determine scenes
    if scene_name and scene_name != "all":
        scene_dirs = [RAW_TRAJ_ROOT / scene_name]
    else:
        scene_dirs = sorted(
            d for d in RAW_TRAJ_ROOT.iterdir()
            if d.is_dir() and d.name.startswith("scene_")
        )

    # Determine object IDs
    for sd in scene_dirs:
        if not sd.is_dir():
            print(f"  Scene directory not found: {sd}")
            continue

        if object_name and object_name != "all":
            _, obj_id = _parse_object_name(object_name)
            obj_dir = sd / obj_id
            if obj_dir.is_dir():
                yield sd, obj_id, object_name
            else:
                print(f"  Object {object_name} (id={obj_id}) not found in {sd}")
        else:
            for obj_dir in sorted(sd.iterdir()):
                if obj_dir.is_dir() and obj_dir.name.isdigit():
                    # We need to figure out the full object_name (type_id).
                    # We infer the type from the assets directory.
                    obj_id = obj_dir.name
                    full_name = _infer_object_name(obj_id)
                    if full_name:
                        yield sd, obj_id, full_name


def _infer_object_name(obj_id: str) -> str | None:
    """Infer the full object_name (e.g. 'microwave_7221') from the asset directory."""
    obj_root = get_repo_root() / "assets" / "object"
    if not obj_root.exists():
        return None
    for type_dir in obj_root.iterdir():
        if type_dir.is_dir() and (type_dir / obj_id).is_dir():
            return f"{type_dir.name.lower()}_{obj_id}"
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare automoma trajectory data.")
    parser.add_argument(
        "--object_name",
        type=str,
        default="all",
        help="Object name (e.g. microwave_7221) or 'all'.",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default="all",
        help="Scene name (e.g. scene_0_seed_0) or 'all'.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test"],
        help="Output mode: produces traj_data_train.pt or traj_data_test.pt.",
    )
    parser.add_argument(
        "--only_successful",
        action="store_true",
        default=True,
        help="Only include successful trajectories (default: True).",
    )
    args = parser.parse_args()

    count = 0
    for scene_dir, obj_id, obj_name_full in iter_targets(args.object_name, args.scene_name):
        print(f"\n{'=' * 60}")
        print(f"Processing: {obj_name_full} / {scene_dir.name}")
        print(f"{'=' * 60}")

        obj_traj_dir = scene_dir / obj_id
        output_dir = OUTPUT_TRAJ_ROOT / obj_name_full / scene_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Merge
        print("  Step 1: Merging grasp trajectories...")
        try:
            merged = merge_grasp_trajectories(obj_traj_dir, only_successful=args.only_successful)
        except FileNotFoundError as e:
            print(f"  {e}")
            continue

        n_total = merged["traj"].shape[0]
        print(f"  Merged {n_total} trajectories ({merged['traj'].shape[1]} steps, 11 DOF)")

        # Save intermediate 11d
        merged_path = output_dir / "traj_data_11d.pt"
        torch.save(merged, merged_path)
        print(f"  Saved: {merged_path}")

        # Step 2: Convert 11D -> 12D
        print("  Step 2: Converting 11D -> 12D...")
        converted = convert_11d_to_12d(merged)

        new_steps = converted["traj_robot"].shape[1]
        print(f"  Converted: {n_total} trajs, {new_steps} steps (11D+gripper -> 12D)")

        # Step 3: Save
        output_name = f"traj_data_{args.mode}.pt"
        output_path = output_dir / output_name
        torch.save(converted, output_path)
        print(f"  Saved: {output_path}")

        # Verification
        g = converted["traj_robot"][0, :, 10]
        print(f"\n  --- Verification (first traj, left gripper) ---")
        print(f"  Step 0: {g[0]:.4f} (expect {GRIPPER_OPEN})")
        print(f"  Step {PREPEND_STEPS - 1}: {g[PREPEND_STEPS - 1]:.4f} (expect ~{GRIPPER_CLOSED})")
        print(f"  Step {PREPEND_STEPS}: {g[PREPEND_STEPS]:.4f} (expect {GRIPPER_CLOSED})")
        print(f"  Last step: {g[-1]:.4f} (expect {GRIPPER_CLOSED})")

        count += 1

    print(f"\nDone. Processed {count} object/scene combination(s).")


if __name__ == "__main__":
    main()
