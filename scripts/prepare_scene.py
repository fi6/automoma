#!/usr/bin/env python3
"""Prepare automoma scene assets: fix metadata rotation and scene Z-offset.

This script processes scenes stored under
``assets/scene/infinigen/kitchen_1130/{scene_name}/`` by applying:
1. **Metadata rotation fix** — Applies a 180° self-rotation around Z
   to every static object's 4×4 matrix, recalculates Euler angles,
   and re-transforms bounding-box corners.
2. **Scene Z-offset fix** — Shifts object positions, matrices,
   bounding-box corners, and the USD ``/World/scene`` prim by a fixed
   Z offset (default −0.12 m).

Usage (from lerobot-arena root):
    # Fix a single scene
    python scripts/prepare_scene.py --scene_name scene_0_seed_0

    # Fix all scenes
    python scripts/prepare_scene.py --scene_name all
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


SCENE_ROOT = get_repo_root() / "assets" / "scene" / "infinigen" / "kitchen_1130"
Z_OFFSET = -0.12


def get_scene_state_path(scene_dir: Path) -> Path:
    return scene_dir / "info" / "prepare_scene_state.json"


def load_scene_state(scene_dir: Path) -> dict:
    state_path = get_scene_state_path(scene_dir)
    if not state_path.exists():
        return {}
    try:
        with open(state_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_scene_state(scene_dir: Path, state: dict) -> None:
    state_path = get_scene_state_path(scene_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def clear_scene_state(scene_dir: Path) -> None:
    state_path = get_scene_state_path(scene_dir)
    if state_path.exists():
        state_path.unlink()


def get_usd_path(scene_dir: Path) -> Path:
    return scene_dir / "export" / "export_scene.blend" / "export_scene.usdc"


def ensure_backup(src: Path, backup: Path) -> bool:
    if not src.exists():
        return False
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, backup)
        print(f"  Backed up {src.name} -> {backup.name}")
    return True


def restore_from_backup(target: Path, backup: Path) -> bool:
    if not backup.exists():
        return False
    shutil.copy2(backup, target)
    print(f"  Restored {target.name} <- {backup.name}")
    return True


# ---------------------------------------------------------------------------
# Rotation fix helpers (from fix_metadata_rotation.py)
# ---------------------------------------------------------------------------

def _single_axis_self_rotation(matrix: np.ndarray, axis: str, angle: float) -> np.ndarray:
    """Post-multiply a single-axis rotation (intrinsic / local)."""
    rot = R.from_euler(axis, angle)
    result = matrix.copy()
    result[:3, :3] = matrix[:3, :3] @ rot.as_matrix()
    return result


def _transform_corners(corners, old_matrix: np.ndarray, new_matrix: np.ndarray):
    if not corners:
        return []
    corners_np = np.array(corners)
    ones = np.ones((corners_np.shape[0], 1))
    homo = np.hstack([corners_np, ones])
    try:
        old_inv = np.linalg.inv(old_matrix)
    except np.linalg.LinAlgError:
        return corners
    local = (old_inv @ homo.T).T
    new = (new_matrix @ local.T).T
    return new[:, :3].tolist()


# ---------------------------------------------------------------------------
# Fix metadata rotation
# ---------------------------------------------------------------------------

def fix_metadata_rotation(metadata_path: Path) -> bool:
    """Apply 180° Z self-rotation to all static objects in metadata."""
    with open(metadata_path, "r") as f:
        data = json.load(f)

    if "static_objects" not in data:
        return True

    for _key, obj_data in data["static_objects"].items():
        old_matrix = np.array(obj_data["matrix"])
        new_matrix = _single_axis_self_rotation(old_matrix, axis="z", angle=np.pi)

        # Recalculate Euler
        upper = new_matrix[:3, :3]
        scale = np.linalg.norm(upper, axis=0)
        pure_rot = upper / scale
        obj_data["rotation"] = R.from_matrix(pure_rot).as_euler("xyz").tolist()

        if "bbox_corners" in obj_data:
            obj_data["bbox_corners"] = _transform_corners(
                obj_data["bbox_corners"], old_matrix, new_matrix
            )
        obj_data["matrix"] = new_matrix.tolist()

    with open(metadata_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  [rotation] Fixed metadata rotation: {metadata_path}")
    return True


# ---------------------------------------------------------------------------
# Fix scene Z-offset in metadata
# ---------------------------------------------------------------------------

def fix_metadata_z_offset(metadata_path: Path) -> bool:
    """Shift static-object positions/matrices/corners by Z_OFFSET in metadata."""
    with open(metadata_path, "r") as f:
        data = json.load(f)

    if "static_objects" not in data:
        return True

    for _key, obj_data in data["static_objects"].items():
        if "position" in obj_data and len(obj_data["position"]) == 3:
            obj_data["position"][2] += Z_OFFSET

        if "matrix" in obj_data:
            m = np.array(obj_data["matrix"])
            if m.shape == (4, 4):
                m[2, 3] += Z_OFFSET
                obj_data["matrix"] = m.tolist()

        if "bbox_corners" in obj_data:
            obj_data["bbox_corners"] = [
                [c[0], c[1], c[2] + Z_OFFSET] if len(c) == 3 else c
                for c in obj_data["bbox_corners"]
            ]

    with open(metadata_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  [z-offset] Fixed metadata Z-offset: {metadata_path}")
    return True


# ---------------------------------------------------------------------------
# Fix scene Z-offset in USD
# ---------------------------------------------------------------------------

def fix_usd_z_offset(scene_dir: Path) -> bool:
    """Apply Z_OFFSET to /World/scene translateOp in the scene USD."""
    usd_path = get_usd_path(scene_dir)
    if not usd_path.exists():
        print(f"  [z-offset] USD file not found, skipping: {usd_path}")
        return False

    from pxr import Gf, Usd, UsdGeom
    try:
        stage = Usd.Stage.Open(str(usd_path))
        if not stage:
            print(f"  [z-offset] Failed to open USD stage: {usd_path}")
            return False

        prim = stage.GetPrimAtPath("/World/scene")
        if not prim:
            print("  [z-offset] /World/scene prim not found, skipping.")
            return False

        xform = UsdGeom.Xformable(prim)
        translate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if not translate_op:
            translate_op = xform.AddTranslateOp()

        cur = translate_op.Get() or Gf.Vec3d(0, 0, 0)
        new = Gf.Vec3d(cur[0], cur[1], cur[2] + Z_OFFSET)
        translate_op.Set(new)
        stage.Save()
        print(f"  [z-offset] USD /World/scene: {cur} -> {new}")
        return True
    except Exception as e:
        print(f"  [z-offset] Error processing USD: {e}")
        return False


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _natural_sort_key(p: Path):
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", p.name)]


def iter_scenes(scene_name: str | None):
    """Yield scene directories matching the given filter."""
    if not SCENE_ROOT.exists():
        print(f"Error: Scene root not found: {SCENE_ROOT}")
        return

    if scene_name and scene_name != "all":
        d = SCENE_ROOT / scene_name
        if d.is_dir():
            yield d
        else:
            print(f"Error: Scene '{scene_name}' not found in {SCENE_ROOT}")
    else:
        for d in sorted(SCENE_ROOT.iterdir(), key=_natural_sort_key):
            if d.is_dir() and d.name.startswith("scene_"):
                yield d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare automoma scene assets.")
    parser.add_argument(
        "--scene_name",
        type=str,
        default="all",
        help="Scene name (e.g. scene_0_seed_0) or 'all'.",
    )
    parser.add_argument(
        "--fix_metadata_rotation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run metadata rotation fix. Use --no-fix_metadata_rotation to disable.",
    )
    parser.add_argument(
        "--fix_metadata_z_offset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run metadata Z-offset fix. Use --no-fix_metadata_z_offset to disable.",
    )
    parser.add_argument(
        "--fix_usd_z_offset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run USD Z-offset fix. Use --no-fix_usd_z_offset to disable.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run selected fixes even if scene state flags already exist.",
    )
    parser.add_argument(
        "--restore_backup",
        action="store_true",
        help="Restore metadata/USD from backup before running selected fixes.",
    )
    parser.add_argument(
        "--restore_only",
        action="store_true",
        help="Only restore metadata/USD from backup and skip all fixes.",
    )
    # Backward-compatible aliases from older script usage.
    parser.add_argument(
        "--skip_rotation",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip_z_offset",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.skip_rotation:
        args.fix_metadata_rotation = False
    if args.skip_z_offset:
        args.fix_metadata_z_offset = False
        args.fix_usd_z_offset = False

    if args.restore_only:
        args.restore_backup = True

    count = 0
    for scene_dir in iter_scenes(args.scene_name):
        print(f"\n{'=' * 60}")
        print(f"Processing: {scene_dir.name}")
        print(f"{'=' * 60}")

        metadata_path = scene_dir / "info" / "metadata.json"
        metadata_backup = scene_dir / "info" / "metadata_backup.json"
        usd_path = get_usd_path(scene_dir)
        usd_backup = usd_path.with_name("export_scene_backup.usdc")

        if not metadata_path.exists():
            print(f"  Skipping — metadata.json not found.")
            continue

        state = load_scene_state(scene_dir)

        if args.restore_backup:
            restored_any = False
            restored_any |= restore_from_backup(metadata_path, metadata_backup)
            restored_any |= restore_from_backup(usd_path, usd_backup)
            if restored_any:
                clear_scene_state(scene_dir)
                state = {}
            else:
                print("  No backup files found to restore.")

        if args.restore_only:
            count += 1
            continue

        if args.fix_metadata_rotation:
            ensure_backup(metadata_path, metadata_backup)
            if state.get("metadata_rotation_fixed", False) and not args.force:
                print("  [rotation] Skipped (already fixed; use --force to rerun)")
            else:
                if fix_metadata_rotation(metadata_path):
                    state["metadata_rotation_fixed"] = True

        if args.fix_metadata_z_offset:
            ensure_backup(metadata_path, metadata_backup)
            if state.get("metadata_z_offset_fixed", False) and not args.force:
                print("  [z-offset] Metadata skipped (already fixed; use --force to rerun)")
            else:
                if fix_metadata_z_offset(metadata_path):
                    state["metadata_z_offset_fixed"] = True

        if args.fix_usd_z_offset:
            if usd_path.exists():
                ensure_backup(usd_path, usd_backup)
            if state.get("usd_z_offset_fixed", False) and not args.force:
                print("  [z-offset] USD skipped (already fixed; use --force to rerun)")
            else:
                if fix_usd_z_offset(scene_dir):
                    state["usd_z_offset_fixed"] = True

        save_scene_state(scene_dir, state)

        count += 1

    print(f"\nDone. Processed {count} scene(s).")


if __name__ == "__main__":
    main()
