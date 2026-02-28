#!/usr/bin/env python3
"""Prepare automoma object assets: fix URDF paths and convert URDF to USD.

This script processes objects stored under ``assets/object/{Type}/{id}/`` by:
1. Fixing URDF / mesh filenames that contain hyphens (incompatible with Isaac Sim 5.1).
2. Optionally converting the fixed URDF to USD via IsaacLab's UrdfConverter.

Usage (from lerobot-arena root):
    # Fix URDF + convert for a single object
    python scripts/prepare_object.py --object_type Microwave --object_id 7221
    python scripts/prepare_object.py --object_type Dishwasher --object_id 11622

    # Fix URDF only (no Isaac Sim required)
    python scripts/prepare_object.py --object_type Microwave --object_id 7221 --fix_only

    # Process all objects of a given type
    python scripts/prepare_object.py --object_type Microwave --object_id all

    # Process every object
    python scripts/prepare_object.py --object_type all
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def get_repo_root() -> Path:
    """Return the repository root (parent of ``scripts/``)."""
    return Path(__file__).resolve().parent.parent


OBJECT_ROOT = get_repo_root() / "assets" / "object"


# ---------------------------------------------------------------------------
# URDF / mesh fixing helpers (adapted from fix_urdf_path_510.py)
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Replace hyphens with underscores."""
    return name.replace("-", "_") if name else name


def _fix_mesh_files(mesh_dir: Path) -> None:
    """Rename mesh files and fix internal .mtl references."""
    if not mesh_dir.exists():
        return

    # Rename files with hyphens
    for f in list(mesh_dir.iterdir()):
        if "-" in f.name:
            new_name = _sanitize(f.name)
            f.rename(mesh_dir / new_name)
            print(f"  Renamed: {f.name} -> {new_name}")

    # Fix mtllib references inside .obj files
    for obj_file in mesh_dir.glob("*.obj"):
        try:
            lines = obj_file.read_text(encoding="utf-8", errors="ignore").splitlines(True)
            modified = False
            new_lines = []
            for line in lines:
                if line.strip().startswith("mtllib"):
                    parts = line.strip().split()
                    if len(parts) >= 2 and "-" in parts[1]:
                        line = f"mtllib {_sanitize(parts[1])}\n"
                        modified = True
                new_lines.append(line)
            if modified:
                obj_file.write_text("".join(new_lines), encoding="utf-8")
                print(f"  Fixed mtllib in: {obj_file.name}")
        except Exception as e:
            print(f"  Warning: could not fix {obj_file.name}: {e}")


def _fix_urdf(urdf_path: Path) -> Path:
    """Fix hyphen characters in URDF names / mesh paths.  Returns the fixed URDF path."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for elem in root.iter():
        if "name" in elem.attrib and "-" in elem.attrib["name"]:
            elem.attrib["name"] = _sanitize(elem.attrib["name"])

    for mesh in root.iter("mesh"):
        fname = mesh.attrib.get("filename", "")
        head, tail = os.path.split(fname)
        if "-" in tail:
            mesh.attrib["filename"] = os.path.join(head, _sanitize(tail))

    # Overwrite in-place (backup first)
    backup = urdf_path.with_suffix(".urdf.bak")
    if not backup.exists():
        shutil.copy2(urdf_path, backup)
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    print(f"  Fixed URDF: {urdf_path}")
    return urdf_path


def fix_object(object_dir: Path) -> Path:
    """Fix mesh files and URDF for a single object directory.

    Returns the path to the fixed URDF.
    """
    urdf_path = object_dir / "mobility.urdf"
    if not urdf_path.exists():
        print(f"  Skipping {object_dir.name}: mobility.urdf not found")
        return None

    # Fix mesh directories (textured_objs and any other mesh dirs referenced in URDF)
    for mesh_dir_name in ("textured_objs", "new_mesh"):
        _fix_mesh_files(object_dir / mesh_dir_name)

    return _fix_urdf(urdf_path)


# ---------------------------------------------------------------------------
# USD conversion via IsaacLab UrdfConverter
# ---------------------------------------------------------------------------

def convert_urdf_to_usd(
    urdf_path: Path,
    output_dir: Path,
    fix_base: bool = True,
    joint_stiffness: float = 0.0,
    joint_damping: float = 0.1,
    joint_drive_type: str = "force",
    joint_target_type: str = "position",
    collision_from_visuals: bool = True,
    collider_type: str = "convex_decomposition",
    replace_cylinders_with_capsules: bool = True,
    merge_fixed_joints: bool = False,
) -> None:
    """Convert URDF to USD using IsaacLab's UrdfConverter.

    Requires Isaac Sim / IsaacLab runtime.
    """
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(output_dir),
        usd_file_name="mobility.usd",
        fix_base=fix_base,
        merge_fixed_joints=merge_fixed_joints,
        force_usd_conversion=True,
        make_instanceable=False,
        collision_from_visuals=collision_from_visuals,
        collider_type=collider_type,
        replace_cylinders_with_capsules=replace_cylinders_with_capsules,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type=joint_drive_type,
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=joint_stiffness,
                damping=joint_damping,
            ),
            target_type=joint_target_type,
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"  Converted: {urdf_path} -> {output_dir / 'mobility.usd'}")


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def iter_objects(object_type: str | None, object_id: str | None):
    """Yield (type_dir, id_dir) tuples matching the given filters."""
    if not OBJECT_ROOT.exists():
        print(f"Error: Object root not found: {OBJECT_ROOT}")
        return

    type_dirs = sorted(OBJECT_ROOT.iterdir()) if object_type == "all" or object_type is None else []
    if object_type and object_type != "all":
        # Case-insensitive match
        for d in OBJECT_ROOT.iterdir():
            if d.is_dir() and d.name.lower() == object_type.lower():
                type_dirs = [d]
                break
        else:
            print(f"Error: Object type '{object_type}' not found in {OBJECT_ROOT}")
            return

    for type_dir in type_dirs:
        if not type_dir.is_dir():
            continue
        if object_id and object_id != "all":
            id_dir = type_dir / object_id
            if id_dir.is_dir():
                yield type_dir, id_dir
        else:
            for id_dir in sorted(type_dir.iterdir()):
                if id_dir.is_dir() and id_dir.name.isdigit():
                    yield type_dir, id_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare automoma object assets.")
    parser.add_argument(
        "--object_type",
        type=str,
        default="all",
        help="Object type (e.g. Microwave, Dishwasher) or 'all'.",
    )
    parser.add_argument(
        "--object_id",
        type=str,
        default="all",
        help="Object id (e.g. 7221) or 'all'.",
    )
    parser.add_argument(
        "--fix_only",
        action="store_true",
        help="Only fix URDF/mesh files; skip USD conversion.",
    )
    parser.add_argument(
        "--fix_base",
        default=True,
        help="Fix the base link during URDF->USD conversion.",
    )
    parser.add_argument(
        "--merge_fixed_joints",
        action="store_true",
        default=False,
        help="Merge links connected by fixed joints during conversion.",
    )
    parser.add_argument(
        "--joint_stiffness",
        type=float,
        default=0.0,
        help="Joint drive stiffness for URDF->USD conversion.",
    )
    parser.add_argument(
        "--joint_damping",
        type=float,
        default=0.1,
        help="Joint drive damping for URDF->USD conversion.",
    )
    parser.add_argument(
        "--joint_target_type",
        type=str,
        default="position",
        choices=["position", "velocity", "none"],
        help="Joint drive target type for URDF->USD conversion.",
    )
    parser.add_argument(
        "--joint_drive_type",
        type=str,
        default="force",
        choices=["force", "acceleration"],
        help="Joint drive type for URDF->USD conversion.",
    )
    parser.add_argument(
        "--collision_from_visuals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create collision geometry from visual geometry.",
    )
    parser.add_argument(
        "--collider_type",
        type=str,
        default="convex_decomposition",
        choices=["convex_hull", "convex_decomposition"],
        help="Collider simplification type for URDF->USD conversion.",
    )
    parser.add_argument(
        "--replace_cylinders_with_capsules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace cylinder collision shapes with capsules.",
    )

    args = parser.parse_args()

    # If not fix_only, we need IsaacLab runtime for USD conversion
    if not args.fix_only:
        try:
            from isaaclab.app import AppLauncher

            launcher = AppLauncher(headless=True)
            _sim_app = launcher.app
        except ImportError:
            print(
                "Error: IsaacLab not found.  Use --fix_only to skip USD conversion, "
                "or run in an environment with IsaacLab installed."
            )
            sys.exit(1)

    count = 0
    for type_dir, id_dir in iter_objects(args.object_type, args.object_id):
        print(f"\n{'=' * 60}")
        print(f"Processing: {type_dir.name}/{id_dir.name}")
        print(f"{'=' * 60}")

        urdf_path = fix_object(id_dir)
        if urdf_path is None:
            continue

        if not args.fix_only:
            usd_output_dir = id_dir / "mobility"
            try:
                convert_urdf_to_usd(
                    urdf_path=urdf_path,
                    output_dir=usd_output_dir,
                    fix_base=args.fix_base,
                    joint_stiffness=args.joint_stiffness,
                    joint_damping=args.joint_damping,
                    joint_drive_type=args.joint_drive_type,
                    joint_target_type=args.joint_target_type,
                    collision_from_visuals=args.collision_from_visuals,
                    collider_type=args.collider_type,
                    replace_cylinders_with_capsules=args.replace_cylinders_with_capsules,
                    merge_fixed_joints=args.merge_fixed_joints,
                )
            except Exception as e:
                msg = str(e)
                print(f"  Error converting URDF->USD for {type_dir.name}/{id_dir.name}: {e}")
                if "CXXABI_1.3.15" in msg or "libstdc++.so.6" in msg:
                    print("  Hint: this is a libstdc++ runtime mismatch in the current conda/env setup.")
                    print(
                        "  Try: conda install -n <env> -c conda-forge 'libstdcxx-ng>=13' 'libgcc-ng>=13'"
                    )
                continue

        count += 1

    print(f"\nDone. Processed {count} object(s).")

    # Close sim if it was started
    if not args.fix_only:
        try:
            _sim_app.close()  # noqa: F821
        except Exception:
            pass


if __name__ == "__main__":
    main()
