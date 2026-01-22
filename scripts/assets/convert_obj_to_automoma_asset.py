#!/usr/bin/env python3
"""
Convert a single OBJ + grasp pose into an Automoma asset folder.

Creates:
  assets/object/<asset_type>/<asset_id>/
    textured_objs/<obj, mtl, textures>
    mobility.urdf
    grasp/0000.npy

Grasp pose format: [x, y, z, qw, qx, qy, qz]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np


def _read_text_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _parse_mtllibs(obj_path: Path) -> List[str]:
    mtllibs: List[str] = []
    for line in _read_text_lines(obj_path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("mtllib "):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                mtllibs.append(parts[1].strip())
    return mtllibs


def _parse_mtl_textures(mtl_path: Path) -> Set[str]:
    textures: Set[str] = set()
    for line in _read_text_lines(mtl_path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Common texture keys
        if line.lower().startswith((
            "map_kd ", "map_ka ", "map_ks ", "map_ns ", "map_d ",
            "map_bump ", "bump ", "disp ", "decal ", "refl ",
        )):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                tex = parts[1].strip()
                # Some exporters include options like "-s 1 1 1 tex.png"
                if tex.startswith("-"):
                    tex_parts = tex.split()
                    tex = tex_parts[-1]
                textures.add(tex)
    return textures


def _copy_with_parents(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_obj_bundle(obj_path: Path, dst_dir: Path) -> Tuple[Path, List[Path]]:
    """
    Copy OBJ + referenced MTL + textures into dst_dir.
    Returns (copied_obj_path, copied_extra_paths).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []

    copied_obj = dst_dir / obj_path.name
    _copy_with_parents(obj_path, copied_obj)
    copied.append(copied_obj)

    # Copy mtllib and textures if present
    for mtl_rel in _parse_mtllibs(obj_path):
        mtl_path = (obj_path.parent / mtl_rel).resolve()
        if not mtl_path.exists():
            continue
        copied_mtl = dst_dir / mtl_path.name
        _copy_with_parents(mtl_path, copied_mtl)
        copied.append(copied_mtl)

        for tex_rel in _parse_mtl_textures(mtl_path):
            tex_path = (mtl_path.parent / tex_rel).resolve()
            if tex_path.exists():
                copied_tex = dst_dir / tex_path.name
                _copy_with_parents(tex_path, copied_tex)
                copied.append(copied_tex)

    return copied_obj, copied


def _write_minimal_urdf(urdf_path: Path, mesh_rel_path: str) -> None:
    urdf_text = f"""<?xml version=\"1.0\" ?>
<robot name=\"mug_asset\">
  <link name=\"base\">
    <visual>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry>
        <mesh filename=\"{mesh_rel_path}\"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry>
        <mesh filename=\"{mesh_rel_path}\"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    urdf_path.write_text(urdf_text, encoding="utf-8")


def _save_grasp(grasp_path: Path, grasp: np.ndarray) -> None:
    grasp_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(grasp_path, grasp)


def _load_grasp_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.grasp_npy:
        grasp = np.load(args.grasp_npy, allow_pickle=True)
        return np.array(grasp, dtype=np.float64)
    if args.grasp is None or len(args.grasp) != 7:
        raise ValueError("Provide --grasp with 7 floats or --grasp-npy")
    return np.array(args.grasp, dtype=np.float64)


def _validate_output(asset_dir: Path, mesh_rel: str) -> None:
    urdf_path = asset_dir / "mobility.urdf"
    grasp_path = asset_dir / "grasp" / "0000.npy"

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if not grasp_path.exists():
        raise FileNotFoundError(f"Grasp not found: {grasp_path}")

    mesh_path = asset_dir / mesh_rel
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    grasp = np.load(grasp_path, allow_pickle=True)
    if grasp.shape != (7,):
        raise ValueError(f"Grasp shape is {grasp.shape}, expected (7,)")

    # Basic quaternion norm check (avoid strict failure if zero)
    qw, qx, qy, qz = grasp[3:7]
    norm = float(np.linalg.norm([qw, qx, qy, qz]))
    if norm < 1e-6:
        raise ValueError("Quaternion norm is too small; check grasp pose format")

    print("Validation OK:")
    print(f"- URDF: {urdf_path}")
    print(f"- Mesh: {mesh_path}")
    print(f"- Grasp: {grasp_path} (norm={norm:.6f})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert OBJ + grasp pose into Automoma asset folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--obj", required=True, help="Path to mug OBJ file")
    parser.add_argument(
        "--asset-type",
        default="Mug",
        help="Asset type folder name under assets/object",
    )
    parser.add_argument(
        "--asset-id",
        default="1000",
        help="Asset ID folder name under assets/object/<asset_type>",
    )
    parser.add_argument(
        "--out-root",
        default="assets/object",
        help="Output root under project (relative path)",
    )
    parser.add_argument(
        "--grasp",
        type=float,
        nargs=7,
        help="Grasp pose [x y z qw qx qy qz]",
    )
    parser.add_argument(
        "--grasp-npy",
        help="Path to .npy file with grasp pose [x y z qw qx qy qz]",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output files after conversion",
    )

    args = parser.parse_args()

    obj_path = Path(args.obj).expanduser().resolve()
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ not found: {obj_path}")

    grasp = _load_grasp_from_args(args)
    if grasp.shape != (7,):
        raise ValueError("Grasp must be 7 elements: [x, y, z, qw, qx, qy, qz]")

    project_root = Path(__file__).resolve().parents[2]
    out_root = (project_root / args.out_root).resolve()
    asset_dir = out_root / args.asset_type / args.asset_id
    textured_dir = asset_dir / "textured_objs"

    copied_obj, copied_files = _copy_obj_bundle(obj_path, textured_dir)
    mesh_rel = str((Path("assets") / "object" / args.asset_type / args.asset_id / "textured_objs" / copied_obj.name))

    urdf_path = asset_dir / "mobility.urdf"
    _write_minimal_urdf(urdf_path, mesh_rel)

    grasp_path = asset_dir / "grasp" / "0000.npy"
    _save_grasp(grasp_path, grasp)

    print("Conversion complete:")
    print(f"- Asset dir: {asset_dir}")
    print(f"- URDF: {urdf_path}")
    print(f"- Mesh: {mesh_rel}")
    print(f"- Grasp: {grasp_path}")
    if copied_files:
        print("- Copied files:")
        for p in copied_files:
            print(f"  - {p}")

    if args.validate:
        _validate_output(asset_dir, mesh_rel)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
