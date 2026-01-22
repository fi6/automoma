#!/usr/bin/env python3
"""
Convert ACRONYM grasp poses (HDF5) to Automoma .npy format.

Output format per file:
  [x, y, z, qw, qx, qy, qz]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np


def _rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert a 3x3 rotation matrix to quaternion (qw, qx, qy, qz).
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    return float(qw), float(qx), float(qy), float(qz)


def _matrix_to_pose(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 transform to [x, y, z, qw, qx, qy, qz]."""
    xyz = T[:3, 3]
    R = T[:3, :3]
    qw, qx, qy, qz = _rotation_matrix_to_quaternion(R)
    return np.array([xyz[0], xyz[1], xyz[2], qw, qx, qy, qz], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert ACRONYM H5 grasps to Automoma npy format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--h5",
        required=True,
        help="Path to ACRONYM grasp H5 file",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for .npy files",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Only export successful grasps",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Optional max number of grasps to export",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Visualize each grasp and prompt to save (Y/N)",
    )
    parser.add_argument(
        "--mesh-root",
        default=None,
        help="Mesh root for visualization (defaults to ACRONYM data root)",
    )

    args = parser.parse_args()

    h5_path = Path(args.h5).expanduser().resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 not found: {h5_path}")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure acronym_tools is importable
    project_root = Path(__file__).resolve().parents[2]
    acronym_root = project_root / "third_party" / "acronym"
    sys.path.insert(0, str(acronym_root))

    from acronym_tools import load_grasps  # type: ignore

    if args.interactive:
        import trimesh  # type: ignore
        from acronym_tools import load_mesh, create_gripper_marker  # type: ignore

    T_all, success = load_grasps(str(h5_path))

    if args.success_only:
        indices = np.where(success == 1)[0]
    else:
        indices = np.arange(len(T_all))

    if args.max is not None:
        indices = indices[: args.max]

    saved_count = 0
    for idx in indices:
        if args.interactive:
            mesh_root_dir = (
                Path(args.mesh_root).expanduser().resolve()
                if args.mesh_root
                else (acronym_root / "data" / "examples")
            )
            obj_mesh = load_mesh(str(h5_path), mesh_root_dir=str(mesh_root_dir))
            grasp_marker = create_gripper_marker(color=[0, 255, 0]).apply_transform(T_all[idx])
            trimesh.Scene([obj_mesh, grasp_marker]).show()

            user_input = input("Save this grasp? [Y/N] (default: N): ").strip().lower()
            if user_input != "y":
                continue

        pose = _matrix_to_pose(T_all[idx])
        np.save(out_dir / f"{saved_count:04d}.npy", pose)
        saved_count += 1

    print(f"Saved {saved_count} grasps to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
