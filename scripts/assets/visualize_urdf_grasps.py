#!/usr/bin/env python3
"""
Visualize an object URDF and grasp poses one by one.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import trimesh


def _quat_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Quaternion (qw, qx, qy, qz) to 3x3 rotation matrix."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """[x, y, z, qw, qx, qy, qz] -> 4x4 transform."""
    x, y, z, qw, qx, qy, qz = pose.tolist()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_to_matrix(qw, qx, qy, qz)
    T[:3, 3] = [x, y, z]
    return T


def _parse_urdf_mesh(urdf_path: Path) -> Tuple[Path, Optional[np.ndarray]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    mesh_elem = root.find(".//mesh")
    if mesh_elem is None:
        raise ValueError("No <mesh> found in URDF")
    filename = mesh_elem.attrib.get("filename")
    if not filename:
        raise ValueError("<mesh> has no filename")

    scale_attr = mesh_elem.attrib.get("scale")
    scale = None
    if scale_attr:
        parts = scale_attr.split()
        if len(parts) == 3:
            scale = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
    return Path(filename), scale


def _load_mesh(urdf_path: Path) -> Tuple[trimesh.Trimesh, np.ndarray]:
    mesh_rel, scale = _parse_urdf_mesh(urdf_path)
    project_root = urdf_path.resolve().parents[4]
    mesh_path = (project_root / mesh_rel).resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    if scale is None:
        scale_vec = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    else:
        scale_vec = scale
        mesh.apply_scale(scale_vec)
    return mesh, scale_vec


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize URDF + grasp poses one by one",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--urdf",
        required=True,
        help="Path to object URDF",
    )
    parser.add_argument(
        "--grasp-dir",
        required=True,
        help="Directory with grasp .npy files",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max number of grasps to show",
    )

    args = parser.parse_args()

    urdf_path = Path(args.urdf).expanduser().resolve()
    grasp_dir = Path(args.grasp_dir).expanduser().resolve()

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if not grasp_dir.exists():
        raise FileNotFoundError(f"Grasp dir not found: {grasp_dir}")

    print(f"Loading URDF: {urdf_path}")
    mesh, scale_vec = _load_mesh(urdf_path)

    # Ensure acronym_tools is importable
    project_root = Path(__file__).resolve().parents[2]
    acronym_root = project_root / "third_party" / "acronym"
    sys.path.insert(0, str(acronym_root))

    # Lazy import to avoid dependency if not needed elsewhere
    from acronym_tools import create_gripper_marker  # type: ignore

    grasp_files = sorted(grasp_dir.glob("*.npy"))
    if args.max is not None:
        grasp_files = grasp_files[: args.max]

    for grasp_file in grasp_files:
        pose = np.load(grasp_file, allow_pickle=True)
        if pose.shape != (7,):
            print(f"Skip {grasp_file.name}: invalid shape {pose.shape}")
            continue

        T = _pose_to_matrix(pose)
        # Scale translation to match URDF mesh scaling
        T[:3, 3] = T[:3, 3] * scale_vec

        marker = create_gripper_marker(color=[0, 255, 0])
        marker.apply_scale(float(np.mean(scale_vec)))
        marker = marker.apply_transform(T)
        trimesh.Scene([mesh, marker]).show()
        input(f"Showing {grasp_file.name}. Press Enter for next...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# visualize_urdf_grasps.py --urdf /home/xinhai/projects/automoma/assets/object/Mug/1000/mobility.urdf --grasp-dir /home/xinhai/projects/automoma/assets/object/Mug/1000/grasp