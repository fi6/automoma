#!/usr/bin/env python3
"""Compute dimensions and bbox corners for a URDF visual mesh and update metadata.json."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

try:
    import trimesh
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "Missing dependency: trimesh. Install with: pip install trimesh"
    ) from exc


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def quat_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = w * w + x * x + y * y + z * z
    if n == 0.0:
        return np.eye(3)
    s = 2.0 / n
    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=float,
    )


def quat_to_rpy(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    # XYZ (roll, pitch, yaw)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def parse_origin(origin: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.zeros(3, dtype=float)
    rpy = np.zeros(3, dtype=float)
    if origin is None:
        return xyz, rpy

    if "xyz" in origin.attrib:
        xyz = np.array([float(v) for v in origin.attrib["xyz"].split()], dtype=float)
    if "rpy" in origin.attrib:
        rpy = np.array([float(v) for v in origin.attrib["rpy"].split()], dtype=float)
    return xyz, rpy


def load_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    return mesh


def load_visual_meshes(urdf_path: str) -> list[trimesh.Trimesh]:
    base_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    meshes: list[trimesh.Trimesh] = []

    for visual in root.findall(".//visual"):
        geom = visual.find("geometry")
        if geom is None:
            continue
        mesh_elem = geom.find("mesh")
        if mesh_elem is None:
            continue
        filename = mesh_elem.attrib.get("filename")
        if not filename:
            continue

        mesh_path = os.path.join(base_dir, filename)
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        mesh = load_mesh(mesh_path)
        scale = mesh_elem.attrib.get("scale")
        if scale:
            scale_vals = [float(v) for v in scale.split()]
            mesh.apply_scale(scale_vals)

        origin = visual.find("origin")
        xyz, rpy = parse_origin(origin)
        rotation = rpy_to_matrix(rpy[0], rpy[1], rpy[2])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = xyz
        mesh.apply_transform(transform)
        meshes.append(mesh)

    return meshes


def compute_bounds(meshes: list[trimesh.Trimesh]) -> tuple[np.ndarray, np.ndarray]:
    if not meshes:
        raise ValueError("No visual meshes found in URDF.")

    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)

    for mesh in meshes:
        bounds = mesh.bounds
        mins = np.minimum(mins, bounds[0])
        maxs = np.maximum(maxs, bounds[1])

    return mins, maxs


def build_pose_matrix(position: np.ndarray, quat: np.ndarray) -> np.ndarray:
    rotation = quat_to_matrix(quat[0], quat[1], quat[2], quat[3])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


def bbox_corners_from_bounds(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    corners = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [maxs[0], maxs[1], mins[2]],
        ],
        dtype=float,
    )
    return corners


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    hom = np.hstack([points, np.ones((points.shape[0], 1), dtype=float)])
    transformed = (matrix @ hom.T).T
    return transformed[:, :3]


def update_metadata(
    metadata_path: str,
    object_key: str,
    position: np.ndarray,
    quat: np.ndarray,
    dimensions: np.ndarray,
    bbox_corners: np.ndarray,
) -> None:
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if "static_objects" not in metadata:
        metadata["static_objects"] = {}
    if object_key not in metadata["static_objects"]:
        metadata["static_objects"][object_key] = {"name": object_key}

    roll, pitch, yaw = quat_to_rpy(quat[0], quat[1], quat[2], quat[3])
    pose_matrix = build_pose_matrix(position, quat)

    obj = metadata["static_objects"][object_key]
    obj["matrix"] = pose_matrix.tolist()
    obj["position"] = position.tolist()
    obj["rotation"] = [roll, pitch, yaw]
    obj.setdefault("scale", [1.0, 1.0, 1.0])
    obj["dimensions"] = dimensions.tolist()
    obj["bbox_corners"] = bbox_corners.tolist()

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True, help="Path to URDF file")
    parser.add_argument(
        "--pose",
        nargs=7,
        type=float,
        required=True,
        metavar=("X", "Y", "Z", "W", "QX", "QY", "QZ"),
        help="Pose as x y z w qx qy qz",
    )
    parser.add_argument("--metadata", required=True, help="Path to metadata.json to update")
    parser.add_argument("--object-key", required=True, help="Static object key in metadata.json")

    args = parser.parse_args()

    position = np.array(args.pose[:3], dtype=float)
    quat = np.array(args.pose[3:], dtype=float)

    meshes = load_visual_meshes(args.urdf)
    mins, maxs = compute_bounds(meshes)
    dimensions = maxs - mins

    local_corners = bbox_corners_from_bounds(mins, maxs)
    world_matrix = build_pose_matrix(position, quat)
    world_corners = transform_points(local_corners, world_matrix)

    update_metadata(args.metadata, args.object_key, position, quat, dimensions, world_corners)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# python scripts/compute_urdf_metadata.py --urdf assets/object/Refrigerator/10000/10000_0_scaling.urdf --pose 0.5361366205337185 -4.457830299261277 1.0164388733467529 0.70711 0.70711 0.0 0.0 --metadata assets/scene/mshab/kitchen_0130/scene_0_seed_0/info/metadata.json --object-key "StaticCategoryFactory(Refrigerator_10000_0_scaling_mobility)"