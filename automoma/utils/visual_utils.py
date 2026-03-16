# Copyright (c) 2024-2025, AutoMoMa Authors. All rights reserved.
# SPDX-License-Identifier: MIT
"""Visualization utilities for debugging collision worlds and voxel grids."""

from __future__ import annotations


def visualize_voxel_grid_with_cuboid(
    cuboid,
    voxel_grid,
    threshold=0.0,
    mesh_obstacle=None,
    color_outside=(0, 0, 255, 255),       # Blue, opaque
    color_inside=(0, 0, 255, 255),         # Blue, opaque (same = no highlight)
    cuboid_outline_color=(0, 255, 0, 255), # Green
    show_cuboid_outline=True,
    show_mesh=True,
):
    """Visualize the voxel grid using trimesh, highlighting occupied voxels.

    Only occupied voxels are shown for clarity.

    Args:
        cuboid: The cuboid region to highlight (cuRobo ``Cuboid``).
        voxel_grid: The voxel grid to visualize (cuRobo ``VoxelGrid``).
        threshold: Threshold to determine occupied voxels.
        mesh_obstacle: Optional ``Mesh`` obstacle to render.
        color_outside: RGBA for occupied voxels *outside* cuboid.
        color_inside: RGBA for occupied voxels *inside* cuboid.
        cuboid_outline_color: RGBA for cuboid outline wireframe.
        show_cuboid_outline: Whether to show cuboid outline.
        show_mesh: Whether to show ``mesh_obstacle``.
    """
    import numpy as np
    import trimesh

    if voxel_grid.feature_tensor is None or voxel_grid.xyzr_tensor is None:
        raise ValueError("feature_tensor and xyzr_tensor must be initialized.")

    # Extract all voxel coordinates and features
    points_xyz = voxel_grid.xyzr_tensor[:, :3].cpu().numpy()
    features = voxel_grid.feature_tensor.cpu().numpy()
    occupied_mask = features > threshold
    occupied_points = points_xyz[occupied_mask]

    if len(occupied_points) == 0:
        print("No voxels above threshold.")
        return

    # Cuboid centre and dimensions
    center = np.array(cuboid.pose[:3])
    dims = np.array(cuboid.dims)
    half_dims = dims / 2.0

    # Cuboid rotation (quaternion [qw, qx, qy, qz])
    rot_matrix = np.eye(3)
    if len(cuboid.pose) == 7:
        from scipy.spatial.transform import Rotation as Rot
        quat = cuboid.pose[3:]  # [qw, qx, qy, qz]
        rot_matrix = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

    # Transform occupied points to cuboid local frame
    local_points = occupied_points - center
    local_points = local_points @ rot_matrix

    # Determine if inside cuboid
    inside_cuboid = np.all(
        (local_points >= -half_dims) & (local_points <= half_dims), axis=1
    )

    # Assign colours
    colors = np.tile(color_outside, (occupied_points.shape[0], 1))
    colors[inside_cuboid] = color_inside

    # Create trimesh PointCloud for occupied voxels
    point_cloud = trimesh.points.PointCloud(occupied_points, colors=colors)

    scene_objects = [point_cloud]

    # Cuboid outline (wireframe box)
    if show_cuboid_outline:
        transform = np.eye(4)
        transform[:3, 3] = center
        if len(cuboid.pose) == 7:
            transform[:3, :3] = rot_matrix
        box_primitive = trimesh.primitives.Box(extents=dims, transform=transform)
        box_lines = trimesh.load_path(box_primitive.vertices[box_primitive.edges_unique])
        box_lines.colors = [cuboid_outline_color] * len(box_lines.entities)
        scene_objects.append(box_lines)

    # Mesh obstacle
    if show_mesh and mesh_obstacle is not None:
        try:
            mesh_trimesh = mesh_obstacle.get_trimesh_mesh().copy()
            mesh_transform = np.eye(4)
            mesh_transform[:3, 3] = mesh_obstacle.pose[:3]
            if len(mesh_obstacle.pose) == 7:
                from scipy.spatial.transform import Rotation as Rot
                mesh_quat = mesh_obstacle.pose[3:]
                mesh_rot = Rot.from_quat(
                    [mesh_quat[1], mesh_quat[2], mesh_quat[3], mesh_quat[0]]
                ).as_matrix()
                mesh_transform[:3, :3] = mesh_rot
            mesh_trimesh.apply_transform(mesh_transform)
            scene_objects.append(mesh_trimesh)
        except Exception as e:
            print(f"Warning: Could not load or add mesh obstacle: {e}")

    # Create scene and show
    scene = trimesh.Scene(scene_objects)
    try:
        scene.background = [30, 30, 30, 255]
    except Exception:
        pass
    try:
        scene.show()
    except Exception as e:
        import time
        import os
        print(f"Could not open window for visualization ({type(e).__name__}: {e}).")
        filename = f"collision_viz_{int(time.time())}.glb"
        try:
            scene.export(filename)
            print(f"Saved 3D visualization to {os.path.abspath(filename)}")
            print("You can view this file in a 3D viewer (e.g., https://gltf-viewer.donmccurdy.com/)")
        except Exception as e2:
            print(f"Failed to export visualization: {e2}")
