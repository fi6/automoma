# Copyright (c) 2024-2025, AutoMoMa Authors. All rights reserved.
# SPDX-License-Identifier: MIT
"""Math utilities for pose manipulation, IK clustering, and collision helpers.

All functions that were previously imported from ``cuakr.utils.math`` and
``cuakr.utils.voxel`` are consolidated here so the planner has **zero**
external cuakr dependencies.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# cuRobo types (always available in the planning env)
from curobo.types.math import Pose
from curobo.geom.types import Cuboid


# ============================================================================
# Pose / quaternion helpers
# ============================================================================

def _convert_to_list(x: Union[List[float], np.ndarray, torch.Tensor]) -> List[float]:
    """Reliably convert data into a plain Python list."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def pose_multiply(p1, p2):
    """Multiply two 7-D poses ``[x, y, z, qw, qx, qy, qz]``.

    The result type matches the type of *p2* (list / ndarray / Tensor).
    """
    original_type = type(p2)
    device = getattr(p2, "device", None)
    dtype = getattr(p2, "dtype", None)

    r = Pose.from_list(_convert_to_list(p1)).multiply(
        Pose.from_list(_convert_to_list(p2))
    ).to_list()

    if original_type is torch.Tensor:
        return torch.as_tensor(r, device=device, dtype=dtype)
    if original_type is np.ndarray:
        return np.array(r, dtype=dtype)
    return r


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions in ``[w, x, y, z]`` order."""
    q1, q2 = np.asarray(q1), np.asarray(q2)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
    ], axis=-1)


def quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Shortest angular distance (radians) between two ``[w,x,y,z]`` quats."""
    q1 = np.squeeze(q1) / (np.linalg.norm(np.squeeze(q1)) + 1e-8)
    q2 = np.squeeze(q2) / (np.linalg.norm(np.squeeze(q2)) + 1e-8)
    dot = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
    angle = 2.0 * np.arccos(dot)
    return angle if angle <= np.pi else 2 * np.pi - angle


# ============================================================================
# Matrix ↔ pose conversions
# ============================================================================

def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    """4×4 matrix → 7-D pose ``[x, y, z, qw, qx, qy, qz]``."""
    assert matrix.shape == (4, 4)
    quat_xyzw = R.from_matrix(matrix[:3, :3]).as_quat()  # scipy uses xyzw
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return np.concatenate([matrix[:3, 3], quat_wxyz])


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """7-D pose → 4×4 matrix."""
    assert pose.shape == (7,)
    quat_xyzw = np.array([pose[4], pose[5], pose[6], pose[3]])
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    mat[:3, 3] = pose[:3]
    return mat


def single_axis_self_rotation(matrix: np.ndarray, axis: str, angle: float) -> np.ndarray:
    """Rotate a 4×4 transform about a local axis."""
    result = matrix.copy()
    result[:3, :3] = matrix[:3, :3] @ R.from_euler(axis, angle).as_matrix()
    return result


# ============================================================================
# IK helpers
# ============================================================================

def expand_to_pairs(
    start_iks: torch.Tensor, goal_iks: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cartesian product: ``(N, D)`` × ``(M, D)`` → ``(N*M, D)`` × ``(N*M, D)``."""
    goal_exp = goal_iks.repeat(start_iks.shape[0], 1).clone()
    start_exp = torch.repeat_interleave(start_iks, goal_iks.shape[0], dim=0).clone()
    return start_exp, goal_exp


def stack_iks_angle(iks: torch.Tensor, angle: float) -> torch.Tensor:
    """Append a scalar joint angle column to IK solutions."""
    col = torch.full((iks.shape[0], 1), angle, device=iks.device, dtype=iks.dtype)
    return torch.cat([iks, col], dim=1)


# ============================================================================
# get_open_ee_pose  (standalone util — V1 correct logic)
# ============================================================================

def get_open_ee_pose(
    object_pose: Pose,
    grasp_pose: Pose,
    object_urdf,
    handle_link: str,
    joint_cfg: Dict[str, float],
    default_joint_cfg: Dict[str, float],
) -> Pose:
    """Compute the end-effector pose after articulating the object.

    Transform chain:
        ``T_world_ee = T_world_object × T_object_handle_new × T_handle_grasp``

    This follows V1's correct implementation: reset to the *default* config
    first, compute the grasp-to-handle offset, then re-apply the target
    joint config to obtain the updated grasp pose in world coordinates.

    Args:
        object_pose: Object base frame in world coordinates.
        grasp_pose:  Grasp pose relative to the *object base* in the default
                     joint configuration.
        object_urdf: ``yourdfpy.URDF`` instance.
        handle_link: Name of the handle / end-effector link in the URDF.
        joint_cfg:   Target joint configuration (e.g. ``{"joint_0": 1.57}``).
        default_joint_cfg: Default joint configuration (e.g. ``{"joint_0": 0.0}``).

    Returns:
        End-effector ``Pose`` in world coordinates.
    """
    # 1. Ensure URDF is in the default (closed) state
    object_urdf.update_cfg(default_joint_cfg)
    T_obj_handle_default = Pose.from_matrix(object_urdf.get_transform(handle_link, "world"))

    # 2. Grasp relative to handle in the default config
    T_handle_grasp = T_obj_handle_default.inverse().multiply(grasp_pose)

    # 3. Move to target config and get the new handle pose
    object_urdf.update_cfg(joint_cfg)
    T_obj_handle_new = Pose.from_matrix(object_urdf.get_transform(handle_link, "world"))

    # 4. New grasp in object frame
    T_obj_grasp_new = T_obj_handle_new.multiply(T_handle_grasp)

    # 5. Transform to world coordinates
    return object_pose.multiply(T_obj_grasp_new)


# ============================================================================
# IK clustering  (V1 KMeans → AP fallback, returns mask indices)
# ============================================================================

def ik_clustering(
    all_iks: torch.Tensor,
    *,
    kmeans_clusters: int = 500,
    ap_fallback_clusters: int = 30,
    ap_clusters_upperbound: int = 80,
    ap_clusters_lowerbound: int = 10,
) -> np.ndarray:
    """Cluster IK solutions using KMeans → Affinity-Propagation fallback.

    Returns a **boolean mask** over the original ``all_iks`` rows so that the
    caller can save all solutions and mark selected ones.

    All hyper-parameters are passed explicitly from the YAML config.
    """
    from sklearn.cluster import AffinityPropagation, KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    from sklearn.metrics.pairwise import cosine_similarity

    emb = all_iks.detach().cpu().numpy()
    n = emb.shape[0]

    # --- Stage 1: KMeans pre-filter ---
    if n <= kmeans_clusters:
        km_indices = np.arange(n)
    else:
        km = KMeans(n_clusters=kmeans_clusters, random_state=0, n_init=10).fit(emb)
        km_indices = pairwise_distances_argmin_min(km.cluster_centers_, emb)[0]

    emb_reduced = emb[km_indices]

    # --- Stage 2: AP (or KMeans fallback) on reduced set ---
    if emb_reduced.shape[0] <= ap_fallback_clusters:
        ap_local_indices = np.arange(emb_reduced.shape[0])
    else:
        af = AffinityPropagation(affinity="precomputed", damping=0.9, random_state=0)
        sim = cosine_similarity(emb_reduced)
        af.fit(sim)
        labels = af.labels_
        n_unique = len(np.unique(labels))

        if n_unique > ap_clusters_upperbound or n_unique < ap_clusters_lowerbound:
            km2 = KMeans(
                n_clusters=min(ap_fallback_clusters, emb_reduced.shape[0]),
                random_state=0,
                n_init=10,
            ).fit(emb_reduced)
            labels = km2.labels_

        # Pick median from each cluster
        ap_local_indices = []
        for lbl in np.unique(labels):
            cluster_idx = np.where(labels == lbl)[0]
            order = emb_reduced[cluster_idx, 0].argsort()
            median = cluster_idx[order[len(order) // 2]]
            ap_local_indices.append(median)
        ap_local_indices = np.array(ap_local_indices)

    # Map back to original indices
    selected_indices = km_indices[ap_local_indices]

    # Build boolean mask
    mask = np.zeros(n, dtype=bool)
    mask[selected_indices] = True
    return mask


# ============================================================================
# Collision helper
# ============================================================================

def mark_cuboid_as_empty(esdf, cuboid: Cuboid, empty_value: float | None = None):
    """Mark voxel-grid points inside *cuboid* as free space.

    Modifies ``esdf.feature_tensor`` in-place and returns the updated grid.
    """
    if esdf.feature_tensor is None or esdf.xyzr_tensor is None:
        raise ValueError("feature_tensor and xyzr_tensor must be initialised.")

    pts = esdf.xyzr_tensor[:, :3].cpu().numpy()
    center = np.array(cuboid.pose[:3])
    half = np.array(cuboid.dims) / 2.0

    rot = np.eye(3)
    if len(cuboid.pose) == 7:
        q = cuboid.pose[3:]  # [qw, qx, qy, qz]
        rot = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

    local = (pts - center) @ rot
    mask = np.all((local >= -half) & (local <= half), axis=1)

    if empty_value is None:
        empty_value = max(esdf.feature_tensor.min().item(), -1.0)
    esdf.feature_tensor[mask] = empty_value
    return esdf
