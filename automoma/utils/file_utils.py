# Copyright (c) 2024-2025, AutoMoMa Authors. All rights reserved.
# SPDX-License-Identifier: MIT
"""File I/O utilities for configs, IK/traj data, grasp poses, and scene metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from curobo.util_file import load_yaml

from automoma.core.types import IKResult, TrajResult
from automoma.utils.math_utils import matrix_to_pose, single_axis_self_rotation


# ============================================================================
# Path helpers
# ============================================================================

def get_project_dir() -> str:
    """Return the project root (two levels up from this file)."""
    return str(Path(__file__).resolve().parent.parent.parent)


def get_abs_path(rel_path: str) -> str:
    return os.path.join(get_project_dir(), rel_path)


# ============================================================================
# Robot config
# ============================================================================

def process_robot_cfg(robot_cfg: Dict) -> Dict:
    """Resolve relative paths inside a cuRobo robot-config dict."""
    root = get_project_dir()
    for key in ("urdf_path", "external_asset_path", "asset_root_path"):
        val = robot_cfg.get("kinematics", {}).get(key, "")
        if val:
            robot_cfg["kinematics"][key] = os.path.join(root, val)
    return robot_cfg


def load_robot_cfg(robot_cfg_path: Union[str, Dict]) -> Dict[str, Any]:
    """Load a cuRobo robot YAML and return the ``robot_cfg`` section."""
    if isinstance(robot_cfg_path, dict):
        return robot_cfg_path
    loaded = load_yaml(robot_cfg_path)["robot_cfg"]
    print(f"Robot configuration loaded from {robot_cfg_path}")
    return loaded


# ============================================================================
# IK / Trajectory persistence
# ============================================================================

def save_ik(ik_result: IKResult, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"target_poses": ik_result.target_poses, "iks": ik_result.iks},
        path,
    )
    print(f"IK data saved to {path}")


def load_ik(path: str) -> IKResult:
    data = torch.load(path, weights_only=False)
    print(f"IK data loaded from {path}")
    if "target_poses" in data and "iks" in data:
        return IKResult(target_poses=data["target_poses"], iks=data["iks"])
    # Legacy format
    iks = data.get("start_iks", data.get("iks"))
    return IKResult(
        target_poses=data.get("target_poses", torch.zeros(iks.shape[0], 7)),
        iks=iks,
    )


def save_traj(traj_result: TrajResult, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "start_states": traj_result.start_states.cpu(),
            "goal_states": traj_result.goal_states.cpu(),
            "trajectories": traj_result.trajectories.cpu(),
            "success": traj_result.success.cpu(),
        },
        path,
    )
    print(f"Trajectory data saved to {path}")


def load_traj(path: str) -> TrajResult:
    data = torch.load(path, weights_only=False)
    print(f"Trajectory data loaded from {path}")
    return TrajResult.from_dict(data)


# ============================================================================
# Grasp poses
# ============================================================================

def get_grasp_poses(
    grasp_dir: str,
    num_grasps: int = 20,
    scaling_factor: float = 1.0,
) -> List[np.ndarray]:
    """Load and scale grasp poses from ``{grasp_dir}/XXXX.npy``."""
    poses: List[np.ndarray] = []
    for i in range(num_grasps):
        fp = os.path.join(grasp_dir, f"{i:04d}.npy")
        if os.path.exists(fp):
            g = np.load(fp).copy()
            g[:3] *= scaling_factor
            poses.append(g)
    print(f"Loaded {len(poses)} grasp poses from {grasp_dir}")
    return poses


# ============================================================================
# Scene / object metadata
# ============================================================================

def load_object_from_metadata(
    metadata_path: str,
    object_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Populate *object_cfg* with pose/dimensions from scene metadata JSON.

    Applies the π-rotation about Z that aligns Infinigen coordinates with
    cuRobo conventions (same as V1's ``_process_object_pose``).
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    target_type = object_cfg.get("asset_type")
    target_id = object_cfg.get("asset_id")

    obj_info = None
    for v in metadata["static_objects"].values():
        if v.get("asset_type") == target_type and v.get("asset_id") == target_id:
            obj_info = v
            break

    if obj_info is None:
        raise ValueError(
            f"Object {target_type}/{target_id} not found in {metadata_path}"
        )

    mat = np.array(obj_info["matrix"])
    mat = single_axis_self_rotation(mat, axis="z", angle=np.pi)
    pose_7d = matrix_to_pose(mat).tolist()

    object_cfg.update(
        {
            "name": obj_info["name"],
            "asset_type": obj_info.get("asset_type", target_type),
            "asset_id": obj_info.get("asset_id", target_id),
            "dimensions": obj_info["dimensions"],
            "pose": pose_7d,
        }
    )
    print(
        f"Object loaded: {obj_info['name']} dims={obj_info['dimensions']}"
    )
    return object_cfg
