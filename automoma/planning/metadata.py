# Copyright (c) 2024-2025, AutoMoMa Authors. All rights reserved.
# SPDX-License-Identifier: MIT
"""Alignment helpers for per-trajectory planning metadata."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


METADATA_TENSOR_KEYS = ("grasp_id", "grasp_pose", "goal_angle")
METADATA_SCALAR_KEYS = ("scene_id", "object_id")
METADATA_KEYS = METADATA_TENSOR_KEYS + METADATA_SCALAR_KEYS


def make_planning_metadata(
    count: int,
    *,
    grasp_id: int,
    grasp_pose: Sequence[float] | torch.Tensor,
    goal_angle: float,
    scene_id: str,
    object_id: str,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Create metadata rows for one planner result batch."""
    pose = torch.as_tensor(grasp_pose, dtype=torch.float32, device=device)
    if pose.shape != (7,):
        raise ValueError(f"grasp_pose must have shape [7], got {tuple(pose.shape)}")
    return {
        "grasp_id": torch.full((count,), int(grasp_id), dtype=torch.int64, device=device),
        "grasp_pose": pose.unsqueeze(0).repeat(count, 1),
        "goal_angle": torch.full((count,), float(goal_angle), dtype=torch.float32, device=device),
        "scene_id": str(scene_id),
        "object_id": str(object_id),
    }


def validate_planning_metadata(
    payload: Mapping[str, Any], count: int, *, label: str = "trajectory payload"
) -> None:
    missing = [key for key in METADATA_KEYS if key not in payload]
    if missing:
        raise ValueError(
            f"{label} is missing aligned planning metadata {missing}; "
            "backfill it from uniquely matched per-grasp results before reuse"
        )
    expected_shapes = {"grasp_id": (count,), "grasp_pose": (count, 7), "goal_angle": (count,)}
    for key, shape in expected_shapes.items():
        value = payload[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label}:{key} must be a torch.Tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{label}:{key} has shape {tuple(value.shape)}, expected {shape}")
    for key in METADATA_SCALAR_KEYS:
        if not isinstance(payload[key], str) or not payload[key]:
            raise TypeError(f"{label}:{key} must be a non-empty string")


def planning_metadata(payload: Mapping[str, Any], count: int, *, label: str = "trajectory payload") -> dict[str, Any]:
    validate_planning_metadata(payload, count, label=label)
    return {key: payload[key] for key in METADATA_KEYS}


def index_planning_metadata(metadata: Mapping[str, Any], index: Any) -> dict[str, Any]:
    return {
        **{key: metadata[key][index] for key in METADATA_TENSOR_KEYS},
        **{key: metadata[key] for key in METADATA_SCALAR_KEYS},
    }


def cat_planning_metadata(items: Sequence[Mapping[str, Any]], *, label: str = "metadata merge") -> dict[str, Any]:
    if not items:
        raise ValueError(f"{label}: no metadata batches")
    scene_id = items[0]["scene_id"]
    object_id = items[0]["object_id"]
    for item in items:
        if item["scene_id"] != scene_id or item["object_id"] != object_id:
            raise ValueError(
                f"{label}: scalar metadata mismatch "
                f"({item['scene_id']!r}, {item['object_id']!r}) != ({scene_id!r}, {object_id!r})"
            )
    return {
        **{
            key: torch.cat([item[key].to(items[0][key].device) for item in items], dim=0)
            for key in METADATA_TENSOR_KEYS
        },
        "scene_id": scene_id,
        "object_id": object_id,
    }
