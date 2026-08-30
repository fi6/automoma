# Copyright (c) 2024-2025, AutoMoMa Authors. All rights reserved.
# SPDX-License-Identifier: MIT
"""Lightweight helpers for append-safe planning artifact writes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch

from automoma.core.types import IKResult, TrajResult
from automoma.planning.metadata import (
    METADATA_KEYS,
    cat_planning_metadata,
    planning_metadata,
    validate_planning_metadata,
)


class PlanningIO:
    """Persist planning artifacts without overwriting existing results.

    Round 1 writes directly. Later rounds load the existing file, validate
    compatibility, append along the sample dimension, and atomically replace it.
    """

    def save_ik(self, ik_result: IKResult, path: str) -> IKResult:
        merged = ik_result
        if os.path.exists(path):
            payload = torch.load(path, weights_only=False)
            existing = IKResult(target_poses=payload["target_poses"], iks=payload["iks"])
            merged = self._merge_ik(existing, ik_result, path)
        self._atomic_save(
            {
                "target_poses": merged.target_poses.cpu(),
                "iks": merged.iks.cpu(),
            },
            path,
        )
        print(f"IK data saved to {path} ({merged.iks.shape[0]} total)")
        return merged

    def save_traj(self, traj_result: TrajResult, path: str) -> TrajResult:
        merged = traj_result
        if os.path.exists(path):
            merged = self._merge_traj(
                TrajResult.from_dict(torch.load(path, weights_only=False)), traj_result, path
            )
        self._atomic_save(
            {
                "start_states": merged.start_states.cpu(),
                "goal_states": merged.goal_states.cpu(),
                "trajectories": merged.trajectories.cpu(),
                "success": merged.success.cpu(),
            },
            path,
        )
        print(f"Trajectory data saved to {path} ({merged.success.shape[0]} total)")
        return merged

    def save_traj_with_metadata(
        self,
        traj_result: TrajResult,
        metadata: Dict[str, Any],
        path: str,
    ) -> tuple[TrajResult, dict[str, Any]]:
        """Append a raw trajectory batch and its exactly aligned metadata."""
        validate_planning_metadata(metadata, traj_result.num_samples, label=f"new {path}")
        merged_traj = traj_result
        merged_metadata = metadata
        if os.path.exists(path):
            existing_payload = torch.load(path, weights_only=False)
            existing_traj = TrajResult.from_dict(existing_payload)
            existing_metadata = planning_metadata(
                existing_payload, existing_traj.num_samples, label=f"existing {path}"
            )
            merged_traj = self._merge_traj(existing_traj, traj_result, path)
            merged_metadata = cat_planning_metadata(
                [existing_metadata, metadata], label=f"append metadata at {path}"
            )
        payload = {
            "start_states": merged_traj.start_states.cpu(),
            "goal_states": merged_traj.goal_states.cpu(),
            "trajectories": merged_traj.trajectories.cpu(),
            "success": merged_traj.success.cpu(),
            **{key: merged_metadata[key].cpu() for key in ("grasp_id", "grasp_pose", "goal_angle")},
            "scene_id": merged_metadata["scene_id"],
            "object_id": merged_metadata["object_id"],
        }
        self._atomic_save(payload, path)
        print(f"Trajectory data saved to {path} ({merged_traj.success.shape[0]} total)")
        return merged_traj, merged_metadata

    def save_converted(
        self, payload: Dict[str, Any], path: str, *, append: bool = True
    ) -> dict[str, Any]:
        merged = payload
        if append and os.path.exists(path):
            merged = self._merge_payload(torch.load(path, weights_only=False), payload, path)
        self._atomic_save(
            {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in merged.items()},
            path,
        )
        total = int(merged["traj_success"].shape[0]) if "traj_success" in merged else 0
        print(f"Converted trajectory data saved to {path} ({total} total)")
        return merged

    def _merge_ik(self, existing: IKResult, new: IKResult, path: str) -> IKResult:
        if existing.iks.shape[0] == 0:
            return new
        if new.iks.shape[0] == 0:
            return existing
        self._require_same_rank(existing.target_poses, new.target_poses, path, "target_poses")
        self._require_same_shape(existing.target_poses, new.target_poses, path, "target_poses")
        self._require_same_rank(existing.iks, new.iks, path, "iks")
        self._require_same_shape(existing.iks, new.iks, path, "iks")
        self._require_same_dtype(existing.target_poses, new.target_poses, path, "target_poses")
        self._require_same_dtype(existing.iks, new.iks, path, "iks")
        return IKResult.cat([existing, new])

    def _merge_traj(self, existing: TrajResult, new: TrajResult, path: str) -> TrajResult:
        if existing.success.shape[0] == 0:
            return new
        if new.success.shape[0] == 0:
            return existing
        self._require_same_rank(existing.start_states, new.start_states, path, "start_states")
        self._require_same_shape(existing.start_states, new.start_states, path, "start_states")
        self._require_same_rank(existing.goal_states, new.goal_states, path, "goal_states")
        self._require_same_shape(existing.goal_states, new.goal_states, path, "goal_states")
        self._require_same_rank(existing.trajectories, new.trajectories, path, "trajectories")
        self._require_same_shape(existing.trajectories, new.trajectories, path, "trajectories")
        self._require_same_rank(existing.success, new.success, path, "success")
        self._require_same_shape(existing.success, new.success, path, "success")
        self._require_same_dtype(existing.start_states, new.start_states, path, "start_states")
        self._require_same_dtype(existing.goal_states, new.goal_states, path, "goal_states")
        self._require_same_dtype(existing.trajectories, new.trajectories, path, "trajectories")
        self._require_same_dtype(existing.success, new.success, path, "success")
        return TrajResult.cat([existing, new])

    def _merge_payload(
        self,
        existing: Dict[str, Any],
        new: Dict[str, Any],
        path: str,
    ) -> dict[str, Any]:
        tensor_keys = [
            "start_robot",
            "start_obj",
            "goal_robot",
            "goal_obj",
            "traj_robot",
            "traj_obj",
            "traj_success",
            "grasp_id",
            "grasp_pose",
            "goal_angle",
        ]
        keys = tensor_keys + list(METADATA_KEYS[-2:])
        missing = [key for key in keys if key not in existing or key not in new]
        if missing:
            raise ValueError(f"Cannot append converted payload at {path}: missing keys {missing}")

        if all(existing[key].shape[0] == 0 for key in tensor_keys):
            return new
        if all(new[key].shape[0] == 0 for key in tensor_keys):
            return existing

        for key in ("scene_id", "object_id"):
            if existing[key] != new[key]:
                raise ValueError(
                    f"Cannot append converted payload at {path}: {key} mismatch "
                    f"{existing[key]!r} != {new[key]!r}"
                )
        merged: dict[str, Any] = {}
        for key in tensor_keys:
            existing_tensor = existing[key]
            new_tensor = new[key]
            if not isinstance(existing_tensor, torch.Tensor) or not isinstance(new_tensor, torch.Tensor):
                raise TypeError(f"Cannot append converted payload at {path}: key '{key}' must be a tensor")
            self._require_same_rank(existing_tensor, new_tensor, path, key)
            self._require_same_shape(existing_tensor, new_tensor, path, key)
            self._require_same_dtype(existing_tensor, new_tensor, path, key)
            merged[key] = torch.cat([existing_tensor, new_tensor], dim=0)
        merged["scene_id"] = existing["scene_id"]
        merged["object_id"] = existing["object_id"]
        return merged

    def _require_same_rank(self, existing: torch.Tensor, new: torch.Tensor, path: str, key: str) -> None:
        if existing.ndim != new.ndim:
            raise ValueError(
                f"Cannot append {key} at {path}: rank mismatch {existing.ndim} != {new.ndim}"
            )

    def _require_same_shape(self, existing: torch.Tensor, new: torch.Tensor, path: str, key: str) -> None:
        if existing.shape[1:] != new.shape[1:]:
            raise ValueError(
                f"Cannot append {key} at {path}: shape mismatch {tuple(existing.shape)} != {tuple(new.shape)}"
            )

    def _require_same_dtype(self, existing: torch.Tensor, new: torch.Tensor, path: str, key: str) -> None:
        if existing.dtype != new.dtype:
            raise ValueError(
                f"Cannot append {key} at {path}: dtype mismatch {existing.dtype} != {new.dtype}"
            )

    def _atomic_save(self, payload: Dict[str, Any], path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, target)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
