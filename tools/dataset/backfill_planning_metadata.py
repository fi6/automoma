#!/usr/bin/env python3
"""Strictly backfill canonical planning metadata from per-grasp results.

Every canonical row must exactly match one and only one converted source row.
The command refuses ambiguous or missing matches and never infers identity from
directory iteration order.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


TRAJ_KEYS = (
    "start_robot",
    "start_obj",
    "goal_robot",
    "goal_obj",
    "traj_robot",
    "traj_obj",
    "traj_success",
)


def convert_raw(raw: dict[str, Any], *, prepend_steps: int, gripper_open: float, gripper_closed: float) -> dict:
    start = raw["start_states"]
    goal = raw["goal_states"]
    trajectory = raw["trajectories"]
    count, steps, _ = trajectory.shape

    def state(value: torch.Tensor, grip: float) -> tuple[torch.Tensor, torch.Tensor]:
        gripper = torch.full((value.shape[0], 2), grip, dtype=value.dtype)
        return torch.cat([value[:, :10], gripper], dim=-1), -value[:, 10:]

    start_robot, start_obj = state(start, gripper_open)
    goal_robot, goal_obj = state(goal, gripper_closed)
    grasp_arm = trajectory[:, :1, :10].repeat(1, prepend_steps, 1)
    closing = torch.linspace(gripper_open, gripper_closed, prepend_steps, dtype=trajectory.dtype)
    grasp_gripper = closing.reshape(1, prepend_steps, 1).repeat(count, 1, 2)
    pull_gripper = torch.full((count, steps, 2), gripper_closed, dtype=trajectory.dtype)
    return {
        "start_robot": start_robot,
        "start_obj": start_obj,
        "goal_robot": goal_robot,
        "goal_obj": goal_obj,
        "traj_robot": torch.cat(
            [torch.cat([grasp_arm, grasp_gripper], dim=-1), torch.cat([trajectory[:, :, :10], pull_gripper], dim=-1)],
            dim=1,
        ),
        "traj_obj": torch.cat([-trajectory[:, :1, 10:].repeat(1, prepend_steps, 1), -trajectory[:, :, 10:]], dim=1),
        "traj_success": raw["success"],
    }


def source_rows(
    source_root: Path,
    grasp_pose_dir: Path,
    *,
    scale: float,
    scene_id: str,
    object_id: str,
    prepend_steps: int = 4,
    gripper_open: float = 0.04,
    gripper_closed: float = 0.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(source_root.glob("grasp_*/traj_data.pt")):
        try:
            grasp_id = int(path.parent.name.split("_", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"Cannot parse grasp id from {path}") from error
        raw = torch.load(path, map_location="cpu", weights_only=False)
        converted = convert_raw(
            raw,
            prepend_steps=prepend_steps,
            gripper_open=gripper_open,
            gripper_closed=gripper_closed,
        )
        count = int(converted["traj_success"].shape[0])
        if all(key in raw for key in ("grasp_id", "grasp_pose", "goal_angle")):
            grasp_ids = raw["grasp_id"]
            grasp_poses = raw["grasp_pose"]
            goal_angles = raw["goal_angle"]
        else:
            pose_path = grasp_pose_dir / f"{grasp_id:04d}.npy"
            if not pose_path.is_file():
                raise FileNotFoundError(f"Missing grasp pose for {path}: {pose_path}")
            pose = np.load(pose_path).copy()
            pose[:3] *= scale
            grasp_ids = torch.full((count,), grasp_id, dtype=torch.int64)
            grasp_poses = torch.as_tensor(pose, dtype=torch.float32).reshape(1, 7).repeat(count, 1)
            # This comes from the source result itself: the planner appends the
            # negative articulated goal angle as its last state column.
            goal_angles = -raw["goal_states"][:, -1].to(dtype=torch.float32)
        for index in range(count):
            rows.append(
                {
                    "source": f"{path}:{index}",
                    **{key: converted[key][index] for key in TRAJ_KEYS},
                    "grasp_id": grasp_ids[index],
                    "grasp_pose": grasp_poses[index],
                    "goal_angle": goal_angles[index],
                    "scene_id": scene_id,
                    "object_id": object_id,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No grasp_*/traj_data.pt files under {source_root}")
    return rows


def _row_matches(canonical: dict[str, Any], index: int, candidate: dict[str, Any]) -> bool:
    return all(torch.equal(canonical[key][index].cpu(), candidate[key].cpu()) for key in TRAJ_KEYS)


def backfill(canonical_path: Path, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    canonical = torch.load(canonical_path, map_location="cpu", weights_only=False)
    missing = [key for key in TRAJ_KEYS if key not in canonical]
    if missing:
        raise ValueError(f"{canonical_path} is missing trajectory keys {missing}")
    count = int(canonical["traj_robot"].shape[0])
    matched: list[dict[str, Any]] = []
    for index in range(count):
        matches = [candidate for candidate in candidates if _row_matches(canonical, index, candidate)]
        if len(matches) != 1:
            sources = [match["source"] for match in matches]
            raise ValueError(
                f"Canonical trajectory {index} has {len(matches)} exact per-grasp matches; "
                f"expected exactly one. Matches: {sources}"
            )
        matched.append(matches[0])

    scalar_pairs = {(row["scene_id"], row["object_id"]) for row in matched}
    if len(scalar_pairs) != 1:
        raise ValueError(f"Matched source rows disagree on scene/object identity: {scalar_pairs}")
    scene_id, object_id = scalar_pairs.pop()
    canonical.update(
        {
            "grasp_id": torch.stack([row["grasp_id"] for row in matched]).to(dtype=torch.int64),
            "grasp_pose": torch.stack([row["grasp_pose"] for row in matched]).to(dtype=torch.float32),
            "goal_angle": torch.stack([row["goal_angle"] for row in matched]).to(dtype=torch.float32),
            "scene_id": scene_id,
            "object_id": object_id,
        }
    )
    with tempfile.NamedTemporaryFile(
        dir=canonical_path.parent, prefix=f".{canonical_path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(canonical, temporary)
        os.replace(temporary, canonical_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--grasp-pose-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--object-id", required=True)
    args = parser.parse_args()
    candidates = source_rows(
        args.source_root,
        args.grasp_pose_dir,
        scale=args.scale,
        scene_id=args.scene_id,
        object_id=args.object_id,
    )
    payload = backfill(args.canonical, candidates)
    print(f"Backfilled {payload['traj_robot'].shape[0]} uniquely matched rows in {args.canonical}")


if __name__ == "__main__":
    main()
