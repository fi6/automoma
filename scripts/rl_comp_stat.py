#!/usr/bin/env python
"""Compute per-joint min/max from sampled trajectories for a given stage."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import matplotlib.pyplot as plt

JOINT_NAMES: List[str] = [
    "root_x_axis_joint",
    "root_y_axis_joint",
    "root_z_rotation_joint",
    "torso_lift_joint",
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "r_gripper_finger_joint",
    "l_gripper_finger_joint",
]


def _select_traj_tensor(stage_name: str, traj_data: Dict) -> torch.Tensor:
    if stage_name == "open_stage" and "open_traj" in traj_data:
        return traj_data["open_traj"]
    if stage_name == "reach_stage" and "reach_traj" in traj_data:
        return traj_data["reach_traj"]
    if "traj" in traj_data:
        return traj_data["traj"]
    if "open_traj" in traj_data:
        return traj_data["open_traj"]
    if "reach_traj" in traj_data:
        return traj_data["reach_traj"]
    raise KeyError("No trajectory tensor found in traj_data.pt")


def _normalize_traj_dim(traj: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
    if traj.dim() != 3:
        raise ValueError(f"Expected traj dim [N, T, D], got {traj.shape}")

    expected_with_object = len(JOINT_NAMES) + 1
    if traj.shape[2] >= expected_with_object:
        return traj[:, :, :expected_with_object], JOINT_NAMES + ["object_joint"]
    if traj.shape[2] == len(JOINT_NAMES):
        # Pad object_joint with NaN
        pad = torch.full((traj.shape[0], traj.shape[1], 1), float("nan"), dtype=traj.dtype)
        return torch.cat([traj, pad], dim=2), JOINT_NAMES + ["object_joint"]

    # If fewer, pad with NaNs to match expected size
    pad_cols = expected_with_object - traj.shape[2]
    pad = torch.full((traj.shape[0], traj.shape[1], pad_cols), float("nan"), dtype=traj.dtype)
    return torch.cat([traj, pad], dim=2), JOINT_NAMES + ["object_joint"]


def _sample_traj(traj: torch.Tensor, sample_size: int, seed: int) -> torch.Tensor:
    n = traj.shape[0]
    if n == 0:
        return traj
    sample_size = min(sample_size, n)
    random.seed(seed)
    indices = random.sample(range(n), sample_size)
    return traj[indices]


def _compute_min_max(traj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if traj.numel() == 0:
        return torch.empty((traj.shape[2],)), torch.empty((traj.shape[2],))
    nan_mask = torch.isnan(traj)
    traj_min = torch.where(nan_mask, torch.tensor(float("inf"), dtype=traj.dtype), traj)
    traj_max = torch.where(nan_mask, torch.tensor(float("-inf"), dtype=traj.dtype), traj)

    traj_min = traj_min.min(dim=1).values
    traj_min = traj_min.min(dim=0).values
    traj_max = traj_max.max(dim=1).values
    traj_max = traj_max.max(dim=0).values

    traj_min = torch.where(torch.isinf(traj_min), torch.tensor(float("nan"), dtype=traj.dtype), traj_min)
    traj_max = torch.where(torch.isinf(traj_max), torch.tensor(float("nan"), dtype=traj.dtype), traj_max)
    return traj_min, traj_max


def _plot_min_max(output_path: Path, labels: List[str], mins: torch.Tensor, maxs: torch.Tensor) -> None:
    x = list(range(len(labels)))
    plt.figure(figsize=(12, 5))
    plt.plot(x, mins.cpu().numpy(), label="min", marker="o")
    plt.plot(x, maxs.cpu().numpy(), label="max", marker="o")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("joint")
    plt.ylabel("value")
    plt.title("Trajectory min/max per joint")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    script_stem = Path(__file__).stem

    parser = argparse.ArgumentParser(description="Compute joint min/max from trajectories.")
    parser.add_argument(
        "--stage",
        choices=["open_stage", "reach_stage"],
        default="reach_stage",
        help="Stage name to process (default: infer from script name).",
    )
    parser.add_argument(
        "--traj-root",
        default="output/collect_0130/traj/fetch",
        help="Root directory containing scene trajectories.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/rl_comp",
        help="Output directory for CSV and plots.",
    )
    parser.add_argument("--sample-size", type=int, default=100, help="Number of trajectories to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")

    args = parser.parse_args()
    stage_name = args.stage or (script_stem if script_stem in {"open_stage", "reach_stage"} else "open_stage")

    traj_root = Path(args.traj_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{stage_name}_traj_minmax.csv"
    plot_path = output_dir / f"{stage_name}_traj_minmax.png"

    rows: List[List] = []
    plot_mins: List[torch.Tensor] = []
    plot_maxs: List[torch.Tensor] = []

    for scene_dir in sorted(traj_root.glob("scene_*")):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name

        object_dirs = [d for d in scene_dir.iterdir() if d.is_dir()]
        for object_dir in sorted(object_dirs):
            grasp_dir = object_dir / "grasp_0000" / stage_name
            traj_path = grasp_dir / "traj_data.pt"
            if not traj_path.exists():
                continue

            traj_data = torch.load(traj_path, map_location="cpu", weights_only=False)
            traj = _select_traj_tensor(stage_name, traj_data)
            traj = traj.float()

            traj, labels = _normalize_traj_dim(traj)
            sampled = _sample_traj(traj, args.sample_size, args.seed)
            traj_min, traj_max = _compute_min_max(sampled)

            rows.append([scene_id, stage_name, sampled.shape[0], "min", *traj_min.tolist()])
            rows.append([scene_id, stage_name, sampled.shape[0], "max", *traj_max.tolist()])

            if traj_min.numel() == len(labels):
                plot_mins.append(traj_min)
                plot_maxs.append(traj_max)

    if not rows:
        print(f"No trajectories found under {traj_root} for stage {stage_name}")
        return

    header = ["scene_id", "stage_name", "traj_num", "type", *labels]
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    if plot_mins and plot_maxs:
        stacked_min = torch.stack(plot_mins, dim=0)
        stacked_max = torch.stack(plot_maxs, dim=0)
        mean_min = torch.nanmean(stacked_min, dim=0)
        mean_max = torch.nanmean(stacked_max, dim=0)
        _plot_min_max(plot_path, labels, mean_min, mean_max)

    print(f"CSV saved to: {csv_path}")
    if plot_path.exists():
        print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
