#!/usr/bin/env python3
"""Measure AKR microwave-body (link_1) drift over planned trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType

from automoma.utils.file_utils import load_robot_cfg, process_robot_cfg


PERCENTILES = (50, 90, 95, 99, 100)
POSITION_FILTERS_M = (0.01, 0.005, 0.002, 0.001, 0.0005)
ROTATION_FILTERS_RAD = (0.05, 0.04, 0.03, 0.02, 0.01)


def summarize(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    out: dict[str, float | int] = {"count": int(arr.size), "mean": float(arr.mean())}
    for percentile in PERCENTILES:
        key = "max" if percentile == 100 else f"p{percentile}"
        out[key] = float(np.percentile(arr, percentile))
    return out


def quaternion_angle(reference: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    reference = torch.nn.functional.normalize(reference, dim=-1)
    current = torch.nn.functional.normalize(current, dim=-1)
    dot = torch.sum(reference * current, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def filter_retention(metrics: dict[str, list[float]]) -> dict[str, object]:
    position = np.asarray(metrics["trajectory_max_position_m"], dtype=np.float64)
    rotation = np.asarray(metrics["trajectory_max_rotation_rad"], dtype=np.float64)
    total = int(position.size)

    def count(mask: np.ndarray) -> dict[str, float | int]:
        kept = int(mask.sum())
        return {
            "kept": kept,
            "total": total,
            "retention": float(kept / total) if total else 0.0,
        }

    return {
        "position_only": {
            f"{threshold:g}": count(position < threshold)
            for threshold in POSITION_FILTERS_M
        },
        "rotation_only": {
            f"{threshold:g}": count(rotation < threshold)
            for threshold in ROTATION_FILTERS_RAD
        },
        "paired": {
            f"position<{position_threshold:g},rotation<{rotation_threshold:g}": count(
                (position < position_threshold) & (rotation < rotation_threshold)
            )
            for position_threshold, rotation_threshold in zip(
                POSITION_FILTERS_M, ROTATION_FILTERS_RAD, strict=True
            )
        },
    }


def build_model(config_path: Path, tensor_args: TensorDeviceType) -> CudaRobotModel:
    robot_cfg = process_robot_cfg(load_robot_cfg(str(config_path)))
    model_cfg = CudaRobotModelConfig.from_data_dict(
        robot_cfg["kinematics"], tensor_args=tensor_args
    )
    return CudaRobotModel(model_cfg)


def analyze_grasp(
    path: Path,
    config_path: Path,
    tensor_args: TensorDeviceType,
    chunk_size: int,
) -> tuple[dict[str, object], dict[str, list[float]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    success = payload["success"].bool()
    trajectories = payload["trajectories"][success].float()
    goals = payload["goal_states"][success].float()
    model = build_model(config_path, tensor_args)

    metrics = {
        "waypoint_position_m": [],
        "waypoint_rotation_rad": [],
        "trajectory_max_position_m": [],
        "trajectory_max_rotation_rad": [],
        "terminal_position_m": [],
        "terminal_rotation_rad": [],
        "terminal_object_joint": [],
        "base_yaw_max_from_start_rad": [],
        "base_yaw_total_variation_rad": [],
        "base_yaw_max_step_rad": [],
        "base_xy_max_from_start_m": [],
        "base_xy_path_length_m": [],
        "arm_max_step_rad": [],
        "arm_max_second_difference_rad": [],
        "arm_max_third_difference_rad": [],
    }
    goal_angle_counts: dict[str, int] = {}

    for start in range(0, trajectories.shape[0], chunk_size):
        traj = trajectories[start : start + chunk_size].to(tensor_args.device)
        goal = goals[start : start + chunk_size].to(tensor_args.device)
        batch, steps, dof = traj.shape
        flat = traj.reshape(batch * steps, dof)
        state = model.get_state(flat, link_name="link_1")
        goal_state = model.get_state(goal, link_name="link_1")
        position = state.ee_position.reshape(batch, steps, 3)
        quaternion = state.ee_quaternion.reshape(batch, steps, 4)
        goal_position = goal_state.ee_position[:, None, :]
        goal_quaternion = goal_state.ee_quaternion[:, None, :]

        position_error = torch.linalg.vector_norm(position - goal_position, dim=-1)
        rotation_error = quaternion_angle(goal_quaternion.expand_as(quaternion), quaternion)
        object_joint_error = torch.abs(goal[:, -1] - traj[:, -1, -1])

        # AKR planning states are [base_x, base_y, base_yaw, arm(7), object_joint].
        # Wrap yaw increments before accumulating so a +/-pi crossing is not counted
        # as a full revolution.
        base_xy = traj[:, :, :2]
        yaw_step = torch.diff(traj[:, :, 2], dim=1)
        yaw_step = torch.remainder(yaw_step + torch.pi, 2.0 * torch.pi) - torch.pi
        yaw_relative = torch.cat(
            [torch.zeros_like(yaw_step[:, :1]), torch.cumsum(yaw_step, dim=1)], dim=1
        )
        xy_step = torch.diff(base_xy, dim=1)
        arm_step = torch.diff(traj[:, :, 3:10], dim=1)
        arm_second_difference = torch.diff(arm_step, dim=1)
        arm_third_difference = torch.diff(arm_second_difference, dim=1)

        metrics["waypoint_position_m"].extend(position_error.flatten().cpu().tolist())
        metrics["waypoint_rotation_rad"].extend(rotation_error.flatten().cpu().tolist())
        metrics["trajectory_max_position_m"].extend(position_error.max(dim=1).values.cpu().tolist())
        metrics["trajectory_max_rotation_rad"].extend(rotation_error.max(dim=1).values.cpu().tolist())
        metrics["terminal_position_m"].extend(position_error[:, -1].cpu().tolist())
        metrics["terminal_rotation_rad"].extend(rotation_error[:, -1].cpu().tolist())
        metrics["terminal_object_joint"].extend(object_joint_error.cpu().tolist())
        metrics["base_yaw_max_from_start_rad"].extend(
            yaw_relative.abs().max(dim=1).values.cpu().tolist()
        )
        metrics["base_yaw_total_variation_rad"].extend(
            yaw_step.abs().sum(dim=1).cpu().tolist()
        )
        metrics["base_yaw_max_step_rad"].extend(yaw_step.abs().max(dim=1).values.cpu().tolist())
        metrics["base_xy_max_from_start_m"].extend(
            torch.linalg.vector_norm(base_xy - base_xy[:, :1], dim=-1).max(dim=1).values.cpu().tolist()
        )
        metrics["base_xy_path_length_m"].extend(
            torch.linalg.vector_norm(xy_step, dim=-1).sum(dim=1).cpu().tolist()
        )
        metrics["arm_max_step_rad"].extend(
            arm_step.abs().flatten(1).max(dim=1).values.cpu().tolist()
        )
        metrics["arm_max_second_difference_rad"].extend(
            arm_second_difference.abs().flatten(1).max(dim=1).values.cpu().tolist()
        )
        metrics["arm_max_third_difference_rad"].extend(
            arm_third_difference.abs().flatten(1).max(dim=1).values.cpu().tolist()
        )

        for value in goal[:, -1].abs().cpu().tolist():
            key = f"{value:.3f}"
            goal_angle_counts[key] = goal_angle_counts.get(key, 0) + 1

    report = {
        "grasp_id": int(path.parent.name.split("_")[-1]),
        "successful": int(success.sum().item()),
        "goal_angle_counts": goal_angle_counts,
        "metrics": {key: summarize(values) for key, values in metrics.items()},
    }
    return report, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--akr-template",
        default="assets/object/Microwave/7221/summit_franka_7221_0_grasp_{grasp_id:04d}.yml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()

    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    per_grasp = []
    combined: dict[str, list[float]] = {}
    for path in sorted(args.run_root.glob("grasp_*/traj_data.pt")):
        grasp_id = int(path.parent.name.split("_")[-1])
        config_path = Path(args.akr_template.format(grasp_id=grasp_id))
        report, metrics = analyze_grasp(path, config_path, tensor_args, args.chunk_size)
        per_grasp.append(report)
        for key, values in metrics.items():
            combined.setdefault(key, []).extend(values)

    result = {
        "run_root": str(args.run_root),
        "link": "link_1",
        "reference": "FK(goal_states) for the same AKR grasp model",
        "successful": sum(int(item["successful"]) for item in per_grasp),
        "metrics": {key: summarize(values) for key, values in combined.items()},
        "posthoc_filter_retention": filter_retention(combined),
        "per_grasp": per_grasp,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
