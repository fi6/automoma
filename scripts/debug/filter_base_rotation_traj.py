#!/usr/bin/env python3
"""
Filter trajectories with base rotation limits.

This script scans all grasp folders under a given input path, loads traj_data.pt,
filters trajectories where the base rotation exceeds the limit, and writes
filtered_traj_data.pt next to the source file.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

BASE_Z_IDX = 2


def apply_base_rotation_filter(
    trajectories: torch.Tensor,
    success: torch.Tensor,
    rotation_limit: float,
) -> torch.Tensor:
    """Return updated success mask after base rotation filtering."""
    if trajectories is None or trajectories.shape[0] == 0:
        return success.to(torch.bool)

    success_mask = success.to(torch.bool).clone()

    for i in range(trajectories.shape[0]):
        if not success_mask[i]:
            continue

        base_z_traj = trajectories[i][:, BASE_Z_IDX]
        rotation_diff = base_z_traj[-1] - base_z_traj[0]

        if abs(rotation_diff) > rotation_limit:
            success_mask[i] = False

    return success_mask


def update_traj_data(
    traj_data: Dict,
    rotation_limit: float,
) -> Tuple[Dict, Dict[str, Tuple[int, int, int]]]:
    """
    Update trajectory success fields in-place, returning stats.

    Returns:
        updated traj_data and a stats dict keyed by stage key.
    """
    stats: Dict[str, Tuple[int, int, int]] = {}

    def _update(traj_key: str, success_key: str) -> None:
        if traj_key not in traj_data or success_key not in traj_data:
            return

        trajectories = traj_data[traj_key]
        success = traj_data[success_key]
        before = int(success.to(torch.bool).sum().item())
        updated_success = apply_base_rotation_filter(trajectories, success, rotation_limit)
        after = int(updated_success.sum().item())
        total = int(trajectories.shape[0])

        traj_data[success_key] = updated_success
        stats[success_key] = (before, after, total)

    _update("traj", "success")
    _update("reach_traj", "reach_success")
    _update("open_traj", "open_success")

    return traj_data, stats


def process_traj_file(traj_file: Path, rotation_limit: float, dry_run: bool) -> None:
    traj_data = torch.load(traj_file, weights_only=False)
    traj_data, stats = update_traj_data(traj_data, rotation_limit)

    if stats:
        for key, (before, after, total) in stats.items():
            print(f"{traj_file}: {key} {before}/{total} -> {after}/{total}")
    else:
        print(f"{traj_file}: no known trajectory keys, skipped")

    if not dry_run:
        output_path = traj_file.parent / "filtered_traj_data.pt"
        torch.save(traj_data, output_path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_input = repo_root / "output/collect_1211/traj/summit_franka/scene_0_seed_0/7221"

    parser = argparse.ArgumentParser(description="Filter trajectories by base rotation.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=default_input,
        help="Path containing grasp_* folders.",
    )
    parser.add_argument(
        "--rotation-limit",
        type=float,
        default=1 * np.pi,
        help="Base rotation limit in radians.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report stats without writing output files.",
    )
    args = parser.parse_args()

    input_path = args.input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    traj_files = sorted(input_path.glob("grasp_*/**/filtered_traj_data.pt"))
    if not traj_files:
        print(f"No traj_data.pt files found under {input_path}")
        return

    for traj_file in traj_files:
        process_traj_file(traj_file, args.rotation_limit, args.dry_run)


if __name__ == "__main__":
    main()
