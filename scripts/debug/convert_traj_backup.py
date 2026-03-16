#!/usr/bin/env python3
"""Reverse format converter for AutoMoMa trajectories.

Converts new-format per-grasp `traj_data.pt` files back to the old 
`filtered_traj_data.pt` format expected by original IsaacSim scripts.

Usage:
    python scripts/debug/convert_traj_backup.py --input_dir data/trajs/summit_franka/microwave_7221/scene_0_seed_0
"""

import argparse
import os
from pathlib import Path

import torch


def convert_grasp_dir(grasp_dir: Path):
    """Convert traj_data.pt to filtered_traj_data.pt in a single directory."""
    new_pt = grasp_dir / "traj_data.pt"
    if not new_pt.exists():
        return

    print(f"Converting {new_pt}")
    data = torch.load(new_pt, map_location="cpu", weights_only=False)

    # Key mapping: new (plural) -> old (singular)
    # The new per-grasp format has: start_states, goal_states, trajectories, success
    # The old format expects: start_state, goal_state, traj, success
    old_data = {
        "start_state": data.get("start_states", data.get("start_state")),
        "goal_state": data.get("goal_states", data.get("goal_state")),
        "traj": data.get("trajectories", data.get("traj")),
        "success": data.get("success"),
    }

    # Verify keys
    missing = [k for k, v in old_data.items() if v is None]
    if missing:
        print(f"  Warning: Missing keys {missing} in {new_pt}")
        return

    old_pt = grasp_dir / "filtered_traj_data.pt"
    torch.save(old_data, old_pt)
    print(f"  Saved: {old_pt}")


def main():
    parser = argparse.ArgumentParser(description="Convert traj data back to backup format.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing grasp_* subdirectories.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    if not input_root.is_dir():
        print(f"Error: {args.input_dir} is not a directory.")
        return

    # Process all grasp directories
    grasp_dirs = sorted(
        d for d in input_root.iterdir()
        if d.is_dir() and d.name.startswith("grasp_")
    )

    if not grasp_dirs:
        # Check if the input_dir itself is a grasp dir
        if input_root.name.startswith("grasp_"):
            grasp_dirs = [input_root]
        else:
            print(f"No grasp_* directories found in {args.input_dir}")
            return

    for gd in grasp_dirs:
        convert_grasp_dir(gd)


if __name__ == "__main__":
    main()
