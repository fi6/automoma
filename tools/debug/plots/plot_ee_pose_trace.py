#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot EE pose and per-joint target-real gaps from action trace CSV.")
    parser.add_argument("--trace_csv", required=True, help="Path to action_trace_joint_states.csv.")
    parser.add_argument("--per_episode_csv", default=None, help="Optional per_episode_results.csv for episode titles.")
    parser.add_argument("--output_dir", required=True, help="Directory for per-episode PNGs.")
    parser.add_argument("--max_episodes", type=int, default=None)
    return parser.parse_args()


def choose_column(columns: set[str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"None of these columns exist in trace CSV: {candidates}")


def load_episode_info(path: str | None) -> dict[int, dict[str, str]]:
    if not path:
        return {}
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    rows = pd.read_csv(csv_path, dtype=str).fillna("")
    if "episode_ix" not in rows.columns:
        return {}
    return {int(row["episode_ix"]): row.to_dict() for _, row in rows.iterrows()}


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def title_for_episode(episode_ix: int, info: dict[str, str]) -> str:
    if not info:
        return f"episode {episode_ix}"
    fields = [f"episode {episode_ix}"]
    if "demo_key" in info and info["demo_key"]:
        fields.append(info["demo_key"])
    if "seed" in info and info["seed"]:
        fields.append(f"seed={info['seed']}")
    if "success" in info:
        fields.append(f"success={truthy(info['success'])}")
    if "final_door_open" in info:
        fields.append(f"door_open={truthy(info['final_door_open'])}")
    if "final_engaged" in info:
        fields.append(f"engaged={truthy(info['final_engaged'])}")
    if "final_handle_distance" in info and info["final_handle_distance"]:
        try:
            fields.append(f"handle_dist={float(info['final_handle_distance']):.4f}")
        except ValueError:
            fields.append(f"handle_dist={info['final_handle_distance']}")
    if "steps" in info and info["steps"]:
        fields.append(f"steps={info['steps']}")
    return " | ".join(fields)


def finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else math.nan


def plot_ee_position_overlay(ax: plt.Axes, x_pose: np.ndarray, step_pose: pd.DataFrame) -> None:
    colors = {
        "x": "#2563eb",
        "y": "#059669",
        "z": "#dc2626",
    }
    for axis_name, color in colors.items():
        target = step_pose[f"target_ee_pos_{axis_name}"].to_numpy(dtype=float)
        real = step_pose[f"real_ee_pos_{axis_name}"].to_numpy(dtype=float)
        ax.plot(x_pose, target, color=color, linewidth=1.7, label=f"target {axis_name}")
        ax.plot(x_pose, real, color=color, linewidth=1.4, linestyle="--", label=f"real {axis_name}")
        ax.fill_between(x_pose, target, real, color=color, alpha=0.12, linewidth=0)


def plot_episode(
    episode: pd.DataFrame,
    *,
    episode_ix: int,
    episode_info: dict[str, str],
    output_path: Path,
    target_joint_col: str,
    real_joint_col: str,
    step_col: str,
) -> None:
    if {"terminated", "truncated"}.issubset(episode.columns):
        episode = episode[
            (episode["terminated"].fillna(0).astype(int) == 0)
            & (episode["truncated"].fillna(0).astype(int) == 0)
        ].copy()
    episode = episode.dropna(subset=["ee_pos_error_m", "ee_rot_error_rad"])
    if episode.empty:
        return

    step_pose = (
        episode.sort_values([step_col, "joint_ix"])
        .drop_duplicates(subset=[step_col], keep="first")
        .sort_values(step_col)
    )
    x_pose = step_pose[step_col].to_numpy(dtype=float)
    pos_error = step_pose["ee_pos_error_m"].to_numpy(dtype=float)
    rot_error = step_pose["ee_rot_error_rad"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
    fig.suptitle(title_for_episode(episode_ix, episode_info), fontsize=12)

    ax_pose = axes[0]
    plot_ee_position_overlay(ax_pose, x_pose, step_pose)
    ax_pose.set_title("EE position target vs real")
    ax_pose.set_xlabel("trace step")
    ax_pose.set_ylabel("world position (m)")
    ax_pose.grid(True, alpha=0.25)
    ax_pose.legend(loc="upper right", fontsize=8, ncol=2)
    ax_pose.text(
        0.01,
        0.98,
        f"max pos={finite_max(pos_error):.4f} m\nmax rot={finite_max(rot_error):.4f} rad",
        transform=ax_pose.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.85},
    )

    ax_joint = axes[1]
    for joint_name, joint_rows in episode.sort_values([step_col, "joint_ix"]).groupby("joint_name", sort=False):
        x = joint_rows[step_col].to_numpy(dtype=float)
        target = joint_rows[target_joint_col].to_numpy(dtype=float)
        real = joint_rows[real_joint_col].to_numpy(dtype=float)
        ax_joint.plot(x, np.abs(target - real), linewidth=1.0, label=str(joint_name))
    ax_joint.set_title("Per-joint absolute target-real gap")
    ax_joint.set_xlabel("trace step")
    ax_joint.set_ylabel("absolute joint gap")
    ax_joint.grid(True, alpha=0.25)
    ax_joint.legend(loc="upper right", fontsize=7, ncol=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    trace = pd.read_csv(args.trace_csv)
    columns = set(trace.columns)
    required = {
        "episode_ix",
        "joint_ix",
        "joint_name",
        "target_ee_pos_x",
        "target_ee_pos_y",
        "target_ee_pos_z",
        "real_ee_pos_x",
        "real_ee_pos_y",
        "real_ee_pos_z",
        "ee_pos_error_m",
        "ee_rot_error_rad",
    }
    missing = sorted(required - columns)
    if missing:
        raise KeyError(f"Trace CSV is missing EE pose columns: {missing}")

    target_joint_col = choose_column(columns, ("interpolated_action_abs", "prepared_action_abs", "policy_action_abs"))
    real_joint_col = choose_column(columns, ("sim_joint_pos_after", "eval_sim_joint_pos_after"))
    step_col = choose_column(columns, ("sim_trace_step_ix", "policy_step_ix"))
    episode_info = load_episode_info(args.per_episode_csv)
    output_dir = Path(args.output_dir)

    episodes = sorted(int(value) for value in trace["episode_ix"].dropna().unique())
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    for episode_ix in episodes:
        episode = trace[trace["episode_ix"] == episode_ix].copy()
        output_path = output_dir / f"episode_{episode_ix:03d}_ee_pose_gap.png"
        plot_episode(
            episode,
            episode_ix=episode_ix,
            episode_info=episode_info.get(episode_ix, {}),
            output_path=output_path,
            target_joint_col=target_joint_col,
            real_joint_col=real_joint_col,
            step_col=step_col,
        )
    print(f"Wrote {len(episodes)} plots to {output_dir}")


if __name__ == "__main__":
    main()
