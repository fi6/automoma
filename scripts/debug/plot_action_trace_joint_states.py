#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-joint action trace diagnostics.")
    parser.add_argument("--csv", required=True, help="Path to action_trace_joint_states.csv.")
    parser.add_argument("--output", default=None, help="Output image path. Defaults to <csv>.png.")
    parser.add_argument("--output_dir", default=None, help="Directory for --per_episode images.")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--per_episode", action="store_true", help="Write one 12-joint figure per episode.")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "episode_ix": int(row["episode_ix"]),
                    "policy_step_ix": int(row["policy_step_ix"]),
                    "sim_trace_step_ix": int(row["sim_trace_step_ix"]),
                    "sim_substep_ix": int(row["sim_substep_ix"]),
                    "is_policy_action_sample": int(row["is_policy_action_sample"]),
                    "joint_ix": int(row["joint_ix"]),
                    "joint_name": row["joint_name"],
                    "policy_action_abs": float(row["policy_action_abs"]),
                    "interpolated_action_abs": float(row["interpolated_action_abs"]),
                    "sim_joint_pos_after": float(row["sim_joint_pos_after"]),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def concatenated_step_maps(rows: list[dict[str, object]], episodes: list[int]) -> tuple[dict[int, dict[int, int]], list[int]]:
    by_episode: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        by_episode[int(row["episode_ix"])].add(int(row["sim_trace_step_ix"]))

    offset = 0
    maps: dict[int, dict[int, int]] = {}
    boundaries: list[int] = []
    for episode_ix in episodes:
        steps = sorted(by_episode.get(episode_ix, set()))
        maps[episode_ix] = {step: offset + local_ix for local_ix, step in enumerate(steps)}
        offset += len(steps)
        boundaries.append(offset)
    return maps, boundaries[:-1]


def local_step_maps(rows: list[dict[str, object]]) -> dict[int, dict[int, int]]:
    by_episode: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        by_episode[int(row["episode_ix"])].add(int(row["sim_trace_step_ix"]))
    return {
        episode_ix: {step: local_ix for local_ix, step in enumerate(sorted(steps))}
        for episode_ix, steps in by_episode.items()
    }


def plot_single_episode(
    rows: list[dict[str, object]],
    *,
    episode_ix: int,
    joints: list[tuple[int, str]],
    output_path: Path,
    dpi: int,
) -> None:
    episode_rows = [row for row in rows if int(row["episode_ix"]) == episode_ix]
    if not episode_rows:
        return

    step_maps = local_step_maps(episode_rows)
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in episode_rows:
        grouped[int(row["joint_ix"])].append(row)

    n_cols = 3
    n_rows = math.ceil(len(joints) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.4 * n_cols, 2.75 * n_rows), squeeze=False)
    for ax, (joint_ix, joint_name) in zip(axes.flat, joints):
        series = sorted(grouped.get(joint_ix, []), key=lambda row: int(row["sim_trace_step_ix"]))
        x = [step_maps[episode_ix][int(row["sim_trace_step_ix"])] for row in series]
        interpolated = [float(row["interpolated_action_abs"]) for row in series]
        sim_after = [float(row["sim_joint_pos_after"]) for row in series]
        policy_points = [row for row in series if int(row["is_policy_action_sample"]) == 1]
        policy_x = [step_maps[episode_ix][int(row["sim_trace_step_ix"])] for row in policy_points]
        policy_y = [float(row["policy_action_abs"]) for row in policy_points]

        ax.plot(x, interpolated, color="#1f77b4", linewidth=1.0, alpha=0.9)
        ax.plot(x, sim_after, color="#d62728", linewidth=1.0, alpha=0.75)
        ax.scatter(policy_x, policy_y, color="#111111", s=8, marker="o", alpha=0.72)
        ax.set_title(f"{joint_ix}: {joint_name}", fontsize=9)
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.tick_params(labelsize=7)

    for ax in axes.flat[len(joints):]:
        ax.axis("off")

    legend_items = [
        Line2D([0], [0], color="#1f77b4", linewidth=1.2, label="interpolated action"),
        Line2D([0], [0], color="#d62728", linewidth=1.2, label="sim joint after step"),
        Line2D([0], [0], color="black", marker="o", linestyle="", markersize=4, label="policy action abs"),
    ]
    fig.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, fontsize=9)
    fig.suptitle(f"Action trace by joint, episode {episode_ix}", y=0.985, fontsize=12)
    fig.supxlabel("sim substep within episode", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else csv_path.with_suffix(".png")
    rows = load_rows(csv_path)

    episodes = sorted({int(row["episode_ix"]) for row in rows})
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
        rows = [row for row in rows if int(row["episode_ix"]) in set(episodes)]

    joints = sorted(
        {(int(row["joint_ix"]), str(row["joint_name"])) for row in rows},
        key=lambda item: item[0],
    )

    if args.per_episode:
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else csv_path.parent / "action_trace_episode_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        for episode_ix in episodes:
            output_path = output_dir / f"action_trace_episode_{episode_ix:02d}.png"
            plot_single_episode(rows, episode_ix=episode_ix, joints=joints, output_path=output_path, dpi=args.dpi)
            print(output_path)
        return

    n_joints = len(joints)
    fig, axes = plt.subplots(n_joints, 1, figsize=(28, 2.2 * n_joints), squeeze=False, sharex=True)
    step_maps, episode_boundaries = concatenated_step_maps(rows, episodes)
    episode_midpoints = []
    for episode_ix in episodes:
        values = list(step_maps[episode_ix].values())
        if values:
            episode_midpoints.append((episode_ix, (min(values) + max(values)) / 2.0))

    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["joint_ix"]), int(row["episode_ix"]))].append(row)

    for ax, (joint_ix, joint_name) in zip(axes.flat, joints):
        for episode_ix in episodes:
            series = sorted(grouped.get((joint_ix, episode_ix), []), key=lambda row: int(row["sim_trace_step_ix"]))
            if not series:
                continue
            x = [step_maps[episode_ix][int(row["sim_trace_step_ix"])] for row in series]
            interpolated = [float(row["interpolated_action_abs"]) for row in series]
            sim_after = [float(row["sim_joint_pos_after"]) for row in series]
            policy_points = [row for row in series if int(row["is_policy_action_sample"]) == 1]
            policy_x = [step_maps[episode_ix][int(row["sim_trace_step_ix"])] for row in policy_points]
            policy_y = [float(row["policy_action_abs"]) for row in policy_points]

            ax.plot(x, interpolated, color="#1f77b4", linewidth=0.9, alpha=0.9)
            ax.plot(x, sim_after, color="#d62728", linewidth=0.9, linestyle="-", alpha=0.75)
            ax.scatter(policy_x, policy_y, color="#111111", s=7, marker="o", alpha=0.72)

        for boundary in episode_boundaries:
            ax.axvline(boundary, color="#666666", linestyle="--", linewidth=0.7, alpha=0.55)
        ax.set_ylabel(f"{joint_ix}\n{joint_name}", fontsize=8, rotation=0, ha="right", va="center")
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.tick_params(labelsize=7)

    legend_items = [
        Line2D([0], [0], color="#1f77b4", linewidth=1.2, label="interpolated action"),
        Line2D([0], [0], color="#d62728", linewidth=1.2, label="sim joint after step"),
        Line2D([0], [0], color="black", marker="o", linestyle="", markersize=4, label="policy action abs"),
    ]
    fig.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, 0.992), ncol=3, fontsize=9)
    top_ax = axes.flat[0]
    for episode_ix, midpoint in episode_midpoints:
        top_ax.text(midpoint, 1.02, f"ep {episode_ix}", transform=top_ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Action trace by joint, concatenated episodes ({len(episodes)} episodes)", y=0.982, fontsize=12)
    fig.supxlabel("concatenated sim substep", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)
    print(output_path)


if __name__ == "__main__":
    main()
