# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Replay AutoMoMa planner trajectories without recording HDF5 demos.

This script shares environment setup with ``record_automoma_demos.py`` but keeps
recorder terms disabled. Add ``--metrics`` to write lightweight per-episode CSV
metrics for the same replay settings.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import add_example_environments_cli_args
from isaaclab_arena.scripts.automoma_replay_common import (
    add_automoma_replay_args,
    add_episode_selection_args,
    parse_episode_indices,
)
from isaaclab_arena.utils.automoma_record_debug import add_record_debug_args, make_record_debugger
from tools.debug.record_drive_and_set.set_state_patch import (
    parse_joint_names,
    patch_summit_franka_set_state_object_joints,
)

# ---- CLI arguments ----
parser = get_isaaclab_arena_cli_parser()
add_automoma_replay_args(parser)
add_episode_selection_args(parser)
parser.add_argument(
    "--metrics",
    action="store_true",
    default=False,
    help="Write per-episode replay metrics. This script never records HDF5 demos.",
)
parser.add_argument(
    "--metrics_file",
    type=str,
    default="metrics.csv",
    help="CSV path used when --metrics is set. Default: metrics.csv in the current directory.",
)
parser.add_argument(
    "--append_metrics",
    action="store_true",
    default=False,
    help="Append to --metrics_file instead of overwriting it at replay start.",
)
parser.add_argument(
    "--no_step_trace",
    action="store_true",
    default=False,
    help="Disable per-step joint/EEF trace artifacts that are written automatically with --metrics.",
)
parser.add_argument(
    "--object_joint_names",
    default="joint_0",
    help=(
        "Comma-separated object joints controlled by set-state actions. "
        "Defaults to joint_0 to match AutoMoMa 13D trajectories."
    ),
)

add_record_debug_args(parser)
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()
if args_cli.metrics and not args_cli.no_step_trace:
    # Reuse the record debug hook in artifact-only mode so metrics replay writes
    # per-step joint target/actual traces and curve PNGs without HDF5 recording.
    explicit_debug = args_cli.debug or args_cli.debug_joint_tracking or args_cli.debug_handle_tracking
    if explicit_debug:
        args_cli.debug_joint_tracking = True
    else:
        args_cli.debug = True

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
patch_summit_franka_set_state_object_joints(parse_joint_names(args_cli.object_joint_names))

"""Rest everything follows."""

import csv
import math
import os
from dataclasses import dataclass
from typing import Any

from isaaclab_arena.scripts.automoma_replay_common import (
    build_automoma_replay_context,
    evaluate_replay_success,
    print_replay_header,
    run_automoma_replay,
)


METRICS_FIELDS = [
    "object_name",
    "scene_name",
    "traj_file",
    "traj_index",
    "success",
    "final_door_open",
    "final_engaged",
    "final_door_openness",
    "final_openness",
    "final_handle_distance",
    "max_openness",
    "max_openness_step",
    "min_handle_distance",
    "min_handle_distance_step",
    "num_steps",
    "metrics_error",
]


class MetricsCsvWriter:
    """Small CSV writer that flushes each row for long simulator runs."""

    def __init__(self, path: str, *, append: bool):
        self.path = path
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        mode = "a" if append else "w"
        self._file = open(path, mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=METRICS_FIELDS, extrasaction="ignore")
        if not append or not file_exists:
            self._writer.writeheader()
            self._file.flush()

    def write(self, row: dict[str, object]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


@dataclass
class EpisodeMetricsAccumulator:
    max_openness: float | None = None
    max_openness_step: int | None = None
    min_handle_distance: float | None = None
    min_handle_distance_step: int | None = None

    def update(self, step: int, result: dict[str, Any]) -> None:
        openness = result.get("final_openness")
        handle_distance = result.get("final_handle_distance")
        if openness is not None and math.isfinite(float(openness)):
            if self.max_openness is None or float(openness) > self.max_openness:
                self.max_openness = float(openness)
                self.max_openness_step = step
        if handle_distance is not None and math.isfinite(float(handle_distance)):
            if self.min_handle_distance is None or float(handle_distance) < self.min_handle_distance:
                self.min_handle_distance = float(handle_distance)
                self.min_handle_distance_step = step


def _metrics_row(
    args_cli: Any,
    ctx: Any,
    actual_ep: int,
    final_result: dict[str, Any],
    acc: EpisodeMetricsAccumulator,
) -> dict[str, object]:
    return {
        "object_name": getattr(args_cli, "object_name", None),
        "scene_name": getattr(args_cli, "scene_name", None),
        "traj_file": args_cli.traj_file,
        "traj_index": actual_ep,
        "success": bool(final_result["success"]),
        "final_door_open": final_result["final_door_open"],
        "final_engaged": final_result["final_engaged"],
        "final_door_openness": final_result["final_door_openness"],
        "final_openness": final_result["final_openness"],
        "final_handle_distance": final_result["final_handle_distance"],
        "max_openness": acc.max_openness,
        "max_openness_step": acc.max_openness_step,
        "min_handle_distance": acc.min_handle_distance,
        "min_handle_distance_step": acc.min_handle_distance_step,
        "num_steps": ctx.policy.n_steps,
        "metrics_error": final_result.get("error", ""),
    }


def main():
    explicit_episode_indices = parse_episode_indices(args_cli)
    if explicit_episode_indices is not None:
        print("[Replay] Explicit episode indices are interpreted as raw trajectory-file indices.")

    record_debugger = make_record_debugger(args_cli)
    ctx = build_automoma_replay_context(
        args_cli,
        recorder_mode="none",
        explicit_episode_indices=explicit_episode_indices,
        require_openable_object=args_cli.metrics,
    )
    record_debugger.setup(ctx.env, ctx.env_cfg, ctx.policy)

    metrics_writer = MetricsCsvWriter(args_cli.metrics_file, append=args_cli.append_metrics) if args_cli.metrics else None
    output_path = args_cli.metrics_file if args_cli.metrics else None
    print_replay_header(ctx, args_cli, title="Replaying AutoMoMa demos without HDF5 recording", output_path=output_path)
    if args_cli.metrics:
        print(
            "Replay metrics: enabled "
            f"(openness_threshold={args_cli.openness_threshold}, "
            f"handle_distance_threshold={args_cli.record_eval_handle_distance_threshold}, "
            f"use_fingertips={not args_cli.disable_fingertip_proximity}, "
            f"append={args_cli.append_metrics})"
        )
    else:
        print("Replay metrics: disabled")

    current_acc = EpisodeMetricsAccumulator()
    successful_count = 0

    def on_step(local_ctx, _ep_idx: int, _actual_ep: int, step: int) -> None:
        if metrics_writer is None:
            return
        result = evaluate_replay_success(local_ctx.env, local_ctx.openable_object, args_cli)
        current_acc.update(step, result)

    def after_episode(local_ctx, _ep_idx: int, actual_ep: int) -> None:
        nonlocal current_acc, successful_count
        if metrics_writer is None:
            return
        final_result = evaluate_replay_success(local_ctx.env, local_ctx.openable_object, args_cli)
        if bool(final_result["success"]):
            successful_count += 1
        print(
            f"[ReplayMetrics] traj={actual_ep} success={final_result['success']} "
            f"door_open={final_result['final_door_open']} "
            f"final_engaged={final_result['final_engaged']} "
            f"openness={final_result['final_door_openness']} "
            f"handle_distance={final_result['final_handle_distance']}",
            flush=True,
        )
        metrics_writer.write(_metrics_row(args_cli, local_ctx, actual_ep, final_result, current_acc))
        current_acc = EpisodeMetricsAccumulator()

    replayed_count = run_automoma_replay(
        ctx,
        args_cli,
        record_debugger=record_debugger,
        on_step=on_step,
        after_episode=after_episode,
    )

    print(f"\n{'=' * 60}")
    if args_cli.metrics:
        success_rate = successful_count / replayed_count if replayed_count else 0.0
        print(
            f"Metrics replay complete: attempted={replayed_count} successful={successful_count} "
            f"success_rate={success_rate:.2%} metrics={args_cli.metrics_file}"
        )
    else:
        print(f"Replay complete: {replayed_count} episodes; no HDF5 written.")
    print(f"{'=' * 60}")

    debug_output_path = args_cli.metrics_file if args_cli.metrics else "/tmp/automoma_replay_debug.csv"
    record_debugger.finish(debug_output_path)
    if metrics_writer is not None:
        metrics_writer.close()
    ctx.env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
