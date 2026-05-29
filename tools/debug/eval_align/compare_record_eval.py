#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import h5py
import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


JOINT_NAMES = (
    "base_x",
    "base_y",
    "base_z",
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
)
BASE_JOINT_NAMES = set(JOINT_NAMES[:3])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create random record-vs-eval visual comparisons from HDF5 and record_dataset_eval outputs."
    )
    parser.add_argument("--dataset_file", required=True)
    parser.add_argument("--eval_output_dir", required=True)
    parser.add_argument("--diagnostic_eval_output_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", default="fix_local", choices=("fix_local", "ego_topdown", "ego_wrist"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--include_all_rows", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    def key_fn(key: str) -> tuple[int, str]:
        try:
            return int(key.rsplit("_", 1)[1]), key
        except (IndexError, ValueError):
            return 1_000_000_000, key

    return sorted(data_group.keys(), key=key_fn)


def read_hdf5_rgb(dataset_file: Path, demo_key: str, camera: str) -> np.ndarray:
    key = f"data/{demo_key}/camera_obs/{camera}_rgb"
    with h5py.File(dataset_file, "r") as f:
        if key not in f:
            raise KeyError(f"Missing HDF5 camera frames: {key}")
        frames = f[key][:]
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames


def read_video(path: Path) -> list[np.ndarray]:
    if not path.exists():
        return []
    reader = imageio.get_reader(str(path))
    try:
        return [np.asarray(frame, dtype=np.uint8) for frame in reader]
    finally:
        reader.close()


def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    h, w = frame.shape[:2]
    if h == height:
        return frame
    width = max(1, int(round(w * height / h)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), thickness=-1)
    cv2.putText(out, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_video(path: Path, frames: list[np.ndarray] | np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def make_side_by_side_video(
    *,
    record_frames: np.ndarray,
    eval_frames: list[np.ndarray],
    out_path: Path,
    fps: int,
    height: int,
    record_label: str,
    eval_label: str,
) -> None:
    n_frames = max(len(record_frames), len(eval_frames))
    if n_frames == 0:
        return
    output_frames: list[np.ndarray] = []
    for idx in range(n_frames):
        record_idx = min(idx, len(record_frames) - 1)
        record_frame = resize_to_height(record_frames[record_idx], height)
        record_frame = label_frame(record_frame, record_label)

        if eval_frames:
            eval_idx = min(idx, len(eval_frames) - 1)
            eval_frame = resize_to_height(eval_frames[eval_idx], height)
            eval_frame = label_frame(eval_frame, eval_label)
        else:
            eval_frame = np.zeros_like(record_frame)
            eval_frame = label_frame(eval_frame, f"{eval_label}: missing")

        if record_frame.shape[0] != eval_frame.shape[0]:
            eval_frame = resize_to_height(eval_frame, record_frame.shape[0])
        output_frames.append(np.concatenate([record_frame, eval_frame], axis=1))
    write_video(out_path, output_frames, fps=fps)


def rows_by_episode(trace_rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in trace_rows:
        grouped[int(row["episode_ix"])].append(row)
    return grouped


def hdf5_action_abs_from_row(row: dict[str, str]) -> float:
    raw_action = to_float(row.get("dataset_action_raw", ""))
    if row.get("joint_name") in BASE_JOINT_NAMES:
        record_obs = to_float(row.get("record_obs_joint_pos", ""))
        return raw_action + record_obs if np.isfinite(raw_action) and np.isfinite(record_obs) else np.nan
    return raw_action


def rows_with_abs_actions(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        enriched = dict(row)
        enriched["hdf5_action_abs"] = fmt_float(hdf5_action_abs_from_row(row), ndigits=8)
        output.append(enriched)
    return output


def write_trace_subset(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    rows = rows_with_abs_actions(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def trace_arrays(rows: list[dict[str, str]]) -> dict[str, dict[str, np.ndarray]]:
    by_joint: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_joint[row["joint_name"]].append(row)

    data: dict[str, dict[str, np.ndarray]] = {}
    for joint_name, joint_rows in by_joint.items():
        joint_rows = sorted(joint_rows, key=lambda r: (int(r["sim_trace_step_ix"]), int(r["joint_ix"])))
        raw_steps = np.asarray([int(r["sim_trace_step_ix"]) for r in joint_rows], dtype=np.int32)
        local_steps = raw_steps - int(raw_steps.min()) if raw_steps.size else raw_steps
        dataset_action_raw = np.asarray([to_float(r["dataset_action_raw"]) for r in joint_rows])
        record_obs_joint_pos = np.asarray([to_float(r["record_obs_joint_pos"]) for r in joint_rows])
        if joint_name in BASE_JOINT_NAMES:
            dataset_action_abs = dataset_action_raw + record_obs_joint_pos
        else:
            dataset_action_abs = dataset_action_raw.copy()

        data[joint_name] = {
            "step": local_steps,
            "global_step": raw_steps,
            "dataset_action_raw": dataset_action_raw,
            "dataset_action_abs": dataset_action_abs,
            "prepared_action_abs": np.asarray([to_float(r["prepared_action_abs"]) for r in joint_rows]),
            "interpolated_action_abs": np.asarray([to_float(r["interpolated_action_abs"]) for r in joint_rows]),
            "eval_sim_joint_pos_after": np.asarray([to_float(r["eval_sim_joint_pos_after"]) for r in joint_rows]),
            "record_obs_joint_pos": record_obs_joint_pos,
            "record_state_joint_pos": np.asarray([to_float(r["record_state_joint_pos"]) for r in joint_rows]),
            "abs_eval_vs_record_state": np.asarray([to_float(r["abs_eval_vs_record_state"]) for r in joint_rows]),
        }
    return data


def summarize_trace_data(data: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    per_joint: list[dict[str, Any]] = []
    all_errors: list[float] = []
    for joint_name in JOINT_NAMES:
        joint = data.get(joint_name)
        if joint is None or joint["step"].size == 0:
            continue
        eval_record = joint["eval_sim_joint_pos_after"] - joint["record_state_joint_pos"]
        prepared_record = joint["prepared_action_abs"] - joint["record_state_joint_pos"]
        record_action_prepared = joint["prepared_action_abs"] - joint["dataset_action_abs"]
        record_action_record = joint["dataset_action_abs"] - joint["record_state_joint_pos"]
        executed_eval = joint["eval_sim_joint_pos_after"] - joint["interpolated_action_abs"]
        abs_eval_record = np.abs(eval_record)
        all_errors.extend(abs_eval_record[np.isfinite(abs_eval_record)].tolist())
        final_ix = -1
        per_joint.append(
            {
                "joint_name": joint_name,
                "max_abs_eval_record": float(np.nanmax(abs_eval_record)),
                "mean_abs_eval_record": float(np.nanmean(abs_eval_record)),
                "final_signed_eval_record": float(eval_record[final_ix]),
                "final_abs_eval_record": float(abs_eval_record[final_ix]),
                "max_abs_prepared_record": float(np.nanmax(np.abs(prepared_record))),
                "max_abs_hdf5_action_record": float(np.nanmax(np.abs(record_action_record))),
                "max_abs_hdf5_action_prepared": float(np.nanmax(np.abs(record_action_prepared))),
                "max_abs_raw_prepared": float(np.nanmax(np.abs(joint["prepared_action_abs"] - joint["dataset_action_raw"]))),
                "max_abs_executed_eval": float(np.nanmax(np.abs(executed_eval))),
            }
        )

    top = sorted(per_joint, key=lambda item: item["max_abs_eval_record"], reverse=True)[:5]
    return {
        "per_joint": per_joint,
        "top_joints": top,
        "max_abs_eval_record": float(np.nanmax(all_errors)) if all_errors else np.nan,
        "mean_abs_eval_record": float(np.nanmean(all_errors)) if all_errors else np.nan,
    }


def plot_action_grid(data: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    fig, axes = plt.subplots(len(JOINT_NAMES), 2, figsize=(18, 28), sharex=False)
    colors = {
        "raw": "#d97706",
        "prepared": "#2563eb",
        "executed": "#059669",
        "diff": "#dc2626",
        "zero": "#6b7280",
    }
    for row_ix, joint_name in enumerate(JOINT_NAMES):
        ax = axes[row_ix, 0]
        diff_ax = axes[row_ix, 1]
        joint = data.get(joint_name)
        ax.set_title(f"{joint_name}: absolute action targets")
        diff_ax.set_title(f"{joint_name}: absolute target differences")
        if joint is None:
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center")
            diff_ax.text(0.5, 0.5, "missing", transform=diff_ax.transAxes, ha="center", va="center")
            continue
        x = joint["step"]
        ax.plot(
            x,
            joint["dataset_action_abs"],
            label="HDF5 action absolute target",
            linewidth=1.4,
            color=colors["raw"],
        )
        ax.plot(
            x,
            joint["prepared_action_abs"],
            label="eval prepared absolute target",
            linewidth=1.5,
            color=colors["prepared"],
        )
        ax.plot(
            x,
            joint["interpolated_action_abs"],
            label="eval executed/interpolated target",
            linewidth=1.1,
            linestyle="--",
            color=colors["executed"],
        )
        diff_ax.axhline(0.0, color=colors["zero"], linewidth=0.7, alpha=0.65)
        diff_ax.plot(
            x,
            joint["prepared_action_abs"] - joint["dataset_action_abs"],
            label="eval_prepared_abs - HDF5_abs",
            linewidth=1.2,
            color=colors["diff"],
        )
        diff_ax.plot(
            x,
            joint["interpolated_action_abs"] - joint["prepared_action_abs"],
            label="executed_target - prepared_abs",
            linewidth=1.2,
            linestyle="--",
            color=colors["executed"],
        )
        ax.grid(True, alpha=0.25)
        diff_ax.grid(True, alpha=0.25)
        if row_ix == len(JOINT_NAMES) - 1:
            ax.set_xlabel("sim trace step")
            diff_ax.set_xlabel("sim trace step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles + handles2, labels + labels2, loc="upper center", ncol=5)
    fig.suptitle("Action comparison: base HDF5 deltas are converted to absolute targets before plotting")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_joint_grid(data: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    fig, axes = plt.subplots(len(JOINT_NAMES), 2, figsize=(18, 28), sharex=False)
    colors = {
        "record": "#111827",
        "obs": "#6b7280",
        "eval": "#2563eb",
        "target": "#059669",
        "err": "#dc2626",
    }
    for row_ix, joint_name in enumerate(JOINT_NAMES):
        ax = axes[row_ix, 0]
        err_ax = axes[row_ix, 1]
        joint = data.get(joint_name)
        ax.set_title(f"{joint_name}: record/eval state")
        err_ax.set_title(f"{joint_name}: signed errors")
        if joint is None:
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center")
            err_ax.text(0.5, 0.5, "missing", transform=err_ax.transAxes, ha="center", va="center")
            continue
        x = joint["step"]
        ax.plot(x, joint["record_state_joint_pos"], label="HDF5 record sim state", linewidth=1.5, color=colors["record"])
        ax.plot(x, joint["record_obs_joint_pos"], label="HDF5 obs joint", linewidth=1.0, alpha=0.55, color=colors["obs"])
        ax.plot(
            x,
            joint["dataset_action_abs"],
            label="HDF5 action absolute target",
            linewidth=1.0,
            linestyle=":",
            alpha=0.9,
            color="#d97706",
        )
        ax.plot(x, joint["prepared_action_abs"], label="eval prepared target", linewidth=1.0, alpha=0.8, color=colors["target"])
        ax.plot(x, joint["eval_sim_joint_pos_after"], label="eval sim state after step", linewidth=1.4, color=colors["eval"])
        err_ax.axhline(0.0, color=colors["obs"], linewidth=0.7, alpha=0.65)
        err_ax.fill_between(
            x,
            0.0,
            joint["eval_sim_joint_pos_after"] - joint["record_state_joint_pos"],
            color=colors["err"],
            alpha=0.22,
            label="eval_state - record_state",
        )
        err_ax.plot(
            x,
            joint["prepared_action_abs"] - joint["record_state_joint_pos"],
            color=colors["target"],
            linewidth=1.0,
            label="prepared_target - record_state",
        )
        err_ax.plot(
            x,
            joint["prepared_action_abs"] - joint["dataset_action_abs"],
            color="#d97706",
            linewidth=1.0,
            linestyle=":",
            label="prepared_target - HDF5_abs_action",
        )
        ax.grid(True, alpha=0.25)
        err_ax.grid(True, alpha=0.25)
        if row_ix == len(JOINT_NAMES) - 1:
            ax.set_xlabel("sim trace step")
            err_ax.set_xlabel("sim trace step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles + handles2, labels + labels2, loc="upper center", ncol=5)
    fig.suptitle("Joint state comparison: target alignment and physical replay drift are separated")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_heatmap(data: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    max_steps = 0
    for joint in data.values():
        if joint["step"].size:
            max_steps = max(max_steps, int(np.nanmax(joint["step"])) + 1)
    if max_steps == 0:
        return
    heatmap = np.full((len(JOINT_NAMES), max_steps), np.nan, dtype=np.float64)
    for joint_ix, joint_name in enumerate(JOINT_NAMES):
        joint = data.get(joint_name)
        if joint is None:
            continue
        for step, value in zip(joint["step"], joint["abs_eval_vs_record_state"], strict=False):
            if 0 <= step < max_steps:
                heatmap[joint_ix, int(step)] = value

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_yticks(np.arange(len(JOINT_NAMES)))
    ax.set_yticklabels(JOINT_NAMES)
    ax.set_xlabel("sim trace step")
    ax.set_title("abs(eval sim joint - hdf5 record state)")
    fig.colorbar(im, ax=ax, label="absolute error")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_episode_summary(data: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    max_by_joint = []
    base_series = []
    arm_step_errors = []
    base_step_errors = []
    gripper_step_errors = []

    max_steps = 0
    for joint in data.values():
        if joint["step"].size:
            max_steps = max(max_steps, int(np.nanmax(joint["step"])) + 1)

    for joint_name in JOINT_NAMES:
        joint = data.get(joint_name)
        if joint is None or joint["step"].size == 0:
            max_by_joint.append(np.nan)
            continue
        err = np.abs(joint["eval_sim_joint_pos_after"] - joint["record_state_joint_pos"])
        max_by_joint.append(float(np.nanmax(err)))
        if joint_name in {"base_x", "base_y", "base_z"}:
            base_series.append((joint_name, joint))

    for step in range(max_steps):
        base_vals, arm_vals, gripper_vals = [], [], []
        for joint_name in JOINT_NAMES:
            joint = data.get(joint_name)
            if joint is None:
                continue
            matches = np.where(joint["step"] == step)[0]
            if not matches.size:
                continue
            ix = int(matches[-1])
            value = abs(joint["eval_sim_joint_pos_after"][ix] - joint["record_state_joint_pos"][ix])
            if joint_name.startswith("base_"):
                base_vals.append(value)
            elif joint_name.startswith("panda_finger"):
                gripper_vals.append(value)
            else:
                arm_vals.append(value)
        base_step_errors.append(np.nanmax(base_vals) if base_vals else np.nan)
        arm_step_errors.append(np.nanmax(arm_vals) if arm_vals else np.nan)
        gripper_step_errors.append(np.nanmax(gripper_vals) if gripper_vals else np.nan)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_err = fig.add_subplot(gs[0, 1:])
    base_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    colors = ["#dc2626" if name.startswith("base_") else "#2563eb" for name in JOINT_NAMES]
    ax_bar.bar(np.arange(len(JOINT_NAMES)), max_by_joint, color=colors, alpha=0.85)
    ax_bar.set_xticks(np.arange(len(JOINT_NAMES)))
    ax_bar.set_xticklabels(JOINT_NAMES, rotation=55, ha="right")
    ax_bar.set_ylabel("max abs error")
    ax_bar.set_title("Where is the replay mismatch?")
    ax_bar.grid(axis="y", alpha=0.25)

    x = np.arange(max_steps)
    ax_err.plot(x, base_step_errors, label="base max error", color="#dc2626", linewidth=2.0)
    ax_err.plot(x, arm_step_errors, label="arm max error", color="#2563eb", linewidth=1.8)
    ax_err.plot(x, gripper_step_errors, label="gripper max error", color="#059669", linewidth=1.5)
    ax_err.set_title("Error over time by joint group")
    ax_err.set_xlabel("episode step")
    ax_err.set_ylabel("max abs error")
    ax_err.grid(True, alpha=0.25)
    ax_err.legend()

    for ax, (joint_name, joint) in zip(base_axes, base_series[:3], strict=False):
        x = joint["step"]
        ax.plot(x, joint["record_state_joint_pos"], label="record state", color="#111827", linewidth=1.8)
        ax.plot(
            x,
            joint["dataset_action_abs"],
            label="record action abs",
            color="#d97706",
            linewidth=1.4,
            linestyle=":",
        )
        ax.plot(x, joint["eval_sim_joint_pos_after"], label="eval state", color="#2563eb", linewidth=1.6)
        ax.plot(x, joint["prepared_action_abs"], label="eval target", color="#059669", linewidth=1.1, alpha=0.85)
        ax.set_title(f"{joint_name}: record/eval absolute values")
        ax.set_xlabel("episode step")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle("Replay alignment summary")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt_float(value: Any, ndigits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.{ndigits}f}"


def relpath(path: str | Path, root: Path) -> str:
    return html.escape(Path(path).resolve().relative_to(root.resolve()).as_posix())


def resolve_eval_video(row: dict[str, str], eval_output_dir: Path) -> Path | None:
    raw = row.get("video_path") or ""
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = eval_output_dir / path
        if path.exists():
            return path
    candidate = eval_output_dir / "videos" / "summit_franka_open_door_eval_0" / f"eval_episode_{row['episode_ix']}.mp4"
    return candidate if candidate.exists() else None


def write_summary_csv(output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for item in manifest:
        for joint in item.get("trace_summary", {}).get("per_joint", []):
            rows.append(
                {
                    "rank": item["rank"],
                    "episode_ix": item["episode_ix"],
                    "demo_key": item["demo_key"],
                    **joint,
                }
            )
    if not rows:
        return
    with (output_dir / "per_joint_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def diagnostic_summary(diagnostic_eval_output_dir: Path | None) -> dict[str, Any] | None:
    if diagnostic_eval_output_dir is None:
        return None
    per_episode_path = diagnostic_eval_output_dir / "per_episode_results.csv"
    if not per_episode_path.exists():
        return None
    rows = read_csv_rows(per_episode_path)
    max_errors = [to_float(row.get("max_abs_eval_vs_record_state", "")) for row in rows]
    mean_errors = [to_float(row.get("mean_abs_eval_vs_record_state", "")) for row in rows]
    max_errors = [value for value in max_errors if np.isfinite(value)]
    mean_errors = [value for value in mean_errors if np.isfinite(value)]
    return {
        "path": str(diagnostic_eval_output_dir),
        "n": len(rows),
        "global_max": float(np.nanmax(max_errors)) if max_errors else np.nan,
        "mean_of_mean": float(np.nanmean(mean_errors)) if mean_errors else np.nan,
    }


def make_html_index(
    output_dir: Path,
    manifest: list[dict[str, Any]],
    *,
    dataset_file: Path,
    eval_output_dir: Path,
    diagnostic_eval_output_dir: Path | None = None,
) -> None:
    eval_info = read_json_if_exists(eval_output_dir / "eval_info.json")
    record_cfg = eval_info.get("record_dataset_eval", {})
    record_env_args = record_cfg.get("record_env_args", {})
    record_sim_args = record_env_args.get("sim_args", {}) if isinstance(record_env_args, dict) else {}
    diag = diagnostic_summary(diagnostic_eval_output_dir)
    max_errors = [float(item["trace_summary"]["max_abs_eval_record"]) for item in manifest if item.get("trace_summary")]
    mean_errors = [float(item["trace_summary"]["mean_abs_eval_record"]) for item in manifest if item.get("trace_summary")]
    worst_item = max(manifest, key=lambda item: float(item.get("trace_summary", {}).get("max_abs_eval_record", -np.inf)))
    top_joint_counts: dict[str, int] = defaultdict(int)
    for item in manifest:
        for joint in item.get("trace_summary", {}).get("top_joints", [])[:2]:
            top_joint_counts[joint["joint_name"]] += 1
    top_joint_text = ", ".join(f"{name} x{count}" for name, count in sorted(top_joint_counts.items(), key=lambda kv: kv[1], reverse=True))

    rows_html = []
    for item in manifest:
        top = item.get("trace_summary", {}).get("top_joints", [])[:3]
        top_html = "<br>".join(
            f"{html.escape(j['joint_name'])}: max {fmt_float(j['max_abs_eval_record'])}, final {fmt_float(j['final_abs_eval_record'])}"
            for j in top
        )
        rows_html.append(
            "<tr>"
            f"<td>{item['rank']}</td>"
            f"<td>{item['episode_ix']}</td>"
            f"<td>{html.escape(str(item['demo_key']))}</td>"
            f"<td>{fmt_float(item.get('max_abs_eval_vs_record_state'))}</td>"
            f"<td>{fmt_float(item.get('mean_abs_eval_vs_record_state'))}</td>"
            f"<td>{top_html}</td>"
            f"<td><a href='#{html.escape(item['anchor'])}'>查看</a></td>"
            "</tr>"
        )

    cards_html = []
    for item in manifest:
        side = relpath(item["side_by_side_video"], output_dir)
        summary = relpath(item["summary_plot"], output_dir)
        action = relpath(item["action_plot"], output_dir)
        joints = relpath(item["joint_plot"], output_dir)
        heatmap = relpath(item["error_heatmap"], output_dir)
        trace = relpath(item["trace_csv"], output_dir)
        top = item.get("trace_summary", {}).get("top_joints", [])[:5]
        top_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(j['joint_name'])}</td>"
            f"<td>{fmt_float(j['max_abs_eval_record'])}</td>"
            f"<td>{fmt_float(j['mean_abs_eval_record'])}</td>"
            f"<td>{fmt_float(j['final_abs_eval_record'])}</td>"
            f"<td>{fmt_float(j['max_abs_hdf5_action_prepared'])}</td>"
            f"<td>{fmt_float(j['max_abs_prepared_record'])}</td>"
            f"<td>{fmt_float(j['max_abs_executed_eval'])}</td>"
            "</tr>"
            for j in top
        )
        cards_html.append(
            f"""
            <section class="episode" id="{html.escape(item['anchor'])}">
              <div class="episode-head">
                <div>
                  <h2>Sample {item['rank']:02d} · Episode {item['episode_ix']} · {html.escape(str(item['demo_key']))}</h2>
                  <p>整体误差：max {fmt_float(item.get('max_abs_eval_vs_record_state'))}，mean {fmt_float(item.get('mean_abs_eval_vs_record_state'))}。Trace CSV: <a href="{trace}">{trace}</a></p>
                </div>
              </div>
              <div class="video-grid">
                <div>
                  <h3>Record HDF5 vs Eval Replay</h3>
                  <video src="{side}" controls muted preload="metadata"></video>
                </div>
              </div>
              <figure class="summary-figure"><img src="{summary}" alt="summary"><figcaption>先看这张：左上定位哪个 joint 出错，右上看误差什么时候出现，下方看 base 是否是主要来源。</figcaption></figure>
              <details>
                <summary>展开完整 action / joint / heatmap 细节图</summary>
                <div class="plot-grid">
                  <figure><img src="{action}" alt="action comparison"><figcaption>Action：左侧全部是 absolute target。base 的 HDF5 raw delta 已先加回 record obs，变成 HDF5 action absolute target；右侧直接画 eval prepared target 与 HDF5 absolute target 的差。</figcaption></figure>
                  <figure><img src="{joints}" alt="joint state comparison"><figcaption>Joint：左侧画 record state、HDF5 absolute action target、eval prepared target 和 eval state；右侧画 signed error。橙色虚线用于检查 eval target 是否已经偏离 HDF5 absolute action。</figcaption></figure>
                  <figure><img src="{heatmap}" alt="error heatmap"><figcaption>绝对误差热力图：颜色越亮，eval replay 和 HDF5 record state 差异越大。</figcaption></figure>
                </div>
              </details>
              <table class="joint-table">
                <thead><tr><th>Top joint</th><th>max |eval-record|</th><th>mean |eval-record|</th><th>final |eval-record|</th><th>max |eval_target-HDF5_abs|</th><th>max |prepared-record|</th><th>max |eval-executed_target|</th></tr></thead>
                <tbody>{top_rows}</tbody>
              </table>
            </section>
            """
        )

    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Record Dataset Eval 对齐分析</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #6b7280;
      --line: #d8dee8;
      --accent: #2563eb;
      --danger: #dc2626;
      --ok: #059669;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); line-height: 1.55; }}
    header {{ background: #111827; color: white; padding: 40px 44px 32px; }}
    header h1 {{ margin: 0 0 10px; font-size: 34px; letter-spacing: 0; }}
    header p {{ margin: 0; color: #cbd5e1; max-width: 1180px; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 28px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 14px; margin: 0 0 22px; }}
    .metric, .panel, .episode {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06); }}
    .metric {{ padding: 18px; }}
    .metric .label {{ color: var(--muted); font-size: 13px; }}
    .metric .value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
    .panel {{ padding: 22px; margin-bottom: 22px; }}
    h2 {{ margin: 0 0 10px; font-size: 22px; }}
    h3 {{ margin: 0 0 10px; font-size: 16px; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; vertical-align: top; }}
    th {{ color: #374151; background: #f9fafb; font-weight: 650; }}
    .episode {{ padding: 22px; margin: 28px 0; }}
    .episode-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin-bottom: 14px; }}
    .episode p {{ margin: 0; color: var(--muted); }}
    video {{ width: 100%; max-height: 560px; background: #000; border-radius: 8px; border: 1px solid var(--line); }}
    .video-grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
    .plot-grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; margin-top: 18px; }}
    figure {{ margin: 0; background: #fbfcfe; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    figure img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ padding: 10px 12px; color: var(--muted); font-size: 13px; border-top: 1px solid var(--line); }}
    details {{ margin-top: 16px; }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 650; margin: 10px 0; }}
    .summary-figure {{ margin-top: 18px; }}
    .callout {{ border-left: 4px solid var(--danger); padding: 12px 14px; background: #fff7f7; margin: 14px 0 0; }}
    .ok {{ border-left-color: var(--ok); background: #f0fdf4; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    @media (max-width: 900px) {{ .metrics {{ grid-template-columns: 1fr 1fr; }} main {{ padding: 16px; }} header {{ padding: 28px 20px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Record Dataset Eval 对齐分析</h1>
    <p>数据集：<code>{html.escape(str(dataset_file))}</code>；eval 输出：<code>{html.escape(str(eval_output_dir))}</code>。本报告把 HDF5 record 视频、eval replay 视频、action target、实际 sim joint state 和误差放在一起对比。</p>
  </header>
  <main>
    <div class="metrics">
      <div class="metric"><div class="label">样本数</div><div class="value">{len(manifest)}</div></div>
      <div class="metric"><div class="label">全局 max |eval-record|</div><div class="value">{fmt_float(np.nanmax(max_errors) if max_errors else np.nan)}</div></div>
      <div class="metric"><div class="label">平均 mean |eval-record|</div><div class="value">{fmt_float(np.nanmean(mean_errors) if mean_errors else np.nan)}</div></div>
      <div class="metric"><div class="label">最坏 episode</div><div class="value">{worst_item['episode_ix']}</div></div>
    </div>

    <section class="panel">
      <h2>先看结论</h2>
      <p>这份 HDF5 检测到的 record sim 参数是 <code>dt={html.escape(str(record_sim_args.get('dt', 'unknown')))}</code>、<code>decimation={html.escape(str(record_sim_args.get('decimation', 'unknown')))}</code>、<code>render_interval={html.escape(str(record_sim_args.get('render_interval', 'unknown')))}</code>。当前 replay 使用 <code>decimation={html.escape(str(record_cfg.get('decimation', 'unknown')))}</code>，来源是 <code>{html.escape(str(record_cfg.get('decimation_source', 'unknown')))}</code>。</p>
      <p>报告中的 action 图已经把 <code>base_x/y/z</code> 的 HDF5 raw delta 转成 absolute target：<code>HDF5_abs = record_obs_base + HDF5_delta</code>。arm / gripper 本来就是 absolute action，保持原值。因此 action 图里所有曲线现在都在同一个 absolute joint target 坐标系里。</p>
      <p>真正关键的是三条差值：<code>eval_prepared_abs - HDF5_abs_action</code>、<code>prepared_abs - record_state</code> 和 <code>eval_state_after - record_state</code>。如果第一条接近 0，说明 eval target 与 record action 对齐；如果后两条仍大，说明 record action target 本身和 record state 有差，或 eval_state 闭环积分/物理执行产生了偏移。</p>
      <div class="callout">
        当前默认报告使用 <code>base_relative_reference=eval_state</code>，这是普通 policy eval 的真实语义。它会暴露闭环 delta action 的漂移风险。
      </div>
      <div class="callout ok">
        {f"诊断模式 <code>base_relative_reference=record_obs</code> 的结果：global max {fmt_float(diag['global_max'])}，mean {fmt_float(diag['mean_of_mean'])}。这说明剩余大部分误差来自 eval_state delta 积分漂移，而不是 HDF5 原始绝对 target 无法执行。" if diag else "未提供 record_obs 参考诊断输出。"}
      </div>
    </section>

    <section class="panel">
      <h2>怎么读图</h2>
      <p>每个 sample 先看 summary 图。左上柱状图告诉你哪个 joint 最大；右上按 base / arm / gripper 分组画误差随时间变化；下面三张只看 base 的 record state、record action abs、eval state 和 eval target。只有需要追细节时再展开完整图。</p>
      <p>经验判断：arm/finger 误差在 0.01 到 0.05 rad 级别通常已经说明 action 和 joint 顺序基本对齐；base 如果后段变大，先看橙色 record action abs 和绿色 eval target 是否一致。不一致通常是 relative base delta 基于 eval 当前状态积分导致，属于闭环 replay 的累计偏差。</p>
    </section>

    <section class="panel">
      <h2>样本总览</h2>
      <table>
        <thead><tr><th>Rank</th><th>Episode</th><th>Demo</th><th>max |eval-record|</th><th>mean |eval-record|</th><th>Top joints</th><th>Detail</th></tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table>
    </section>

    {''.join(cards_html)}
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(doc, encoding="utf-8")
    (output_dir / "analysis_report_zh.html").write_text(doc, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_file = Path(args.dataset_file).resolve()
    eval_output_dir = Path(args.eval_output_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path("outputs/debug/eval_align") / eval_output_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    per_episode_path = eval_output_dir / "per_episode_results.csv"
    trace_path = eval_output_dir / "action_trace_joint_states.csv"
    rows = read_csv_rows(per_episode_path)
    trace_grouped = rows_by_episode(read_csv_rows(trace_path))

    with h5py.File(dataset_file, "r") as f:
        available_demos = set(sorted_demo_keys(f["data"]))

    eligible = [row for row in rows if row.get("demo_key") in available_demos]
    if not eligible:
        raise ValueError("No eval rows match HDF5 demo keys.")

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(eligible))[: min(args.n, len(eligible))]
    selected = [eligible[int(ix)] for ix in indices]
    if args.include_all_rows:
        selected = eligible

    manifest: list[dict[str, Any]] = []
    selected_rows_path = output_dir / "selected_episodes.csv"
    with selected_rows_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected)

    for rank, row in enumerate(selected):
        episode_ix = int(row["episode_ix"])
        demo_key = row["demo_key"]
        prefix = f"sample_{rank:02d}_episode_{episode_ix}_{demo_key}"

        record_frames = read_hdf5_rgb(dataset_file, demo_key, args.camera)
        record_video_path = output_dir / f"{prefix}_hdf5_{args.camera}.mp4"
        write_video(record_video_path, [resize_to_height(frame, args.height) for frame in record_frames], args.fps)

        eval_video_src = resolve_eval_video(row, eval_output_dir)
        eval_frames = read_video(eval_video_src) if eval_video_src else []
        eval_video_path = output_dir / f"{prefix}_eval.mp4"
        if eval_video_src:
            shutil.copy2(eval_video_src, eval_video_path)

        side_by_side_path = output_dir / f"{prefix}_side_by_side.mp4"
        make_side_by_side_video(
            record_frames=record_frames,
            eval_frames=eval_frames,
            out_path=side_by_side_path,
            fps=args.fps,
            height=args.height,
            record_label=f"HDF5 {demo_key} {args.camera}",
            eval_label=f"eval episode {episode_ix}",
        )

        trace_rows = trace_grouped.get(episode_ix, [])
        trace_subset_path = output_dir / f"{prefix}_trace.csv"
        write_trace_subset(trace_subset_path, trace_rows)
        data = trace_arrays(trace_rows)

        action_plot_path = output_dir / f"{prefix}_actions.png"
        joint_plot_path = output_dir / f"{prefix}_joint_states.png"
        error_heatmap_path = output_dir / f"{prefix}_error_heatmap.png"
        summary_plot_path = output_dir / f"{prefix}_summary.png"
        plot_episode_summary(data, summary_plot_path)
        plot_action_grid(data, action_plot_path)
        plot_joint_grid(data, joint_plot_path)
        plot_error_heatmap(data, error_heatmap_path)
        trace_summary = summarize_trace_data(data)

        manifest.append(
            {
                "rank": rank,
                "anchor": f"sample-{rank:02d}",
                "episode_ix": episode_ix,
                "demo_key": demo_key,
                "record_success": row.get("record_success", ""),
                "eval_success": row.get("success", ""),
                "max_abs_eval_vs_record_state": row.get("max_abs_eval_vs_record_state", ""),
                "mean_abs_eval_vs_record_state": row.get("mean_abs_eval_vs_record_state", ""),
                "record_video": str(record_video_path),
                "eval_video": str(eval_video_path) if eval_video_src else "",
                "side_by_side_video": str(side_by_side_path),
                "summary_plot": str(summary_plot_path),
                "action_plot": str(action_plot_path),
                "joint_plot": str(joint_plot_path),
                "error_heatmap": str(error_heatmap_path),
                "trace_csv": str(trace_subset_path),
                "trace_summary": trace_summary,
            }
        )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_summary_csv(output_dir, manifest)
    diagnostic_eval_output_dir = (
        Path(args.diagnostic_eval_output_dir).resolve() if args.diagnostic_eval_output_dir else None
    )
    make_html_index(
        output_dir,
        manifest,
        dataset_file=dataset_file,
        eval_output_dir=eval_output_dir,
        diagnostic_eval_output_dir=diagnostic_eval_output_dir,
    )
    print(json.dumps({"output_dir": str(output_dir), "n": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
