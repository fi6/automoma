#!/usr/bin/env python3
"""Find and record microwave_7221 mid-run drive disengagement examples.

This is a targeted debug workflow for reviewing "mid-drop" failures:

1. replay drive mode without HDF5/video recording;
2. scan replay metrics for trajectories that get close to the handle mid-run
   and end far from it;
3. write drive joint/EEF/handle traces for selected candidates;
4. record 1080p drive and set-state comparison videos for the same indices;
5. assemble a per-trajectory review package.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py

from tools.debug.record_drive_and_set.run_record_drive_and_set import (
    DEFAULT_AUTOMOMA_PYTHON,
    HIGHRES_ENV,
    HIGHRES_ENV_NAME,
    ISAACLAB_ARENA,
    RECORD_SCRIPT,
    REPLAY_SCRIPT,
    REPO_ROOT,
    VIDEO_SCRIPT,
    RunSpec,
    _base_env,
    _hdf5_summary,
    _run_subprocess,
    _traj_file,
    _write_json,
    _write_success_csv,
)


OBJECT_NAME = "microwave_7221"
SCENE_NAME = "scene_0_seed_0"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "debug" / "2026-06-11_record_drive_and_set" / "output" / "middrop_7221"


@dataclass(frozen=True)
class Candidate:
    rank: int
    traj_index: int
    success: bool
    final_door_open: bool
    final_engaged: bool
    final_handle_distance: float
    min_handle_distance: float
    min_handle_distance_step: int
    release_delta: float
    max_openness: float
    max_openness_step: int
    final_openness: float
    num_steps: int
    score: float


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _int(row: dict[str, str], key: str, default: int = -1) -> int:
    try:
        return int(float(row.get(key, "")))
    except (TypeError, ValueError):
        return default


def _bool(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes", "y"}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_total_trajectories() -> int:
    import torch

    traj_file = _traj_file(OBJECT_NAME, SCENE_NAME)
    data = torch.load(traj_file, map_location="cpu")
    return int(data["traj_success"].shape[0])


def _write_indices(path: Path, indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(idx) for idx in indices) + "\n", encoding="utf-8")


def _command_base(args: argparse.Namespace) -> list[str]:
    return [
        args.python_executable,
        str(REPLAY_SCRIPT),
        "--traj_file",
        str(_traj_file(OBJECT_NAME, SCENE_NAME)),
        "--interpolated",
        str(args.interpolated),
        "--interpolation_type",
        args.interpolation_type,
        "--decimation",
        str(args.decimation),
        "--init_steps",
        str(args.init_steps),
        "--robot_object_static_friction",
        str(args.static_friction),
        "--robot_object_dynamic_friction",
        str(args.dynamic_friction),
    ]


def _replay_command(
    *,
    indices_file: Path,
    count: int,
    metrics_file: Path,
    trace: bool,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        *_command_base(args),
        "--num_episodes",
        str(count),
        "--episode_indices_file",
        str(indices_file.resolve()),
        "--metrics",
        "--metrics_file",
        str(metrics_file.resolve()),
    ]
    if trace:
        cmd.extend(["--debug", "--debug_joint_tracking_topk", "12", "--debug_joint_tracking_fk_link", "ee_link"])
    else:
        cmd.append("--no_step_trace")
    if args.headless:
        cmd.append("--headless")
    cmd.extend(
        [
            "summit_franka_open_door",
            "--object_name",
            OBJECT_NAME,
            "--scene_name",
            SCENE_NAME,
            "--object_center",
        ]
    )
    return cmd


def _scan_order(args: argparse.Namespace) -> list[int]:
    total = _load_total_trajectories()
    indices = list(range(total))
    rng = random.Random(args.seed)
    rng.shuffle(indices)

    pinned = []
    for item in args.include_indices.split(","):
        item = item.strip()
        if item:
            pinned.append(int(item))
    pinned = [idx for idx in pinned if 0 <= idx < total]
    ordered = pinned + [idx for idx in indices if idx not in set(pinned)]
    if args.scan_limit > 0:
        ordered = ordered[: args.scan_limit]
    return ordered


def _score_candidates(rows: list[dict[str, str]], args: argparse.Namespace) -> list[Candidate]:
    candidates: list[Candidate] = []
    for row in rows:
        success = _bool(row, "success")
        final_door_open = _bool(row, "final_door_open")
        final_engaged = _bool(row, "final_engaged")
        final_handle = _float(row, "final_handle_distance")
        min_handle = _float(row, "min_handle_distance")
        min_step = _int(row, "min_handle_distance_step")
        max_open = _float(row, "max_openness")
        max_open_step = _int(row, "max_openness_step")
        final_open = _float(row, "final_openness")
        num_steps = _int(row, "num_steps")
        release_delta = final_handle - min_handle

        if success:
            continue
        if final_engaged:
            continue
        if min_handle > args.approach_threshold:
            continue
        if final_handle < args.final_distance_threshold:
            continue
        if release_delta < args.release_delta_threshold:
            continue
        if min_step < args.min_contact_step:
            continue
        if num_steps > 0 and min_step > num_steps - args.final_margin_steps:
            continue
        if max_open < args.min_max_openness:
            continue

        mid_bonus = 0.0
        if num_steps > 0:
            # Prefer examples where the closest approach is not at either edge.
            center = 0.5 * num_steps
            mid_bonus = 1.0 - min(abs(min_step - center) / center, 1.0)
        score = release_delta + 0.35 * max_open + 0.1 * mid_bonus
        candidates.append(
            Candidate(
                rank=0,
                traj_index=_int(row, "traj_index"),
                success=success,
                final_door_open=final_door_open,
                final_engaged=final_engaged,
                final_handle_distance=final_handle,
                min_handle_distance=min_handle,
                min_handle_distance_step=min_step,
                release_delta=release_delta,
                max_openness=max_open,
                max_openness_step=max_open_step,
                final_openness=final_open,
                num_steps=num_steps,
                score=score,
            )
        )

    candidates.sort(key=lambda item: (item.score, item.max_openness, item.release_delta), reverse=True)
    return [
        Candidate(**{**asdict(candidate), "rank": rank})
        for rank, candidate in enumerate(candidates[: args.target_count], start=1)
    ]


def _scan_metrics(args: argparse.Namespace) -> list[dict[str, str]]:
    scan_dir = args.output_root / "scan"
    order = _scan_order(args)
    all_rows: list[dict[str, str]] = []
    scanned = 0

    for chunk_id, start in enumerate(range(0, len(order), args.chunk_size)):
        chunk = order[start : start + args.chunk_size]
        indices_file = scan_dir / f"indices_{chunk_id:04d}.txt"
        metrics_file = scan_dir / f"metrics_{chunk_id:04d}.csv"
        log_file = scan_dir / f"replay_{chunk_id:04d}.log"
        _write_indices(indices_file, chunk)

        if metrics_file.exists() and not args.force_scan:
            print(f"[scan][skip] chunk={chunk_id} rows={len(_read_csv_rows(metrics_file))}", flush=True)
        else:
            cmd = _replay_command(
                indices_file=indices_file,
                count=len(chunk),
                metrics_file=metrics_file,
                trace=False,
                args=args,
            )
            print(f"[scan] chunk={chunk_id} episodes={len(chunk)}", flush=True)
            rc = _run_subprocess(cmd, cwd=ISAACLAB_ARENA, env=_base_env(args), log_path=log_file, dry_run=args.dry_run)
            if rc != 0:
                raise RuntimeError(f"scan chunk {chunk_id} failed with returncode={rc}; see {log_file}")

        rows = _read_csv_rows(metrics_file)
        if not rows and not args.dry_run:
            raise RuntimeError(f"scan chunk {chunk_id} wrote no metrics rows; see {log_file}")
        all_rows.extend(rows)
        scanned += len(rows)
        current = _score_candidates(all_rows, args)
        print(f"[scan] scanned={scanned} middrop_candidates={len(current)}", flush=True)
        if len(current) >= args.target_count and scanned >= args.min_scan_episodes:
            break

    return all_rows


def _write_candidates(candidates: list[Candidate], args: argparse.Namespace) -> None:
    rows = [asdict(candidate) for candidate in candidates]
    _write_json(
        args.output_root / "candidates.json",
        {
            "object_name": OBJECT_NAME,
            "scene_name": SCENE_NAME,
            "selection_rule": {
                "approach_threshold": args.approach_threshold,
                "final_distance_threshold": args.final_distance_threshold,
                "release_delta_threshold": args.release_delta_threshold,
                "min_contact_step": args.min_contact_step,
                "final_margin_steps": args.final_margin_steps,
                "min_max_openness": args.min_max_openness,
            },
            "candidates": rows,
        },
    )
    _write_csv(
        args.output_root / "candidates.csv",
        rows,
        [
            "rank",
            "traj_index",
            "score",
            "success",
            "final_door_open",
            "final_engaged",
            "min_handle_distance",
            "min_handle_distance_step",
            "final_handle_distance",
            "release_delta",
            "max_openness",
            "max_openness_step",
            "final_openness",
            "num_steps",
        ],
    )
    _write_indices(args.output_root / "candidate_indices.txt", [candidate.traj_index for candidate in candidates])


def _run_trace(candidates: list[Candidate], args: argparse.Namespace) -> None:
    trace_dir = args.output_root / "trace"
    indices_file = trace_dir / "drive_trace_indices.txt"
    metrics_file = trace_dir / "drive_trace_metrics.csv"
    log_file = trace_dir / "drive_trace.log"
    indices = [candidate.traj_index for candidate in candidates]
    _write_indices(indices_file, indices)
    if metrics_file.exists() and (trace_dir / "debug_curves").exists() and not args.force_trace:
        print("[trace][skip] existing drive trace artifacts", flush=True)
        return
    cmd = _replay_command(
        indices_file=indices_file,
        count=len(indices),
        metrics_file=metrics_file,
        trace=True,
        args=args,
    )
    print(f"[trace] drive episodes={len(indices)}", flush=True)
    rc = _run_subprocess(cmd, cwd=ISAACLAB_ARENA, env=_base_env(args), log_path=log_file, dry_run=args.dry_run)
    if rc != 0:
        raise RuntimeError(f"drive trace failed with returncode={rc}; see {log_file}")


def _record_command(spec: RunSpec, dataset_file: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python_executable,
        str(RECORD_SCRIPT),
        "--environment",
        HIGHRES_ENV,
        "--enable_cameras",
        "--mobile_base_relative",
        "--traj_file",
        str(_traj_file(spec.object_name, spec.scene_name)),
        "--dataset_file",
        str(dataset_file),
        "--num_episodes",
        str(len(spec.traj_indices)),
        "--episode_indices",
        ",".join(str(idx) for idx in spec.traj_indices),
        "--interpolated",
        str(args.interpolated),
        "--interpolation_type",
        args.interpolation_type,
        "--decimation",
        str(args.decimation),
        "--init_steps",
        str(args.init_steps),
        "--validate_record_success",
        "--keep_failed_record_demos",
        "--debug",
        "--debug_joint_tracking_topk",
        "12",
        "--debug_joint_tracking_fk_link",
        "ee_link",
        "--robot_object_static_friction",
        str(args.static_friction),
        "--robot_object_dynamic_friction",
        str(args.dynamic_friction),
    ]
    if spec.set_state:
        cmd.extend(["--set_state", "--object_joint_names", args.object_joint_names])
    if args.headless:
        cmd.append("--headless")
    cmd.extend(
        [
            HIGHRES_ENV_NAME,
            "--object_name",
            spec.object_name,
            "--scene_name",
            spec.scene_name,
            "--object_center",
            "--camera_width",
            str(spec.width),
            "--camera_height",
            str(spec.height),
            "--camera_data_types",
            args.camera_data_types,
        ]
    )
    return cmd


def _record_mode(mode: str, candidates: list[Candidate], args: argparse.Namespace) -> Path:
    record_dir = args.output_root / "recordings" / mode
    indices = [candidate.traj_index for candidate in candidates]
    spec = RunSpec(
        object_name=OBJECT_NAME,
        scene_name=SCENE_NAME,
        mode=mode,
        traj_indices=indices,
        width=args.width,
        height=args.height,
    )
    joined = "-".join(str(idx) for idx in indices)
    dataset_file = record_dir / f"{OBJECT_NAME}_{SCENE_NAME}_{mode}_midrelease10_traj{joined}_{args.width}x{args.height}.hdf5"
    hdf5_summary_path = record_dir / "hdf5_summary.json"
    success_csv_path = record_dir / "record_success.csv"
    log_path = record_dir / "record.log"
    status_path = record_dir / "status.json"

    if dataset_file.exists() and hdf5_summary_path.exists() and not args.force_record:
        print(f"[record][skip] {mode} existing HDF5", flush=True)
        return dataset_file

    cmd = _record_command(spec, dataset_file, args)
    _write_json(
        status_path,
        {
            **asdict(spec),
            "dataset_file": str(dataset_file),
            "record_cmd": cmd,
            "status": "dry_run" if args.dry_run else "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    print(f"[record] {mode} episodes={len(indices)}", flush=True)
    started = time.time()
    rc = _run_subprocess(cmd, cwd=ISAACLAB_ARENA, env=_base_env(args), log_path=log_path, dry_run=args.dry_run)
    if rc != 0:
        _write_json(status_path, {"status": "failed", "returncode": rc, "dataset_file": str(dataset_file)})
        raise RuntimeError(f"{mode} record failed with returncode={rc}; see {log_path}")

    summary = _hdf5_summary(dataset_file)
    _write_json(hdf5_summary_path, summary)
    _write_success_csv(success_csv_path, summary)
    _write_json(
        status_path,
        {
            **asdict(spec),
            "dataset_file": str(dataset_file),
            "status": "completed",
            "returncode": rc,
            "elapsed_s": time.time() - started,
            "hdf5_summary": str(hdf5_summary_path),
            "record_success_csv": str(success_csv_path),
        },
    )
    return dataset_file


def _sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    def key_index(key: str) -> int:
        try:
            return int(key.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 10**12

    return sorted(data_group.keys(), key=key_index)


def _extract_videos(mode: str, dataset_file: Path, args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    video_dir = args.output_root / "recordings" / mode / "videos"
    summary_dir = args.output_root / "recordings" / mode / "video_summaries"
    results: dict[int, dict[str, Any]] = {}
    with h5py.File(dataset_file, "r") as f:
        data = f["data"]
        demos = [(key, int(data[key].attrs.get("traj_index", -1))) for key in _sorted_demo_keys(data)]

    for demo_key, traj_index in demos:
        summary_json = summary_dir / f"traj{traj_index}_{demo_key}_video_summary.json"
        if summary_json.exists() and not args.force_video:
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            results[traj_index] = summary
            print(f"[video][skip] {mode} traj={traj_index}", flush=True)
            continue
        cmd = [
            args.python_executable,
            str(VIDEO_SCRIPT),
            "--hdf5",
            str(dataset_file),
            "--output_dir",
            str(video_dir),
            "--camera",
            args.camera,
            "--demo_key",
            demo_key,
            "--fps",
            str(args.fps),
            "--require_width",
            str(args.width),
            "--require_height",
            str(args.height),
            "--summary_json",
            str(summary_json),
        ]
        log_path = args.output_root / "recordings" / mode / f"video_traj{traj_index}.log"
        print(f"[video] {mode} traj={traj_index} demo={demo_key}", flush=True)
        rc = _run_subprocess(cmd, cwd=REPO_ROOT, env=_base_env(args), log_path=log_path, dry_run=args.dry_run)
        if rc != 0:
            raise RuntimeError(f"{mode} video extraction failed for traj={traj_index}; see {log_path}")
        if summary_json.exists():
            results[traj_index] = json.loads(summary_json.read_text(encoding="utf-8"))
    return results


def _latest_matching_file(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime if path.exists() else 0)
    return matches[-1] if matches else None


def _filter_csv_by_traj(source: Path | None, dest: Path, traj_index: int) -> int:
    if source is None or not source.exists():
        return 0
    rows = _read_csv_rows(source)
    filtered = [row for row in rows if str(row.get("traj", row.get("traj_index", ""))) == str(traj_index)]
    if not filtered:
        return 0
    _write_csv(dest, filtered, list(filtered[0].keys()))
    return len(filtered)


def _copy_episode_plot(source_dir: Path, dest: Path, traj_index: int) -> bool:
    matches = list(source_dir.glob(f"**/*traj_{traj_index:06d}_joint_eef_gap.png"))
    if not matches:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0], dest)
    return True


def _record_success_by_traj(path: Path) -> dict[int, dict[str, str]]:
    rows = _read_csv_rows(path)
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            out[int(float(row["traj_index"]))] = row
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _assemble_review(
    candidates: list[Candidate],
    drive_videos: dict[int, dict[str, Any]],
    set_videos: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    review_dir = args.output_root / "review"
    if review_dir.exists() and args.force_package:
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    trace_curves = args.output_root / "trace" / "debug_curves"
    trace_handle_csv = _latest_matching_file(trace_curves, "*handle-tracking.csv")
    trace_joint_csv = _latest_matching_file(trace_curves, "*joint-tracking.csv")
    record_curves = {
        mode: args.output_root / "recordings" / mode / "debug_curves"
        for mode in ("drive", "set")
    }
    success_rows = {
        mode: _record_success_by_traj(args.output_root / "recordings" / mode / "record_success.csv")
        for mode in ("drive", "set")
    }

    summary_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        traj_dir = review_dir / f"traj{candidate.traj_index}"
        info = {
            **asdict(candidate),
            "object_name": OBJECT_NAME,
            "scene_name": SCENE_NAME,
            "drive_record_success": success_rows["drive"].get(candidate.traj_index, {}),
            "set_record_success": success_rows["set"].get(candidate.traj_index, {}),
        }
        _write_json(traj_dir / "info.json", info)

        for mode, videos in (("drive", drive_videos), ("set", set_videos)):
            mode_dir = traj_dir / mode
            video_summary = videos.get(candidate.traj_index, {})
            video_path = Path(video_summary.get("video_path", ""))
            if video_path.exists():
                mode_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video_path, mode_dir / "video.mp4")
                _write_json(mode_dir / "video_summary.json", video_summary)

            curves_dir = mode_dir / "curves"
            mode_curves = record_curves[mode]
            _filter_csv_by_traj(_latest_matching_file(mode_curves, "*handle-tracking.csv"), curves_dir / "handle_tracking.csv", candidate.traj_index)
            _filter_csv_by_traj(_latest_matching_file(mode_curves, "*joint-tracking.csv"), curves_dir / "joint_eef_tracking.csv", candidate.traj_index)
            _copy_episode_plot(mode_curves, curves_dir / "joint_eef_gap.png", candidate.traj_index)

        trace_dir = traj_dir / "drive" / "trace"
        _filter_csv_by_traj(trace_handle_csv, trace_dir / "handle_tracking.csv", candidate.traj_index)
        _filter_csv_by_traj(trace_joint_csv, trace_dir / "joint_eef_tracking.csv", candidate.traj_index)
        _copy_episode_plot(trace_curves, trace_dir / "joint_eef_gap.png", candidate.traj_index)

        drive_video = traj_dir / "drive" / "video.mp4"
        set_video = traj_dir / "set" / "video.mp4"
        drive_record = success_rows["drive"].get(candidate.traj_index, {})
        set_record = success_rows["set"].get(candidate.traj_index, {})
        summary_rows.append(
            {
                "rank": candidate.rank,
                "traj_index": candidate.traj_index,
                "selection_score": candidate.score,
                "scan_min_handle_distance": candidate.min_handle_distance,
                "scan_min_handle_distance_step": candidate.min_handle_distance_step,
                "scan_final_handle_distance": candidate.final_handle_distance,
                "scan_release_delta": candidate.release_delta,
                "scan_max_openness": candidate.max_openness,
                "scan_max_openness_step": candidate.max_openness_step,
                "drive_success": drive_record.get("success", ""),
                "drive_final_door_open": drive_record.get("final_door_open", ""),
                "drive_final_engaged": drive_record.get("final_engaged", ""),
                "drive_final_openness": drive_record.get("final_door_openness", ""),
                "drive_final_handle_distance": drive_record.get("final_handle_distance", ""),
                "set_success": set_record.get("success", ""),
                "set_final_door_open": set_record.get("final_door_open", ""),
                "set_final_engaged": set_record.get("final_engaged", ""),
                "set_final_openness": set_record.get("final_door_openness", ""),
                "set_final_handle_distance": set_record.get("final_handle_distance", ""),
                "drive_video": str(drive_video.relative_to(args.output_root)) if drive_video.exists() else "",
                "set_video": str(set_video.relative_to(args.output_root)) if set_video.exists() else "",
            }
        )

    _write_csv(
        review_dir / "summary.csv",
        summary_rows,
        [
            "rank",
            "traj_index",
            "selection_score",
            "scan_min_handle_distance",
            "scan_min_handle_distance_step",
            "scan_final_handle_distance",
            "scan_release_delta",
            "scan_max_openness",
            "scan_max_openness_step",
            "drive_success",
            "drive_final_door_open",
            "drive_final_engaged",
            "drive_final_openness",
            "drive_final_handle_distance",
            "set_success",
            "set_final_door_open",
            "set_final_engaged",
            "set_final_openness",
            "set_final_handle_distance",
            "drive_video",
            "set_video",
        ],
    )
    _write_json(review_dir / "summary.json", {"rows": summary_rows})
    readme = f"""# microwave_7221 mid-run disengagement review

Object: `{OBJECT_NAME}`
Scene: `{SCENE_NAME}`
Resolution: `{args.width}x{args.height}`

Each `traj*/` directory contains one drive/set pair:
- `drive/video.mp4`
- `set/video.mp4`
- `info.json`
- `drive/curves/` and `set/curves/`
- `drive/trace/` from the no-HDF5 drive replay pass

`summary.csv` includes both scan-stage metrics and the final fields from the
formal 1080p recording pass.

Selection rule:
- drive failed final eval
- closest handle distance <= {args.approach_threshold}
- closest step >= {args.min_contact_step}
- final handle distance >= {args.final_distance_threshold}
- final-minus-min handle distance >= {args.release_delta_threshold}
- max openness >= {args.min_max_openness}
"""
    (review_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target_count", type=int, default=10)
    parser.add_argument("--scan_limit", type=int, default=600, help="0 scans all trajectories.")
    parser.add_argument("--min_scan_episodes", type=int, default=250)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--include_indices", default="2309", help="Comma-separated indices forced to the front of scan order.")
    parser.add_argument("--approach_threshold", type=float, default=0.08)
    parser.add_argument("--final_distance_threshold", type=float, default=0.12)
    parser.add_argument("--release_delta_threshold", type=float, default=0.08)
    parser.add_argument("--min_contact_step", type=int, default=20)
    parser.add_argument("--final_margin_steps", type=int, default=10)
    parser.add_argument("--min_max_openness", type=float, default=0.05)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--camera", default="fix_local_rgb")
    parser.add_argument("--camera_data_types", default="rgb")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--interpolated", type=int, default=5)
    parser.add_argument("--interpolation_type", default="cubic")
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument("--init_steps", type=int, default=5)
    parser.add_argument("--static_friction", type=float, default=1.0)
    parser.add_argument("--dynamic_friction", type=float, default=1.0)
    parser.add_argument("--object_joint_names", default="joint_0")
    parser.add_argument(
        "--python_executable",
        default=os.environ.get(
            "AUTOMOMA_PYTHON",
            str(DEFAULT_AUTOMOMA_PYTHON if DEFAULT_AUTOMOMA_PYTHON.exists() else Path(sys.executable)),
        ),
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_scan", action="store_true")
    parser.add_argument("--force_trace", action="store_true")
    parser.add_argument("--force_record", action="store_true")
    parser.add_argument("--force_video", action="store_true")
    parser.add_argument("--force_package", action="store_true")
    parser.add_argument("--skip_scan", action="store_true")
    parser.add_argument("--skip_trace", action="store_true")
    parser.add_argument("--skip_record", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output_root"] = str(args.output_root)
    config["python_executable"] = str(args.python_executable)
    _write_json(args.output_root / "config.json", config)

    if args.skip_scan:
        all_rows = []
        for metrics_file in sorted((args.output_root / "scan").glob("metrics_*.csv")):
            all_rows.extend(_read_csv_rows(metrics_file))
    else:
        all_rows = _scan_metrics(args)

    _write_csv(args.output_root / "scan_metrics_merged.csv", all_rows, list(all_rows[0].keys()) if all_rows else [])
    candidates = _score_candidates(all_rows, args)
    if len(candidates) < args.target_count:
        raise RuntimeError(f"Found only {len(candidates)} mid-drop candidates; loosen thresholds or increase scan_limit.")
    _write_candidates(candidates, args)
    print("[candidates] " + ",".join(str(candidate.traj_index) for candidate in candidates), flush=True)

    if not args.skip_trace:
        _run_trace(candidates, args)

    drive_videos: dict[int, dict[str, Any]] = {}
    set_videos: dict[int, dict[str, Any]] = {}
    if not args.skip_record:
        drive_dataset = _record_mode("drive", candidates, args)
        set_dataset = _record_mode("set", candidates, args)
        drive_videos = _extract_videos("drive", drive_dataset, args)
        set_videos = _extract_videos("set", set_dataset, args)

    if not drive_videos:
        for summary_path in sorted((args.output_root / "recordings" / "drive" / "video_summaries").glob("*.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            key = int(summary_path.name.split("traj", 1)[1].split("_", 1)[0])
            drive_videos[key] = summary
    if not set_videos:
        for summary_path in sorted((args.output_root / "recordings" / "set" / "video_summaries").glob("*.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            key = int(summary_path.name.split("traj", 1)[1].split("_", 1)[0])
            set_videos[key] = summary

    _assemble_review(candidates, drive_videos, set_videos, args)
    print(f"[done] output_root={args.output_root}", flush=True)
    print(f"[done] review={args.output_root / 'review'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
