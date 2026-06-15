#!/usr/bin/env python3
"""Record paired drive and set-state AutoMoMa debug artifacts.

Default behavior mirrors ``data/automoma/joint_error``:
- use the same five objects and selected raw trajectory indices,
- write per-object/per-mode HDF5, logs, status JSON, summaries, curves, and mp4,
- record one trajectory per object/mode unless ``--episodes_per_mode`` is changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py


REPO_ROOT = Path(__file__).resolve().parents[3]
ISAACLAB_ARENA = REPO_ROOT / "third_party" / "IsaacLab-Arena"
DEFAULT_SELECTION = REPO_ROOT / "data" / "automoma" / "joint_error" / "scene_0_seed_0_random10_seed20260608.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "debug" / "2026-06-11_record_drive_and_set" / "output"
RECORD_SCRIPT = REPO_ROOT / "tools" / "debug" / "record_drive_and_set" / "record_selected_automoma.py"
VIDEO_SCRIPT = REPO_ROOT / "tools" / "debug" / "record_drive_and_set" / "hdf5_to_video.py"
REPLAY_SCRIPT = REPO_ROOT / "tools" / "debug" / "record_drive_and_set" / "replay_selected_automoma.py"
HIGHRES_ENV = "tools.debug.record_drive_and_set.highres_open_door:HighResSummitFrankaOpenDoorEnvironment"
HIGHRES_ENV_NAME = "summit_franka_open_door_highres"
DEFAULT_AUTOMOMA_PYTHON = Path("/home/xinhai/miniconda3/envs/automoma/bin/python")


@dataclass(frozen=True)
class RunSpec:
    object_name: str
    scene_name: str
    mode: str
    traj_indices: list[int]
    width: int
    height: int

    @property
    def set_state(self) -> bool:
        return self.mode == "set"

    @property
    def name(self) -> str:
        joined = "-".join(str(idx) for idx in self.traj_indices)
        return f"{self.object_name}_{self.scene_name}_{self.mode}_traj{joined}_{self.width}x{self.height}"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def _load_selection(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _traj_file(object_name: str, scene_name: str) -> Path:
    return REPO_ROOT / "data" / "trajs" / "summit_franka" / object_name / scene_name / "train" / "traj_data_train.pt"


def _hdf5_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "demos": [],
        "data_attrs": {},
    }
    if not path.exists():
        return summary

    with h5py.File(path, "r") as f:
        data = f.get("data")
        if data is None:
            return summary
        summary["data_attrs"] = {key: _jsonable_attr(value) for key, value in data.attrs.items()}
        for demo_key in _sorted_demo_keys(data):
            demo = data[demo_key]
            demo_summary = {
                "demo_key": demo_key,
                "attrs": {key: _jsonable_attr(value) for key, value in demo.attrs.items()},
                "datasets": {},
            }

            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    demo_summary["datasets"][name] = {
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                    }

            demo.visititems(visit)
            summary["demos"].append(demo_summary)
    return summary


def _write_success_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = []
    for demo in summary.get("demos", []):
        attrs = demo.get("attrs", {})
        rows.append(
            {
                "demo_key": demo.get("demo_key", ""),
                "traj_index": attrs.get("traj_index", ""),
                "success": attrs.get("success", ""),
                "final_door_open": attrs.get("final_door_open", ""),
                "final_engaged": attrs.get("final_engaged", ""),
                "final_door_openness": attrs.get("final_door_openness", ""),
                "final_handle_distance": attrs.get("final_handle_distance", ""),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "demo_key",
                "traj_index",
                "success",
                "final_door_open",
                "final_engaged",
                "final_door_openness",
                "final_handle_distance",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    def demo_index(key: str) -> int:
        try:
            return int(key.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 10**12

    return sorted(data_group.keys(), key=demo_index)


def _jsonable_attr(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run_subprocess(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        if dry_run:
            return 0
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        return int(proc.returncode)


def _base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["AUTOMOMA_OBJECT_ROOT"] = str(REPO_ROOT / "assets" / "object")
    env["AUTOMOMA_SCENE_ROOT"] = str(REPO_ROOT / "assets" / "scene" / "infinigen" / "kitchen_1130")
    env["AUTOMOMA_ROBOT_ROOT"] = str(REPO_ROOT / "assets" / "robot")
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env.setdefault("ACCEPT_EULA", "Y")
    return env


def _probe_command(
    *,
    object_name: str,
    scene_name: str,
    mode: str,
    traj_file: Path,
    traj_indices: list[int],
    metrics_file: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        args.python_executable,
        str(REPLAY_SCRIPT),
        "--traj_file",
        str(traj_file),
        "--num_episodes",
        str(len(traj_indices)),
        "--episode_indices",
        ",".join(str(idx) for idx in traj_indices),
        "--interpolated",
        str(args.interpolated),
        "--interpolation_type",
        args.interpolation_type,
        "--decimation",
        str(args.decimation),
        "--init_steps",
        str(args.init_steps),
        "--metrics",
        "--metrics_file",
        str(metrics_file),
        "--no_step_trace",
        "--robot_object_static_friction",
        str(args.static_friction),
        "--robot_object_dynamic_friction",
        str(args.dynamic_friction),
    ]
    if args.headless:
        cmd.append("--headless")
    if mode == "set":
        cmd.extend(["--set_state", "--object_joint_names", args.object_joint_names])
    cmd.extend(
        [
            "summit_franka_open_door",
            "--object_name",
            object_name,
            "--scene_name",
            scene_name,
            "--object_center",
        ]
    )
    return cmd


def _read_metrics(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {int(row["traj_index"]): row for row in rows}


def _truthy(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _probe_mode(
    *,
    object_name: str,
    scene_name: str,
    mode: str,
    traj_indices: list[int],
    output_root: Path,
    args: argparse.Namespace,
) -> tuple[int, dict[int, dict[str, str]]]:
    traj_file = _traj_file(object_name, scene_name)
    if not traj_file.exists():
        raise FileNotFoundError(f"Trajectory file missing: {traj_file}")
    probe_dir = output_root / "probes" / object_name / mode
    metrics_file = probe_dir / "metrics.csv"
    log_path = probe_dir / "probe.log"
    cmd = _probe_command(
        object_name=object_name,
        scene_name=scene_name,
        mode=mode,
        traj_file=traj_file,
        traj_indices=traj_indices,
        metrics_file=metrics_file,
        args=args,
    )
    _write_json(
        probe_dir / "status.json",
        {
            "object_name": object_name,
            "scene_name": scene_name,
            "mode": mode,
            "traj_indices": traj_indices,
            "metrics_file": str(metrics_file),
            "cmd": cmd,
            "status": "dry_run" if args.dry_run else "running",
        },
    )
    print(f"[probe] {object_name} {mode} trajs={traj_indices}", flush=True)
    rc = _run_subprocess(cmd, cwd=ISAACLAB_ARENA, env=_base_env(args), log_path=log_path, dry_run=args.dry_run)
    metrics = _read_metrics(metrics_file) if rc == 0 and metrics_file.exists() else {}
    _write_json(
        probe_dir / "status.json",
        {
            "object_name": object_name,
            "scene_name": scene_name,
            "mode": mode,
            "traj_indices": traj_indices,
            "metrics_file": str(metrics_file),
            "cmd": cmd,
            "status": "completed" if rc == 0 else "failed",
            "returncode": rc,
            "metrics_rows": len(metrics),
        },
    )
    return rc, metrics


def _select_pair_index(
    *,
    object_name: str,
    scene_name: str,
    candidate_indices: list[int],
    output_root: Path,
    args: argparse.Namespace,
) -> int:
    if args.dry_run or not args.search_pairs:
        return int(candidate_indices[0])

    set_rc, set_metrics = _probe_mode(
        object_name=object_name,
        scene_name=scene_name,
        mode="set",
        traj_indices=candidate_indices,
        output_root=output_root,
        args=args,
    )
    drive_rc, drive_metrics = _probe_mode(
        object_name=object_name,
        scene_name=scene_name,
        mode="drive",
        traj_indices=candidate_indices,
        output_root=output_root,
        args=args,
    )
    if set_rc != 0 or drive_rc != 0:
        raise RuntimeError(f"Probe failed for {object_name}: set_rc={set_rc}, drive_rc={drive_rc}")

    pair_rows = []
    for index in candidate_indices:
        set_row = set_metrics.get(index, {})
        drive_row = drive_metrics.get(index, {})
        set_eval_success = _truthy(set_row.get("success"))
        set_door_open = _truthy(set_row.get("final_door_open"))
        set_pair_success = set_eval_success if args.set_success_rule == "eval" else set_door_open
        pair_rows.append(
            {
                "traj_index": index,
                "set_success": set_eval_success,
                "set_door_open": set_door_open,
                "set_pair_success": set_pair_success,
                "drive_success": _truthy(drive_row.get("success")),
                "set_final_openness": set_row.get("final_door_openness", ""),
                "drive_final_openness": drive_row.get("final_door_openness", ""),
                "set_final_handle_distance": set_row.get("final_handle_distance", ""),
                "drive_final_handle_distance": drive_row.get("final_handle_distance", ""),
            }
        )

    report_path = output_root / "probes" / object_name / "pair_candidates.json"
    _write_json(report_path, {"object_name": object_name, "rows": pair_rows})
    for row in pair_rows:
        if row["set_pair_success"] and not row["drive_success"]:
            print(
                f"[select] {object_name} traj={row['traj_index']} "
                f"set_{args.set_success_rule}=true drive=false",
                flush=True,
            )
            return int(row["traj_index"])

    raise RuntimeError(
        f"No set-{args.set_success_rule}/drive-failure pair found for {object_name} in {candidate_indices}. "
        f"See {report_path}"
    )


def _record_command(spec: RunSpec, dataset_file: Path, traj_file: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python_executable,
        str(RECORD_SCRIPT),
        "--environment",
        HIGHRES_ENV,
        "--enable_cameras",
        "--mobile_base_relative",
        "--traj_file",
        str(traj_file),
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
    if args.headless:
        cmd.insert(6, "--headless")
    return cmd


def _video_command(dataset_file: Path, output_dir: Path, summary_json: Path, args: argparse.Namespace) -> list[str]:
    return [
        args.python_executable,
        str(VIDEO_SCRIPT),
        "--hdf5",
        str(dataset_file),
        "--output_dir",
        str(output_dir),
        "--camera",
        args.camera,
        "--demo_key",
        "first",
        "--fps",
        str(args.fps),
        "--require_width",
        str(args.width),
        "--require_height",
        str(args.height),
        "--summary_json",
        str(summary_json),
    ]


def _run_one(spec: RunSpec, args: argparse.Namespace, output_root: Path) -> int:
    traj_file = _traj_file(spec.object_name, spec.scene_name)
    if not traj_file.exists():
        raise FileNotFoundError(f"Trajectory file missing: {traj_file}")

    run_dir = output_root / spec.object_name / spec.mode
    dataset_file = run_dir / f"{spec.name}.hdf5"
    status_path = run_dir / "status.json"
    record_log = run_dir / "record.log"
    video_log = run_dir / "video.log"
    hdf5_summary_path = run_dir / "hdf5_summary.json"
    success_csv_path = run_dir / "record_success.csv"
    video_summary_path = run_dir / "video_summary.json"

    if not args.force and dataset_file.exists() and video_summary_path.exists():
        status = {
            **asdict(spec),
            "name": spec.name,
            "status": "skipped_complete",
            "dataset_file": str(dataset_file),
            "video_summary": str(video_summary_path),
        }
        _write_json(status_path, status)
        _append_jsonl(output_root / "manifest.jsonl", status)
        print(f"[skip] {spec.object_name} {spec.mode}", flush=True)
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    record_cmd = _record_command(spec, dataset_file, traj_file, args)
    env = _base_env(args)

    started = time.time()
    initial_status = {
        **asdict(spec),
        "name": spec.name,
        "status": "dry_run" if args.dry_run else "running",
        "dataset_file": str(dataset_file),
        "traj_file": str(traj_file),
        "record_cmd": record_cmd,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(status_path, initial_status)
    _append_jsonl(output_root / "manifest.jsonl", initial_status)
    print(f"[record] {spec.object_name} {spec.mode} traj={spec.traj_indices}", flush=True)

    record_rc = _run_subprocess(record_cmd, cwd=ISAACLAB_ARENA, env=env, log_path=record_log, dry_run=args.dry_run)
    video_rc = 0
    video_cmd: list[str] | None = None
    hdf5_summary: dict[str, Any] = {}

    if record_rc == 0 and not args.dry_run:
        hdf5_summary = _hdf5_summary(dataset_file)
        _write_json(hdf5_summary_path, hdf5_summary)
        _write_success_csv(success_csv_path, hdf5_summary)
        video_cmd = _video_command(dataset_file, run_dir / "videos", video_summary_path, args)
        print(f"[video] {spec.object_name} {spec.mode}", flush=True)
        video_rc = _run_subprocess(video_cmd, cwd=REPO_ROOT, env=env, log_path=video_log, dry_run=False)

    elapsed = time.time() - started
    final_status = {
        **initial_status,
        "status": "completed" if record_rc == 0 and video_rc == 0 else "failed",
        "record_returncode": record_rc,
        "video_returncode": video_rc,
        "video_cmd": video_cmd,
        "elapsed_s": elapsed,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifacts_present": {
            "hdf5": dataset_file.exists(),
            "hdf5_summary": hdf5_summary_path.exists(),
            "record_success_csv": success_csv_path.exists(),
            "video_summary": video_summary_path.exists(),
        },
    }
    _write_json(status_path, final_status)
    _append_jsonl(output_root / "manifest.jsonl", final_status)
    print(
        f"[{final_status['status']}] {spec.object_name} {spec.mode} "
        f"record_rc={record_rc} video_rc={video_rc} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return 0 if final_status["status"] == "completed" else 1


def _objects_from_args(args: argparse.Namespace, selection: dict[str, Any]) -> list[str]:
    if args.objects:
        return [item.strip() for item in args.objects.split(",") if item.strip()]
    return sorted(selection["objects"].keys())


def _indices_for_object(selection: dict[str, Any], object_name: str, episodes_per_mode: int) -> list[int]:
    selected = selection["objects"][object_name]["selected_indices"]
    if episodes_per_mode < 1:
        raise ValueError("--episodes_per_mode must be >= 1")
    if episodes_per_mode > len(selected):
        raise ValueError(f"{object_name} has only {len(selected)} selected indices, requested {episodes_per_mode}")
    return [int(idx) for idx in selected[:episodes_per_mode]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_json", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--objects", default="", help="Comma-separated object names. Defaults to selection JSON order.")
    parser.add_argument("--scene_name", default="", help="Override scene_name from the selection JSON.")
    parser.add_argument("--episodes_per_mode", type=int, default=1)
    parser.add_argument(
        "--search_pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Probe selected random trajectories and choose set-success/drive-failure pairs before recording.",
    )
    parser.add_argument("--probe_only", action="store_true", help="Only run pair-search probes; do not record HDF5/videos.")
    parser.add_argument("--modes", default="set,drive", help="Comma-separated modes from: set,drive.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--camera", default="fix_local_rgb")
    parser.add_argument("--camera_data_types", default="rgb")
    parser.add_argument("--object_joint_names", default="joint_0")
    parser.add_argument(
        "--set_success_rule",
        choices=("door_open", "eval"),
        default="door_open",
        help="Selection rule for set samples. door_open matches report videos; eval requires door_open and final_engaged.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--interpolated", type=int, default=5)
    parser.add_argument("--interpolation_type", default="cubic")
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument("--init_steps", type=int, default=5)
    parser.add_argument("--static_friction", type=float, default=1.0)
    parser.add_argument("--dynamic_friction", type=float, default=1.0)
    parser.add_argument(
        "--python_executable",
        default=os.environ.get(
            "AUTOMOMA_PYTHON",
            str(DEFAULT_AUTOMOMA_PYTHON if DEFAULT_AUTOMOMA_PYTHON.exists() else Path(sys.executable)),
        ),
        help="Python executable used to launch Isaac Sim and video extraction.",
    )
    parser.add_argument("--min_free_gb", type=float, default=20.0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Run at most this many mode/object jobs.")
    args = parser.parse_args()

    selection = _load_selection(args.selection_json)
    scene_name = args.scene_name or selection["scene_name"]
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    selection_copy = output_root / args.selection_json.name
    if args.selection_json.resolve() != selection_copy.resolve():
        shutil.copy2(args.selection_json, selection_copy)

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    bad_modes = sorted(set(modes) - {"set", "drive"})
    if bad_modes:
        raise ValueError(f"Unsupported modes: {bad_modes}")

    specs = []
    for object_name in _objects_from_args(args, selection):
        if object_name not in selection["objects"]:
            raise ValueError(f"{object_name} not in {args.selection_json}")
        candidate_indices = [int(idx) for idx in selection["objects"][object_name]["selected_indices"]]
        if args.episodes_per_mode == 1:
            indices = [
                _select_pair_index(
                    object_name=object_name,
                    scene_name=scene_name,
                    candidate_indices=candidate_indices,
                    output_root=output_root,
                    args=args,
                )
            ]
        else:
            indices = _indices_for_object(selection, object_name, args.episodes_per_mode)
        for mode in modes:
            specs.append(
                RunSpec(
                    object_name=object_name,
                    scene_name=scene_name,
                    mode=mode,
                    traj_indices=indices,
                    width=args.width,
                    height=args.height,
                )
            )

    config = {
        "selection_json": str(args.selection_json),
        "output_root": str(output_root),
        "scene_name": scene_name,
        "objects": [spec.object_name for spec in specs],
        "modes": modes,
        "episodes_per_mode": args.episodes_per_mode,
        "width": args.width,
        "height": args.height,
        "camera": args.camera,
        "camera_data_types": args.camera_data_types,
        "set_success_rule": args.set_success_rule,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dry_run": args.dry_run,
        "probe_only": args.probe_only,
    }
    _write_json(output_root / "record_drive_and_set_config.json", config)

    if args.probe_only:
        print(f"[done] probe_only specs={len(specs)} output_root={output_root}", flush=True)
        return 0

    failures = 0
    launched = 0
    for spec in specs:
        free_gb = shutil.disk_usage(output_root).free / (1024**3)
        if not args.dry_run and free_gb < args.min_free_gb:
            print(
                f"[stop] free space {free_gb:.2f} GiB below --min_free_gb {args.min_free_gb:.2f}; "
                f"not launching {spec.object_name} {spec.mode}",
                flush=True,
            )
            failures += 1
            break
        failures += int(_run_one(spec, args, output_root) != 0)
        launched += 1
        if args.limit and launched >= args.limit:
            break

    print(f"[done] launched={launched} failures={failures} output_root={output_root}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
