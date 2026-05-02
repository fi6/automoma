#!/usr/bin/env python3
"""Run the microwave record ablation grid with isolated outputs.

Default grid:
  interpolated: 2, 3, 4, 5, 10
  decimation: 1, 4
  init_steps: 1, 2, 3, 5, 10, 100
  trajectory_lens: 36, 38, 40

Each experiment writes into:
  data/automoma/ablation_study/i{I}_d{D}_init{S}_len{L}/
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "automoma" / "ablation_study"
DEFAULT_TRYOUT_ROOT = REPO_ROOT / "data" / "automoma" / "ablation_study_tryout"


@dataclass(frozen=True)
class Experiment:
    interpolated: int
    interpolation_type: str
    decimation: int
    init_steps: int
    trajectory_len: int

    @property
    def name(self) -> str:
        return (
            f"i{self.interpolated:02d}_d{self.decimation}"
            f"_init{self.init_steps:03d}_len{self.trajectory_len}"
        )


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _traj_file(object_name: str, scene_name: str, trajectory_len: int) -> Path:
    return (
        REPO_ROOT
        / "data"
        / "trajs"
        / "summit_franka"
        / object_name
        / scene_name
        / "train"
        / f"traj_data_train-{trajectory_len}.pt"
    )


def _expected_joint_csv(exp_dir: Path, dataset_stem: str) -> Path:
    return exp_dir / "debug_curves" / f"{dataset_stem}-joint-tracking.csv"


def _expected_handle_csv(exp_dir: Path, dataset_stem: str) -> Path:
    return exp_dir / "debug_curves" / f"{dataset_stem}-handle-tracking.csv"


def _load_status(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def _experiment_complete(exp_dir: Path, dataset_stem: str) -> bool:
    return (
        (exp_dir / f"{dataset_stem}.hdf5").exists()
        and _expected_joint_csv(exp_dir, dataset_stem).exists()
        and _expected_handle_csv(exp_dir, dataset_stem).exists()
    )


def _run_one(
    exp: Experiment,
    *,
    object_name: str,
    scene_name: str,
    num_episodes: int,
    output_root: Path,
    extra_args: list[str],
    dry_run: bool,
    force: bool,
    include_type_in_path: bool,
) -> int:
    traj_file = _traj_file(object_name, scene_name, exp.trajectory_len)
    if not traj_file.exists():
        raise FileNotFoundError(traj_file)

    exp_dir = output_root / exp.interpolation_type / exp.name if include_type_in_path else output_root / exp.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    dataset_stem = (
        f"summit_franka_open-{object_name}-{scene_name}-{num_episodes}-{exp.name}"
    )
    dataset_file = exp_dir / f"{dataset_stem}.hdf5"
    joint_csv = _expected_joint_csv(exp_dir, dataset_stem)
    handle_csv = _expected_handle_csv(exp_dir, dataset_stem)
    status_path = exp_dir / "status.json"
    run_log_path = exp_dir / "run.log"

    if not force and _experiment_complete(exp_dir, dataset_stem):
        _write_json(
            status_path,
            {
                **asdict(exp),
                "name": exp.name,
                "status": "skipped_complete",
                "dataset_file": str(dataset_file),
                "joint_csv": str(joint_csv),
                "handle_csv": str(handle_csv),
            },
        )
        print(f"[skip] {exp.interpolation_type}/{exp.name}", flush=True)
        return 0

    cmd = [
        "bash",
        "scripts/run_pipeline.sh",
        "record",
        object_name,
        scene_name,
        str(num_episodes),
        "--headless",
        "--dataset_file",
        str(dataset_file),
        "--traj_file",
        str(traj_file),
        "--interpolated",
        str(exp.interpolated),
        "--interpolation_type",
        exp.interpolation_type,
        "--decimation",
        str(exp.decimation),
        "--init_steps",
        str(exp.init_steps),
        "--debug",
        *extra_args,
    ]
    run_info = {
        **asdict(exp),
        "name": exp.name,
        "status": "dry_run" if dry_run else "running",
        "dataset_file": str(dataset_file),
        "traj_file": str(traj_file),
        "joint_csv": str(joint_csv),
        "handle_csv": str(handle_csv),
        "cmd": cmd,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(status_path, run_info)
    _append_jsonl(output_root / "manifest.jsonl", run_info)
    print(f"[run] {exp.interpolation_type}/{exp.name}", flush=True)

    if dry_run:
        print(" ".join(cmd), flush=True)
        return 0

    env = os.environ.copy()
    start_time = time.time()
    with run_log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    elapsed_s = time.time() - start_time
    final_status = {
        **run_info,
        "status": "completed" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "elapsed_s": elapsed_s,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifacts_present": {
            "hdf5": dataset_file.exists(),
            "joint_csv": joint_csv.exists(),
            "handle_csv": handle_csv.exists(),
        },
    }
    _write_json(status_path, final_status)
    _append_jsonl(output_root / "manifest.jsonl", final_status)
    print(
        f"[{final_status['status']}] {exp.interpolation_type}/{exp.name} "
        f"rc={proc.returncode} elapsed={elapsed_s:.1f}s",
        flush=True,
    )
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object_name", default="microwave_7221")
    parser.add_argument("--scene_name", default="scene_0_seed_0")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tryout", type=str, default="", help="Write into ablation_study_tryout/<name> instead.")
    parser.add_argument("--interpolated", default="2,3,4,5,10")
    parser.add_argument("--interpolation_types", default="linear")
    parser.add_argument("--decimation", default="1,4")
    parser.add_argument("--init_steps", default="1,2,3,5,10,100")
    parser.add_argument("--trajectory_lens", default="36,38,40")
    parser.add_argument("--start_at", default="", help="Skip grid entries until this experiment name is reached.")
    parser.add_argument("--limit", type=int, default=0, help="Run at most this many non-skipped experiments.")
    parser.add_argument("--force", action="store_true", help="Rerun even when expected artifacts already exist.")
    parser.add_argument("--min_free_gb", type=float, default=3.0, help="Stop before launching a run if output filesystem has less free space.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    output_root = args.output_root
    if args.tryout:
        output_root = DEFAULT_TRYOUT_ROOT / args.tryout
    output_root.mkdir(parents=True, exist_ok=True)

    interpolation_types = [item.strip() for item in args.interpolation_types.split(",") if item.strip()]
    include_type_in_path = len(interpolation_types) > 1 or interpolation_types != ["linear"]

    experiments = [
        Experiment(interpolated=i, interpolation_type=t, decimation=d, init_steps=s, trajectory_len=l)
        for i, t, d, s, l in itertools.product(
            _parse_int_list(args.interpolated),
            interpolation_types,
            _parse_int_list(args.decimation),
            _parse_int_list(args.init_steps),
            _parse_int_list(args.trajectory_lens),
        )
    ]

    if args.start_at:
        names = [
            f"{exp.interpolation_type}/{exp.name}" if include_type_in_path else exp.name
            for exp in experiments
        ]
        if args.start_at not in names:
            raise ValueError(f"--start_at {args.start_at!r} not in grid")
        experiments = experiments[names.index(args.start_at) :]

    metadata = {
        "object_name": args.object_name,
        "scene_name": args.scene_name,
        "num_episodes": args.num_episodes,
        "output_root": str(output_root),
        "grid_size": len(experiments),
        "interpolation_types": interpolation_types,
        "include_type_in_path": include_type_in_path,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(output_root / "ablation_config.json", metadata)

    failures = 0
    launched = 0
    for exp in experiments:
        exp_dir = output_root / exp.interpolation_type / exp.name if include_type_in_path else output_root / exp.name
        before = _load_status(exp_dir / "status.json")
        free_gb = shutil.disk_usage(output_root).free / (1024**3)
        if not args.dry_run and free_gb < args.min_free_gb:
            print(
                f"[stop] free space {free_gb:.2f} GiB below --min_free_gb {args.min_free_gb:.2f}; "
                f"not launching {exp.name}",
                flush=True,
            )
            failures += 1
            break
        rc = _run_one(
            exp,
            object_name=args.object_name,
            scene_name=args.scene_name,
            num_episodes=args.num_episodes,
            output_root=output_root,
            extra_args=args.extra_args,
            dry_run=args.dry_run,
            force=args.force,
            include_type_in_path=include_type_in_path,
        )
        after = _load_status(exp_dir / "status.json")
        if after.get("status") not in {"skipped_complete", "dry_run"} or before.get("status") != after.get("status"):
            launched += 1
        failures += int(rc != 0)
        if args.limit and launched >= args.limit:
            break

    print(f"[done] launched={launched} failures={failures} output_root={output_root}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
