#!/usr/bin/env python3
"""Batch planner for the public AutoMoMa-500k trajectory release.

The script plans trajectory-only data for every configured object and scene,
then merges only successful trajectories into the canonical per-scene files:

    data/trajs/summit_franka/<object_name>/<scene_name>/train/traj_data_train.pt

Each planning call writes to an isolated round directory first. This avoids the
append-on-existing behavior in PlanningIO from duplicating old trajectories when
we only need to top up a small remaining deficit.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "plan.yaml"
DEFAULT_STAT_SCRIPT = SCRIPT_DIR / "trajectory_statistics.py"

TRAJ_KEYS = (
    "start_robot",
    "start_obj",
    "goal_robot",
    "goal_obj",
    "traj_robot",
    "traj_obj",
    "traj_success",
)


def load_config(path: Path) -> dict[str, Any]:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def object_name(cfg: dict[str, Any], object_id: str) -> str:
    obj_cfg = cfg["objects"][str(object_id)]
    return f"{obj_cfg['asset_type'].lower()}_{object_id}"


def discover_scenes(scene_dir: Path) -> list[str]:
    if not scene_dir.exists():
        return []
    return sorted(p.name for p in scene_dir.iterdir() if p.is_dir())


def canonical_traj_file(
    output_root: Path,
    robot_name: str,
    object_name_value: str,
    scene_name: str,
    split: str,
) -> Path:
    return output_root / robot_name / object_name_value / scene_name / split / f"traj_data_{split}.pt"


def count_successes(path: Path) -> int:
    if not path.exists():
        return 0
    data = torch.load(path, weights_only=False)
    if "traj_success" in data:
        success = data["traj_success"]
        if isinstance(success, torch.Tensor):
            return int(success.bool().sum().item())
    if "traj_robot" in data and isinstance(data["traj_robot"], torch.Tensor):
        return int(data["traj_robot"].shape[0])
    return 0


def load_payload(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, weights_only=False)
    missing = [key for key in TRAJ_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{path} is missing required trajectory keys: {missing}")
    for key in TRAJ_KEYS:
        if not isinstance(payload[key], torch.Tensor):
            raise TypeError(f"{path}:{key} must be a torch.Tensor, got {type(payload[key]).__name__}")
    return payload


def shape_report(payload: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {key: list(payload[key].shape) for key in TRAJ_KEYS}


def filter_successful(payload: dict[str, torch.Tensor], limit: int | None = None) -> dict[str, torch.Tensor]:
    success = payload["traj_success"].bool()
    idx = torch.nonzero(success, as_tuple=False).flatten()
    if limit is not None:
        idx = idx[: max(0, int(limit))]
    filtered = {key: payload[key][idx].cpu() for key in TRAJ_KEYS}
    filtered["traj_success"] = torch.ones(idx.shape[0], dtype=payload["traj_success"].dtype)
    return filtered


def validate_compatible(existing: dict[str, torch.Tensor], new: dict[str, torch.Tensor], label: str) -> None:
    for key in TRAJ_KEYS:
        left = existing[key]
        right = new[key]
        if left.ndim != right.ndim:
            raise ValueError(f"{label}: key {key} rank mismatch {left.ndim} != {right.ndim}")
        if left.shape[1:] != right.shape[1:]:
            raise ValueError(f"{label}: key {key} shape mismatch {tuple(left.shape)} != {tuple(right.shape)}")
        if left.dtype != right.dtype:
            raise ValueError(f"{label}: key {key} dtype mismatch {left.dtype} != {right.dtype}")


def merge_successes(canonical: Path, round_file: Path, max_add: int) -> dict[str, Any]:
    round_payload = filter_successful(load_payload(round_file), max_add)
    added = int(round_payload["traj_success"].shape[0])

    if canonical.exists():
        base_payload = filter_successful(load_payload(canonical))
        validate_compatible(base_payload, round_payload, f"merge {round_file} into {canonical}")
        merged = {key: torch.cat([base_payload[key], round_payload[key]], dim=0).cpu() for key in TRAJ_KEYS}
    else:
        merged = round_payload

    canonical.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=canonical.parent,
        prefix=f".{canonical.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(merged, tmp_path)
        tmp_path.replace(canonical)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {
        "canonical": str(canonical),
        "round_file": str(round_file),
        "added_successful": added,
        "canonical_successful": int(merged["traj_success"].shape[0]),
        "shapes": shape_report(merged),
    }


def run_plan_command(
    *,
    config: Path,
    scene_dir: Path,
    object_id: str,
    scene_name: str,
    split: str,
    round_output_dir: Path,
    max_successful: int,
    extra_overrides: list[str],
    dry_run: bool,
) -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plan.py"),
        "--config",
        str(config),
        f"scene_dir={scene_dir}",
        f"scene_name={scene_name}",
        f"object_id={object_id}",
        f"mode={split}",
        f"output_dir={round_output_dir}",
        f"planner.output.max_successful_trajectories={max_successful}",
        "resume=false",
        *extra_overrides,
    ]
    print(" ".join(cmd), flush=True)
    if dry_run:
        return 0
    round_output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_per_scene_target(value: str, object_target: int, scene_count: int) -> int:
    if value == "auto":
        return int(math.ceil(object_target / scene_count))
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("--per-scene-target must be 'auto' or a positive integer")
    return parsed


def run_statistics(
    *,
    output_dir: Path,
    root: Path,
    config: Path,
    scene_dir: Path,
    split: str,
    target_per_object: int,
    objects: list[str],
    scenes: list[str],
) -> None:
    if not DEFAULT_STAT_SCRIPT.exists():
        print(f"Warning: statistics script not found: {DEFAULT_STAT_SCRIPT}", file=sys.stderr)
        return
    cmd = [
        sys.executable,
        str(DEFAULT_STAT_SCRIPT),
        "--config",
        str(config),
        "--root",
        str(root),
        "--output-dir",
        str(output_dir),
        "--scene-dir",
        str(scene_dir),
        "--split",
        split,
        "--target-per-object",
        str(target_per_object),
        "--objects",
        *objects,
        "--scenes",
        *scenes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="automoma_500k_selftest_") as tmp:
        root = Path(tmp)
        canonical = root / "data" / "trajs" / "summit_franka" / "microwave_7221" / "scene_x" / "train" / "traj_data_train.pt"
        round_file = root / "round" / "traj_data_train.pt"
        payload = {
            "start_robot": torch.zeros(5, 12),
            "start_obj": torch.zeros(5, 1),
            "goal_robot": torch.ones(5, 12),
            "goal_obj": torch.ones(5, 1),
            "traj_robot": torch.zeros(5, 36, 12),
            "traj_obj": torch.zeros(5, 36, 1),
            "traj_success": torch.tensor([True, False, True, True, False]),
        }
        round_file.parent.mkdir(parents=True)
        torch.save(payload, round_file)
        report = merge_successes(canonical, round_file, max_add=2)
        assert report["added_successful"] == 2, report
        assert count_successes(canonical) == 2
        report = merge_successes(canonical, round_file, max_add=1)
        assert report["canonical_successful"] == 3, report
    print("Self-test passed: success filtering, capped merge, and counting are working.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan trajectory-only data for AutoMoMa-500k.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data" / "trajs")
    parser.add_argument("--statistics-dir", type=Path, default=REPO_ROOT / "outputs" / "statistics" / "automoma-500k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-per-object", type=int, default=100_000)
    parser.add_argument("--per-scene-target", default="auto")
    parser.add_argument("--max-rounds-per-object", type=int, default=50)
    parser.add_argument("--objects", nargs="+", default=None, help="Object ids. Defaults to all objects in plan.yaml.")
    parser.add_argument("--scenes", nargs="+", default=None, help="Scene names. Defaults to all directories under scene-dir.")
    parser.add_argument("--round-root", type=Path, default=REPO_ROOT / "data" / "trajs" / "_automoma_500k_rounds")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cleanup-rounds", action="store_true", help="Delete each isolated round directory after a successful merge.")
    parser.add_argument("--skip-statistics", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "plan_overrides",
        nargs=argparse.REMAINDER,
        help="Extra scripts/plan.py OmegaConf overrides. Prefix with -- before the first override.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    cfg = load_config(args.config)
    robot_name = cfg.get("robot_name", "summit_franka")
    scene_dir = args.scene_dir or (REPO_ROOT / cfg.get("scene_dir", "assets/scene/infinigen/kitchen_1130"))
    if not scene_dir.is_absolute():
        scene_dir = REPO_ROOT / scene_dir

    object_ids = [str(obj) for obj in (args.objects or cfg["objects"].keys())]
    scenes = list(args.scenes or discover_scenes(scene_dir))
    if not scenes:
        raise SystemExit(
            f"No scenes found. Checked {scene_dir}. Pass --scenes explicitly after scene assets are available."
        )

    per_scene_target = parse_per_scene_target(args.per_scene_target, args.target_per_object, len(scenes))
    extra_overrides = list(args.plan_overrides)
    if extra_overrides and extra_overrides[0] == "--":
        extra_overrides = extra_overrides[1:]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "config": str(args.config),
        "scene_dir": str(scene_dir),
        "robot_name": robot_name,
        "split": args.split,
        "target_per_object": args.target_per_object,
        "per_scene_target": per_scene_target,
        "objects": object_ids,
        "scenes": scenes,
        "dry_run": args.dry_run,
        "rounds": [],
    }

    manifest_path = args.statistics_dir / f"plan_manifest_{run_id}.json"
    write_json(manifest_path, manifest)
    print(f"Plan manifest: {manifest_path}")
    print(f"Scenes: {len(scenes)} | per-scene target: {per_scene_target} | object target: {args.target_per_object}")

    for object_id in object_ids:
        obj_name = object_name(cfg, object_id)
        print(f"\n=== Object {object_id} ({obj_name}) ===", flush=True)

        object_satisfied = False
        for round_idx in range(1, args.max_rounds_per_object + 1):
            scene_counts = {
                scene: count_successes(
                    canonical_traj_file(args.output_root, robot_name, obj_name, scene, args.split)
                )
                for scene in scenes
            }
            object_total = sum(scene_counts.values())
            print(f"Round {round_idx}: current successful={object_total}", flush=True)
            if object_total >= args.target_per_object and all(canonical_traj_file(args.output_root, robot_name, obj_name, scene, args.split).exists() for scene in scenes):
                print(f"Object target satisfied: {object_total} >= {args.target_per_object}")
                object_satisfied = True
                break

            progressed = False
            for scene in scenes:
                canonical = canonical_traj_file(args.output_root, robot_name, obj_name, scene, args.split)
                current_scene = count_successes(canonical)
                object_total = sum(
                    count_successes(canonical_traj_file(args.output_root, robot_name, obj_name, s, args.split))
                    for s in scenes
                )
                object_remaining = max(0, args.target_per_object - object_total)
                scene_remaining = max(0, per_scene_target - current_scene)

                if canonical.exists() and scene_remaining <= 0:
                    continue
                if object_remaining <= 0 and canonical.exists():
                    continue

                max_successful = scene_remaining
                if object_remaining > 0:
                    max_successful = min(max_successful or object_remaining, object_remaining)
                if max_successful <= 0:
                    max_successful = 1

                round_dir = args.round_root / run_id / obj_name / scene / args.split / f"round_{round_idx:03d}_{int(time.time())}"
                round_file = canonical_traj_file(round_dir, robot_name, obj_name, scene, args.split)
                print(
                    f"Planning {obj_name}/{scene}: current_scene={current_scene}, "
                    f"object_total={object_total}, request={max_successful}",
                    flush=True,
                )
                rc = run_plan_command(
                    config=args.config,
                    scene_dir=scene_dir,
                    object_id=object_id,
                    scene_name=scene,
                    split=args.split,
                    round_output_dir=round_dir,
                    max_successful=max_successful,
                    extra_overrides=extra_overrides,
                    dry_run=args.dry_run,
                )
                if rc != 0:
                    raise SystemExit(f"Planning failed for {obj_name}/{scene} with exit code {rc}")

                round_report: dict[str, Any] = {
                    "object_id": object_id,
                    "object_name": obj_name,
                    "scene": scene,
                    "round": round_idx,
                    "requested_successful": max_successful,
                    "round_dir": str(round_dir),
                    "canonical": str(canonical),
                }
                if not args.dry_run:
                    if not round_file.exists():
                        raise FileNotFoundError(f"Expected planning output not found: {round_file}")
                    merge_report = merge_successes(canonical, round_file, max_successful)
                    round_report.update(merge_report)
                    if args.cleanup_rounds:
                        shutil.rmtree(round_dir, ignore_errors=True)
                manifest["rounds"].append(round_report)
                write_json(manifest_path, manifest)
                progressed = True

                if sum(
                    count_successes(canonical_traj_file(args.output_root, robot_name, obj_name, s, args.split))
                    for s in scenes
                ) >= args.target_per_object and all(
                    canonical_traj_file(args.output_root, robot_name, obj_name, s, args.split).exists()
                    for s in scenes
                ):
                    object_satisfied = True
                    break

            if args.dry_run:
                print(f"Dry-run complete for {obj_name}; no trajectory files were written.")
                break
            if object_satisfied:
                print(f"Object target satisfied after round {round_idx}.")
                break
            if not progressed:
                raise SystemExit(
                    f"No progress possible for {obj_name}; check scene availability and planner output."
                )
        if not object_satisfied and not args.dry_run:
            raise SystemExit(
                f"{obj_name} did not reach {args.target_per_object} successful trajectories "
                f"within {args.max_rounds_per_object} rounds."
            )

    if not args.skip_statistics and not args.dry_run:
        run_statistics(
            output_dir=args.statistics_dir,
            root=args.output_root,
            config=args.config,
            scene_dir=scene_dir,
            split=args.split,
            target_per_object=args.target_per_object,
            objects=object_ids,
            scenes=scenes,
        )

    write_json(manifest_path, manifest)
    print(f"\nDone. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
