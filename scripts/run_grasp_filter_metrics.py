#!/usr/bin/env python3
"""Run metrics-only AutoMoMa replay for grasp filtering.

This script samples planned trajectories from data/trajs/summit_franka, replays
them through the dedicated replay pipeline with --metrics, and keeps only
lightweight CSV/JSON/PNG outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJ_ROOT = REPO_ROOT / "data" / "trajs" / "summit_franka"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "debug" / "grasp_filter"
DEFAULT_PLAN_CONFIG = REPO_ROOT / "configs" / "plan.yaml"


def stable_seed(*parts: str, seed: int) -> int:
    digest = hashlib.sha256(("|".join(parts) + f"|{seed}").encode()).hexdigest()
    return int(digest[:16], 16)


def load_count(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "traj_success" in payload:
        return int(payload["traj_success"].shape[0])
    return int(payload["traj_robot"].shape[0])


def load_grasp_ids_from_config(config_path: Path) -> dict[str, list[int]]:
    try:
        from omegaconf import OmegaConf
    except Exception:
        return {}
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    out: dict[str, list[int]] = {}
    for object_id, obj_cfg in cfg.get("objects", {}).items():
        name = f"{obj_cfg['asset_type'].lower()}_{object_id}"
        out[name] = [int(x) for x in obj_cfg.get("grasp_ids", list(range(20)))]
    return out


def iter_round_grasp_files(object_name: str, scene_name: str) -> list[Path]:
    roots = sorted(DEFAULT_TRAJ_ROOT.parent.glob("_automoma_500k_rounds*"))
    files: list[Path] = []
    suffix = (
        f"summit_franka/{object_name}/{scene_name}/train"
    )
    for root in roots:
        files.extend(root.glob(f"**/{object_name}/{scene_name}/train/round_*/{suffix}/grasp_*/traj_data.pt"))
    return sorted(files, key=lambda p: (str(p.parents[6]) if len(p.parents) > 6 else str(p), str(p)))


def infer_grasp_map(object_name: str, scene_name: str, total: int, config_grasps: list[int]) -> tuple[list[int | None], str]:
    inferred: list[int | None] = []
    for path in iter_round_grasp_files(object_name, scene_name):
        try:
            grasp_id = int(path.parent.name.split("_", 1)[1])
            payload = torch.load(path, map_location="cpu", weights_only=False)
            success = payload.get("success")
            if success is None:
                n_success = int(payload["trajectories"].shape[0])
            else:
                n_success = int(success.bool().sum().item())
        except Exception:
            continue
        inferred.extend([grasp_id] * n_success)
        if len(inferred) >= total:
            return inferred[:total], "round_grasp_success_concat"
    if len(inferred) == total:
        return inferred, "round_grasp_success_concat"
    if config_grasps:
        fallback = [config_grasps[i % len(config_grasps)] for i in range(total)]
        return fallback, "fallback_modulo_config_grasps"
    return [None] * total, "unknown"


def discover_work(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected_scene_names: dict[str, set[str]] = defaultdict(set)
    if args.scene_list_file is not None:
        with args.scene_list_file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"object_name", "scene_name"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{args.scene_list_file} must contain columns: object_name,scene_name")
            for row in reader:
                selected_scene_names[row["object_name"]].add(row["scene_name"])

    by_object: dict[str, list[tuple[str, Path, int]]] = defaultdict(list)
    for path in sorted(args.traj_root.glob("*/*/train/traj_data_train.pt")):
        object_name = path.parents[2].name
        scene_name = path.parents[1].name
        try:
            count = load_count(path)
        except Exception as exc:
            print(f"[discover] skip unreadable {path}: {exc}", flush=True)
            continue
        if count >= args.min_scene_episodes:
            by_object[object_name].append((scene_name, path, count))

    config_grasps = load_grasp_ids_from_config(args.config)
    work: list[dict[str, Any]] = []
    object_items = sorted(by_object.items())
    if args.max_objects is not None:
        object_items = object_items[: args.max_objects]

    for object_name, scenes in object_items:
        scenes = sorted(scenes)
        if selected_scene_names:
            requested = selected_scene_names.get(object_name, set())
            selected = [scene for scene in scenes if scene[0] in requested]
            missing = sorted(requested - {scene[0] for scene in selected})
            if missing:
                raise ValueError(f"missing requested scenes for {object_name}: {missing}")
        else:
            local_rng = random.Random(stable_seed(object_name, seed=args.seed))
            selected = scenes if len(scenes) <= args.scenes_per_object else local_rng.sample(scenes, args.scenes_per_object)
        for scene_name, traj_file, count in sorted(selected):
            sample_n = min(count, args.max_episodes_per_scene)
            scene_rng = random.Random(stable_seed(object_name, scene_name, seed=args.seed))
            indices = sorted(scene_rng.sample(range(count), sample_n)) if count > sample_n else list(range(count))
            grasp_map, mapping_source = infer_grasp_map(
                object_name,
                scene_name,
                count,
                config_grasps.get(object_name, []),
            )
            work.append({
                "object_name": object_name,
                "scene_name": scene_name,
                "traj_file": str(traj_file),
                "available_episodes": count,
                "indices": indices,
                "mapping_source": mapping_source,
                "grasp_ids": [grasp_map[i] if i < len(grasp_map) else None for i in indices],
            })
    return work


def write_manifest(work: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "object_name",
                "scene_name",
                "traj_file",
                "available_episodes",
                "traj_index",
                "grasp_id",
                "mapping_source",
            ],
        )
        writer.writeheader()
        for item in work:
            for traj_index, grasp_id in zip(item["indices"], item["grasp_ids"]):
                writer.writerow({
                    "object_name": item["object_name"],
                    "scene_name": item["scene_name"],
                    "traj_file": item["traj_file"],
                    "available_episodes": item["available_episodes"],
                    "traj_index": traj_index,
                    "grasp_id": "" if grasp_id is None else grasp_id,
                    "mapping_source": item["mapping_source"],
                })
    return path


def write_indices(work: list[dict[str, Any]], output_dir: Path) -> None:
    index_dir = output_dir / "indices"
    index_dir.mkdir(parents=True, exist_ok=True)
    for item in work:
        path = index_dir / f"{item['object_name']}__{item['scene_name']}.txt"
        path.write_text("\n".join(str(x) for x in item["indices"]) + "\n", encoding="utf-8")
        item["indices_file"] = str(path)
        item["metrics_file"] = str(output_dir / "per_scene" / f"{item['object_name']}__{item['scene_name']}.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def union_fieldnames(rows: list[dict[str, Any]], preferred: list[str] | None = None) -> list[str]:
    fields: list[str] = []
    for key in preferred or []:
        if key not in fields:
            fields.append(key)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _float_or_nan(value: Any) -> float:
    try:
        if value in (None, ""):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _max_or_nan(values: list[float]) -> float:
    finite = _finite(values)
    return max(finite) if finite else float("nan")


def _mean_or_nan(values: list[float]) -> float:
    finite = _finite(values)
    return sum(finite) / len(finite) if finite else float("nan")


def _format_float(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.9g}"


def read_trace_summary(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    if not rows:
        return []

    base_name = path.name.split("-joint-tracking", 1)[0]
    object_name = base_name
    scene_name = ""
    if "__" in base_name:
        object_name, scene_name = base_name.split("__", 1)

    out = []
    for row in rows:
        merged = dict(row)
        merged["object_name"] = object_name
        merged["scene_name"] = scene_name
        merged["trace_summary_file"] = str(path)
        out.append(merged)
    return out


def summarize_trace_artifacts(output_dir: Path) -> None:
    trace_rows: list[dict[str, Any]] = []
    for path in sorted(
        (output_dir / "per_scene" / "debug_curves").glob("*-joint-tracking-episode-summary*.csv")
    ):
        trace_rows.extend(read_trace_summary(path))
    if not trace_rows:
        return

    fieldnames = union_fieldnames(trace_rows, preferred=["object_name", "scene_name"])
    write_csv(output_dir / "trajectory_trace_summary.csv", trace_rows, fieldnames)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_obj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        grouped[(row["object_name"], row["scene_name"])].append(row)
        grouped_obj[row["object_name"]].append(row)

    def aggregate(rows: list[dict[str, Any]], extra: dict[str, Any]) -> dict[str, Any]:
        first20_nongripper_gap = [_float_or_nan(row.get("first20_max_nongripper_abs_gap")) for row in rows]
        first20_base_span = [_float_or_nan(row.get("first20_actual_base_span")) for row in rows]
        first20_arm_span = [_float_or_nan(row.get("first20_actual_arm_span")) for row in rows]
        first20_target_nongripper_span = [
            _float_or_nan(row.get("first20_target_nongripper_span")) for row in rows
        ]
        first20_actual_nongripper_span = [
            _float_or_nan(row.get("first20_actual_nongripper_span")) for row in rows
        ]
        eef_pos_gap = [_float_or_nan(row.get("max_eef_pos_gap_m")) for row in rows]
        eef_rot_gap = [_float_or_nan(row.get("max_eef_rot_gap_rad")) for row in rows]
        return {
            **extra,
            "traces": len(rows),
            "first20_max_nongripper_gap": _format_float(_max_or_nan(first20_nongripper_gap)),
            "first20_mean_nongripper_gap": _format_float(_mean_or_nan(first20_nongripper_gap)),
            "first20_max_actual_base_span": _format_float(_max_or_nan(first20_base_span)),
            "first20_max_actual_arm_span": _format_float(_max_or_nan(first20_arm_span)),
            "first20_max_target_nongripper_span": _format_float(_max_or_nan(first20_target_nongripper_span)),
            "first20_max_actual_nongripper_span": _format_float(_max_or_nan(first20_actual_nongripper_span)),
            "max_eef_pos_gap_m": _format_float(_max_or_nan(eef_pos_gap)),
            "mean_eef_pos_gap_m": _format_float(_mean_or_nan(eef_pos_gap)),
            "max_eef_rot_gap_rad": _format_float(_max_or_nan(eef_rot_gap)),
            "mean_eef_rot_gap_rad": _format_float(_mean_or_nan(eef_rot_gap)),
        }

    scene_rows = [
        aggregate(rows, {"object_name": obj, "scene_name": scene})
        for (obj, scene), rows in sorted(grouped.items())
    ]
    object_rows = [
        aggregate(rows, {"object_name": obj})
        for obj, rows in sorted(grouped_obj.items())
    ]
    trace_fields = list(scene_rows[0].keys())
    write_csv(output_dir / "trajectory_trace_by_scene.csv", scene_rows, trace_fields)
    write_csv(output_dir / "trajectory_trace_by_object.csv", object_rows, list(object_rows[0].keys()))


def summarize(output_dir: Path, threshold: float) -> None:
    summarize_trace_artifacts(output_dir)

    manifest_rows = read_csv(output_dir / "manifest.csv")
    manifest = {
        (row["object_name"], row["scene_name"], int(row["traj_index"])): row
        for row in manifest_rows
    }
    raw_rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / "per_scene").glob("*.csv")):
        for row in read_csv(path):
            key = (row["object_name"], row["scene_name"], int(row["traj_index"]))
            m = manifest.get(key, {})
            merged = dict(row)
            merged["grasp_id"] = m.get("grasp_id", "")
            merged["mapping_source"] = m.get("mapping_source", "")
            raw_rows.append(merged)
    if not raw_rows:
        return
    fields = union_fieldnames(raw_rows)
    write_csv(output_dir / "results_raw_joined.csv", raw_rows, fields)

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    grouped_obj: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        obj = row["object_name"]
        gid = row.get("grasp_id", "")
        success = str(row.get("success", "")).lower() == "true"
        scene = row["scene_name"]
        g = grouped.setdefault((obj, gid), {"object_name": obj, "grasp_id": gid, "trials": 0, "successes": 0, "scenes": set()})
        g["trials"] += 1
        g["successes"] += int(success)
        g["scenes"].add(scene)
        go = grouped_obj.setdefault(obj, {"object_name": obj, "trials": 0, "successes": 0, "scenes": set()})
        go["trials"] += 1
        go["successes"] += int(success)
        go["scenes"].add(scene)

    grasp_rows = []
    filters: dict[str, dict[str, list[int]]] = {}
    for (obj, gid), g in sorted(grouped.items()):
        rate = g["successes"] / g["trials"] if g["trials"] else 0.0
        status = "good" if rate >= threshold else "bad"
        grasp_rows.append({
            "object_name": obj,
            "grasp_id": gid,
            "trials": g["trials"],
            "successes": g["successes"],
            "success_rate": f"{rate:.6f}",
            "scene_count": len(g["scenes"]),
            "status": status,
        })
        if gid != "":
            filters.setdefault(obj, {"good": [], "bad": []})[status].append(int(gid))
    write_csv(
        output_dir / "summary_by_grasp.csv",
        grasp_rows,
        ["object_name", "grasp_id", "trials", "successes", "success_rate", "scene_count", "status"],
    )

    object_rows = []
    for obj, g in sorted(grouped_obj.items()):
        rate = g["successes"] / g["trials"] if g["trials"] else 0.0
        object_rows.append({
            "object_name": obj,
            "trials": g["trials"],
            "successes": g["successes"],
            "success_rate": f"{rate:.6f}",
            "scene_count": len(g["scenes"]),
        })
    write_csv(output_dir / "summary_by_object.csv", object_rows, ["object_name", "trials", "successes", "success_rate", "scene_count"])
    (output_dir / "grasp_filter.json").write_text(json.dumps(filters, indent=2, sort_keys=True), encoding="utf-8")
    plot_summaries(output_dir, grasp_rows)


def plot_summaries(output_dir: Path, grasp_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[summarize] matplotlib unavailable, skip plots: {exc}", flush=True)
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    by_obj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in grasp_rows:
        by_obj[row["object_name"]].append(row)
    for obj, rows in by_obj.items():
        rows = sorted(rows, key=lambda r: int(r["grasp_id"]) if str(r["grasp_id"]).isdigit() else 10**9)
        labels = [str(r["grasp_id"]) for r in rows]
        rates = [float(r["success_rate"]) for r in rows]
        colors = ["#2a9d8f" if r["status"] == "good" else "#d1495b" for r in rows]
        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.45), 4.5), constrained_layout=True)
        ax.bar(labels, rates, color=colors)
        ax.axhline(0.7, color="#333333", linewidth=1, linestyle="--")
        ax.set_ylim(0, 1)
        ax.set_xlabel("grasp_id")
        ax.set_ylabel("record success rate")
        ax.set_title(obj)
        fig.savefig(plot_dir / f"{obj}_grasp_success.png", dpi=180)
        plt.close(fig)


def run_work(work: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "ACCEPT_EULA": "Y",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "AUTOMOMA_ROBOT_OBJECT_STATIC_FRICTION": str(args.static_friction),
        "AUTOMOMA_ROBOT_OBJECT_DYNAMIC_FRICTION": str(args.dynamic_friction),
    })
    progress_path = output_dir / "progress.jsonl"
    for idx, item in enumerate(work, start=1):
        metrics_path = Path(item["metrics_file"])
        if metrics_path.exists() and len(read_csv(metrics_path)) >= len(item["indices"]):
            print(f"[{idx}/{len(work)}] skip completed {item['object_name']} {item['scene_name']}", flush=True)
            continue
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "bash",
            "scripts/run_pipeline.sh",
            "replay",
            item["object_name"],
            item["scene_name"],
            str(len(item["indices"])),
            "--headless",
            "--disable_cameras",
            "--metrics",
            "--metrics_file",
            str(metrics_path),
            "--episode_indices_file",
            item["indices_file"],
            "--traj_file",
            item["traj_file"],
            "--interpolated",
            str(args.interpolated),
            "--interpolation_type",
            args.interpolation_type,
            "--decimation",
            str(args.decimation),
            "--init_steps",
            str(args.init_steps),
        ]
        if args.no_step_trace:
            cmd.append("--no_step_trace")
        print(f"[{idx}/{len(work)}] running {item['object_name']} {item['scene_name']} n={len(item['indices'])}", flush=True)
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "start", "index": idx, "total": len(work), **{k: item[k] for k in ("object_name", "scene_name", "metrics_file")}}) + "\n")
        rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode
        rows_written = len(read_csv(metrics_path))
        if rows_written < len(item["indices"]):
            print(
                f"[{idx}/{len(work)}] incomplete metrics for {item['object_name']} {item['scene_name']}: "
                f"{rows_written}/{len(item['indices'])} rows",
                flush=True,
            )
            rc = rc or 1
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "event": "done",
                "index": idx,
                "returncode": rc,
                "metrics_rows": rows_written,
                "expected_rows": len(item["indices"]),
                "object_name": item["object_name"],
                "scene_name": item["scene_name"],
            }) + "\n")
        summarize(output_dir, args.success_threshold)
        if rc != 0 and not args.keep_going:
            raise SystemExit(rc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-root", type=Path, default=DEFAULT_TRAJ_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_PLAN_CONFIG)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--scenes-per-object", type=int, default=5)
    parser.add_argument("--scene-list-file", type=Path, default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--min-scene-episodes", type=int, default=500)
    parser.add_argument("--max-episodes-per-scene", type=int, default=2000)
    parser.add_argument("--success-threshold", type=float, default=0.7)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--interpolated", type=int, default=5)
    parser.add_argument("--interpolation-type", default="cubic")
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument("--init-steps", type=int, default=5)
    parser.add_argument("--static-friction", type=float, default=1.0)
    parser.add_argument("--dynamic-friction", type=float, default=1.0)
    parser.add_argument("--no-step-trace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    if args.max_objects is not None and args.max_objects < 1:
        parser.error("--max-objects must be >= 1 when set")
    if args.scenes_per_object < 1:
        parser.error("--scenes-per-object must be >= 1")
    if args.max_episodes_per_scene < 1:
        parser.error("--max-episodes-per-scene must be >= 1")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir

    work = discover_work(args)
    write_indices(work, output_dir)
    manifest_path = write_manifest(work, output_dir)
    config = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "num_scene_jobs": len(work),
        "total_sampled_episodes": sum(len(item["indices"]) for item in work),
        "args": vars(args) | {
            "output_dir": str(output_dir),
            "traj_root": str(args.traj_root),
            "config": str(args.config),
            "scene_list_file": None if args.scene_list_file is None else str(args.scene_list_file),
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        return 0
    run_work(work, output_dir, args)
    summarize(output_dir, args.success_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
