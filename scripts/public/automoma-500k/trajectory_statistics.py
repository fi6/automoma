#!/usr/bin/env python3
"""Create local statistics and charts for AutoMoMa trajectory data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "plan.yaml"
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
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def object_name(cfg: dict[str, Any], object_id: str) -> str:
    obj_cfg = cfg["objects"][str(object_id)]
    return f"{obj_cfg['asset_type'].lower()}_{object_id}"


def discover_scenes(scene_dir: Path) -> list[str]:
    if not scene_dir.exists():
        return []
    return sorted(p.name for p in scene_dir.iterdir() if p.is_dir())


def discover_data_scenes(root: Path, robot_name: str, object_names: list[str], split: str) -> list[str]:
    scenes: set[str] = set()
    for obj_name in object_names:
        object_root = root / robot_name / obj_name
        if not object_root.exists():
            continue
        for scene_dir in object_root.iterdir():
            if not scene_dir.is_dir():
                continue
            if (scene_dir / split / f"traj_data_{split}.pt").exists():
                scenes.add(scene_dir.name)
    return sorted(scenes)


def trajectory_file(root: Path, robot_name: str, obj_name: str, scene: str, split: str) -> Path:
    return root / robot_name / obj_name / scene / split / f"traj_data_{split}.pt"


def summarize_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "total": 0,
            "successful": 0,
            "success_rate": 0.0,
            "shapes": {},
        }
    data = torch.load(path, weights_only=False)
    shapes = {
        key: list(value.shape)
        for key, value in data.items()
        if key in TRAJ_KEYS and isinstance(value, torch.Tensor)
    }
    total = 0
    successful = 0
    if "traj_success" in data and isinstance(data["traj_success"], torch.Tensor):
        total = int(data["traj_success"].shape[0])
        successful = int(data["traj_success"].bool().sum().item())
    elif "traj_robot" in data and isinstance(data["traj_robot"], torch.Tensor):
        total = int(data["traj_robot"].shape[0])
        successful = total
    return {
        "exists": True,
        "total": total,
        "successful": successful,
        "success_rate": float(successful / total) if total else 0.0,
        "shapes": shapes,
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    robot_name = cfg.get("robot_name", "summit_franka")
    scene_dir = args.scene_dir or (REPO_ROOT / cfg.get("scene_dir", "assets/scene/infinigen/kitchen_1130"))
    if not scene_dir.is_absolute():
        scene_dir = REPO_ROOT / scene_dir

    object_ids = [str(obj) for obj in (args.objects or cfg["objects"].keys())]
    object_names = [object_name(cfg, object_id) for object_id in object_ids]
    if args.scenes:
        scenes = list(args.scenes)
    else:
        scenes = sorted(set(discover_scenes(scene_dir)) | set(discover_data_scenes(args.root, robot_name, object_names, args.split)))

    rows = []
    object_totals = defaultdict(int)
    object_existing_scenes = defaultdict(int)
    scene_totals = defaultdict(int)
    grand_total = 0

    for object_id in object_ids:
        obj_name = object_name(cfg, object_id)
        for scene in scenes:
            path = trajectory_file(args.root, robot_name, obj_name, scene, args.split)
            stats = summarize_file(path)
            row = {
                "object_id": object_id,
                "object_name": obj_name,
                "scene": scene,
                "split": args.split,
                "path": str(path),
                **stats,
            }
            rows.append(row)
            object_totals[obj_name] += stats["successful"]
            scene_totals[scene] += stats["successful"]
            grand_total += stats["successful"]
            if stats["exists"]:
                object_existing_scenes[obj_name] += 1

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(args.root),
        "robot_name": robot_name,
        "split": args.split,
        "target_per_object": args.target_per_object,
        "objects": object_ids,
        "scenes": scenes,
        "rows": rows,
        "object_totals": dict(object_totals),
        "object_existing_scenes": dict(object_existing_scenes),
        "scene_totals": dict(scene_totals),
        "grand_total": grand_total,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "object_id",
        "object_name",
        "scene",
        "split",
        "exists",
        "total",
        "successful",
        "success_rate",
        "path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f5ef",
            "axes.facecolor": "#f7f5ef",
            "axes.edgecolor": "#2f3437",
            "axes.labelcolor": "#2f3437",
            "xtick.color": "#2f3437",
            "ytick.color": "#2f3437",
            "text.color": "#1f2428",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
        }
    )


def plot_object_totals(summary: dict[str, Any], output_dir: Path) -> Path:
    setup_style()
    totals = summary["object_totals"]
    objects = sorted(totals, key=totals.get)
    values = [totals[obj] for obj in objects]
    target = int(summary["target_per_object"])

    fig_height = max(4.5, 0.72 * len(objects) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)
    colors = ["#3b7a78" if v >= target else "#c96b4b" for v in values]
    ax.barh(objects, values, color=colors, height=0.54)
    ax.axvline(target, color="#1f2428", linewidth=1.4, linestyle="--", label=f"target {target:,}")
    x_max = max([target, *values, 1])
    label_offset = x_max * 0.012
    ax.set_xlim(0, x_max * 1.14)
    ax.set_title("AutoMoMa-500k Successful Trajectories by Object")
    ax.set_xlabel("successful trajectories")
    ax.grid(axis="x", color="#d8d2c5", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    for y, value in enumerate(values):
        ax.text(value + label_offset, y, f"{value:,}", va="center", fontsize=10)
    path = output_dir / "object_totals.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scene_heatmap(summary: dict[str, Any], output_dir: Path) -> Path:
    setup_style()
    objects = [row["object_name"] for row in summary["rows"]]
    objects = sorted(set(objects))
    scenes = list(summary["scenes"])
    values = [[0 for _ in scenes] for _ in objects]
    object_index = {name: idx for idx, name in enumerate(objects)}
    scene_index = {name: idx for idx, name in enumerate(scenes)}
    for row in summary["rows"]:
        values[object_index[row["object_name"]]][scene_index[row["scene"]]] = row["successful"]

    width = max(12, 0.58 * len(scenes) + 4)
    height = max(4.8, 0.72 * len(objects) + 2)
    cmap = LinearSegmentedColormap.from_list("automoma", ["#f3ead8", "#95b9aa", "#245b5c"])
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_title("Successful Trajectories by Object and Scene")
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels(scenes, rotation=45, ha="right")
    ax.set_yticks(range(len(objects)))
    ax.set_yticklabels(objects)
    for i, obj_values in enumerate(values):
        for j, value in enumerate(obj_values):
            if value > 0:
                ax.text(j, i, f"{value // 1000}k" if value >= 1000 else str(value), ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("successful trajectories")
    path = output_dir / "object_scene_heatmap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scene_totals(summary: dict[str, Any], output_dir: Path) -> Path:
    setup_style()
    totals = summary["scene_totals"]
    scenes = sorted(totals, key=totals.get, reverse=True)
    values = [totals[scene] for scene in scenes]

    fig_width = max(12, 0.48 * len(scenes) + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8), constrained_layout=True)
    ax.bar(scenes, values, color="#596f9b", width=0.62)
    ax.set_title("Successful Trajectories by Scene")
    ax.set_ylabel("successful trajectories")
    ax.grid(axis="y", color="#d8d2c5", linewidth=0.8)
    ax.tick_params(axis="x", labelrotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    path = output_dir / "scene_totals.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize and plot AutoMoMa trajectory data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene-dir", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "trajs")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "statistics" / "automoma-500k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-per-object", type=int, default=100_000)
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(args)
    json_path = args.output_dir / "trajectory_summary.json"
    csv_path = args.output_dir / "trajectory_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(csv_path, summary["rows"])

    chart_paths = []
    chart_paths.append(plot_object_totals(summary, args.output_dir))
    if summary["scenes"]:
        chart_paths.append(plot_scene_heatmap(summary, args.output_dir))
        chart_paths.append(plot_scene_totals(summary, args.output_dir))

    print(json.dumps(
        {
            "summary_json": str(json_path),
            "summary_csv": str(csv_path),
            "charts": [str(path) for path in chart_paths],
            "grand_total": summary["grand_total"],
            "object_totals": summary["object_totals"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
