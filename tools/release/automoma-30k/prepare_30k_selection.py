#!/usr/bin/env python3
"""Prepare deterministic AutoMoMa-30K trajectory subsets and archive plan files."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch


REQUIRED_KEYS = (
    "start_robot",
    "start_obj",
    "goal_robot",
    "goal_obj",
    "traj_robot",
    "traj_obj",
    "traj_success",
)
SCENE_RE = re.compile(r"^scene_(\d+)_seed_(\d+)$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def scene_name(scene_id: int) -> str:
    return f"scene_{scene_id}_seed_{scene_id}"


def scene_sort_key(name: str) -> tuple[int, int, str]:
    match = SCENE_RE.match(name)
    if match:
        return int(match.group(1)), int(match.group(2)), name
    return 1_000_000_000, 1_000_000_000, name


def discover_scenes(source_base: Path, *, scene_start: int, scene_count: int, scene_mode: str) -> list[tuple[int, str]]:
    if scene_mode == "contiguous":
        return [(scene_id, scene_name(scene_id)) for scene_id in range(scene_start, scene_start + scene_count)]

    rows: list[tuple[int, str]] = []
    for path in source_base.iterdir():
        match = SCENE_RE.match(path.name)
        if not match:
            continue
        scene_id = int(match.group(1))
        seed_id = int(match.group(2))
        if scene_id != seed_id or scene_id < scene_start:
            continue
        traj = path / "train" / "traj_data_train.pt"
        if traj.is_file():
            rows.append((scene_id, path.name))
    rows.sort(key=lambda item: scene_sort_key(item[1]))
    if len(rows) < scene_count:
        raise ValueError(f"Only {len(rows)} scenes available from scene_start={scene_start}; requested {scene_count}")
    return rows[:scene_count]


def load_traj(path: Path) -> dict[str, object]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not load as a dict")
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise KeyError(f"{path} is missing required key(s): {missing}")
    return data


def tensor_count(data: dict[str, object]) -> int:
    traj_robot = data["traj_robot"]
    if not isinstance(traj_robot, torch.Tensor) or traj_robot.ndim < 1:
        raise TypeError("traj_robot must be a tensor with a batch dimension")
    return int(traj_robot.shape[0])


def select_indices(data: dict[str, object], *, count: int, seed: int, successful_only: bool) -> torch.Tensor:
    available = tensor_count(data)
    if successful_only:
        success = data["traj_success"]
        if not isinstance(success, torch.Tensor) or int(success.shape[0]) != available:
            raise TypeError("traj_success must be a tensor aligned with traj_robot")
        eligible = torch.nonzero(success.to(dtype=torch.bool), as_tuple=False).flatten().to(dtype=torch.long)
    else:
        eligible = torch.arange(available, dtype=torch.long)

    if int(eligible.numel()) < count:
        raise ValueError(f"Only {int(eligible.numel())} eligible trajectories; requested {count}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(int(eligible.numel()), generator=generator)
    return eligible[order[:count]].clone()


def subset_traj(data: dict[str, object], source_indices: torch.Tensor) -> dict[str, object]:
    available = tensor_count(data)
    subset: dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and int(value.shape[0]) == available:
            subset[key] = value[source_indices].clone()
        else:
            subset[key] = value
    subset["subset_source_indices"] = source_indices.clone()
    subset["subset_indices"] = torch.arange(int(source_indices.numel()), dtype=torch.long)
    return subset


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--object-name", default="microwave_7221")
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-count", type=int, default=30)
    parser.add_argument("--scene-mode", choices=("contiguous", "first-existing"), default="contiguous")
    parser.add_argument("--episodes-per-scene", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--split", default="train")
    parser.add_argument("--successful-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy-source-plan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    archive = args.archive_root.expanduser().resolve()
    source_base = root / "data" / "trajs" / "summit_franka" / args.object_name
    selected_base = archive / "plan" / "selected_1000" / "summit_franka" / args.object_name
    source_archive_base = archive / "plan" / "source_2500" / "summit_franka" / args.object_name
    manifest_dir = archive / "manifests" / "selection"

    if args.scene_count < 1:
        raise ValueError("--scene-count must be positive")
    if args.episodes_per_scene < 1:
        raise ValueError("--episodes-per-scene must be positive")

    archive.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    selected_scenes = discover_scenes(
        source_base,
        scene_start=args.scene_start,
        scene_count=args.scene_count,
        scene_mode=args.scene_mode,
    )

    for scene_id, scene in selected_scenes:
        source = source_base / scene / args.split / f"traj_data_{args.split}.pt"
        if not source.is_file():
            raise FileNotFoundError(source)

        selected = selected_base / scene / args.split / f"traj_data_{args.split}.pt"
        source_copy = source_archive_base / scene / args.split / f"traj_data_{args.split}.pt"
        mapping_path = manifest_dir / f"{scene}.json"

        data = load_traj(source)
        available = tensor_count(data)
        success = data["traj_success"]
        success_count = int(success.to(dtype=torch.bool).sum().item()) if isinstance(success, torch.Tensor) else -1
        scene_seed = int(args.seed + scene_id)
        indices = select_indices(
            data,
            count=args.episodes_per_scene,
            seed=scene_seed,
            successful_only=args.successful_only,
        )

        if args.copy_source_plan:
            source_copy.parent.mkdir(parents=True, exist_ok=True)
            if source_copy.exists() and not args.overwrite:
                pass
            else:
                shutil.copy2(source, source_copy)

        selected.parent.mkdir(parents=True, exist_ok=True)
        if selected.exists() and not args.overwrite:
            selected_data = load_traj(selected)
            selected_count = tensor_count(selected_data)
            if selected_count != args.episodes_per_scene:
                raise ValueError(f"Existing {selected} has {selected_count} trajectories")
        else:
            torch.save(subset_traj(data, indices), selected)

        row = {
            "object_name": args.object_name,
            "scene": scene,
            "scene_id": scene_id,
            "split": args.split,
            "source_traj_file": str(source),
            "archived_source_traj_file": str(source_copy) if args.copy_source_plan else None,
            "selected_traj_file": str(selected),
            "selection_seed": scene_seed,
            "base_seed": int(args.seed),
            "available_trajectories": available,
            "successful_trajectories": success_count,
            "selected_count": int(indices.numel()),
            "successful_only": bool(args.successful_only),
            "source_indices": [int(value) for value in indices.tolist()],
        }
        row["episodes"] = [
            {
                "selected_episode_index": int(i),
                "source_episode_index": int(source_index),
            }
            for i, source_index in enumerate(indices.tolist())
        ]
        write_json(mapping_path, row)
        rows.append(row)
        print(f"{scene}: selected {int(indices.numel())} / {available} -> {selected}")

    summary = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "archive_root": str(archive),
        "repo_root": str(root),
        "object_name": args.object_name,
        "split": args.split,
        "scene_start": int(args.scene_start),
        "scene_count": int(args.scene_count),
        "scene_mode": args.scene_mode,
        "selected_scene_names": [scene for _scene_id, scene in selected_scenes],
        "episodes_per_scene": int(args.episodes_per_scene),
        "total_selected": int(sum(int(row["selected_count"]) for row in rows)),
        "base_seed": int(args.seed),
        "successful_only": bool(args.successful_only),
        "scenes": rows,
    }
    write_json(archive / "manifests" / "selection_manifest.json", summary)
    print(f"Wrote selection manifest: {archive / 'manifests' / 'selection_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
