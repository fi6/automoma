#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def copy_subset(dataset: LeRobotDataset, indices: list[int], repo_id: str, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    selected = set(indices)
    new_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(dataset.fps),
        root=root,
        robot_type=dataset.meta.robot_type,
        features=dataset.meta.info["features"],
        use_videos=len(dataset.meta.video_keys) > 0,
    )

    prev_episode_index = None
    wrote_any = False
    for frame_idx in range(len(dataset)):
        frame = dataset[frame_idx]
        episode_index = int(frame["episode_index"].item())
        if episode_index not in selected:
            continue

        if prev_episode_index is None:
            prev_episode_index = episode_index
        elif episode_index != prev_episode_index:
            new_dataset.save_episode()
            prev_episode_index = episode_index

        new_frame = {}
        for key, value in frame.items():
            if key in ("task_index", "timestamp", "episode_index", "frame_index", "index", "task"):
                continue
            if hasattr(value, "dim") and value.dim() == 0:
                value = value.unsqueeze(0)
            new_frame[key] = value
        new_frame["task"] = frame["task"]
        new_dataset.add_frame(new_frame)
        wrote_any = True

    if wrote_any:
        new_dataset.save_episode()
    new_dataset.finalize()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build nested validation subsets from a LeRobot dataset")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--subset-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    episode_count = dataset.meta.total_episodes
    ordering = list(range(episode_count))
    random.Random(args.seed).shuffle(ordering)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "repo_id": args.repo_id,
        "root": str(Path(args.root)),
        "seed": args.seed,
        "episode_count": episode_count,
        "ordering": ordering,
        "subsets": {},
    }

    for size in sorted(args.subset_sizes):
        if size > episode_count:
            raise ValueError(f"Subset size {size} exceeds episode count {episode_count}")
        indices = ordering[:size]
        subset_name = f"{args.prefix}-{size}"
        subset_root = output_root / subset_name
        copy_subset(dataset, indices, subset_name, subset_root)
        subset_manifest = {
            "size": size,
            "indices": indices,
            "root": str(subset_root),
            "repo_id": subset_name,
        }
        manifest["subsets"][str(size)] = subset_manifest
        (subset_root / "selected_episodes.json").write_text(json.dumps(subset_manifest, indent=2), encoding="utf-8")

    (output_root / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
