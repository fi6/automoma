#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from lerobot.datasets.dataset_tools import split_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset


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
        if subset_root.exists():
            shutil.rmtree(subset_root)
        split_dataset(dataset, {subset_name: indices}, output_dir=output_root)

        subset_manifest = {
            "size": size,
            "indices": indices,
            "root": str(subset_root),
            "repo_id": subset_name,
        }
        manifest["subsets"][str(size)] = subset_manifest
        (subset_root / "selected_episodes.json").write_text(json.dumps(subset_manifest, indent=2), encoding="utf-8")
        (output_root / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Built {len(manifest['subsets'])} subsets under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
