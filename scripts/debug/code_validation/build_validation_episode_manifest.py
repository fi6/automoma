#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build nested validation episode manifests from a LeRobot dataset")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subset-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    episode_count = dataset.meta.total_episodes
    ordering = list(range(episode_count))
    random.Random(args.seed).shuffle(ordering)

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
        manifest["subsets"][str(size)] = {
            "size": size,
            "episodes": ordering[:size],
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote manifest with {len(manifest['subsets'])} subsets to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
