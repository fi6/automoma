#!/usr/bin/env python3
"""Slice a RobotWin DP3 zarr dataset to the first N episodes."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import zarr


def copy_frame_array(src: zarr.Array, dst_group: zarr.Group, name: str, frame_count: int) -> None:
    chunks = (min(src.chunks[0], frame_count), *src.chunks[1:])
    dst = dst_group.create_dataset(
        name,
        shape=(frame_count, *src.shape[1:]),
        chunks=chunks,
        dtype=src.dtype,
        overwrite=True,
        compressor=src.compressor,
    )
    step = max(chunks[0] * 64, 1)
    for start in range(0, frame_count, step):
        end = min(start + step, frame_count)
        dst[start:end] = src[start:end]
        print(f"{name}: copied {end}/{frame_count}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zarr", type=Path, required=True)
    parser.add_argument("--output-zarr", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_zarr = args.input_zarr.expanduser().resolve()
    output_zarr = args.output_zarr.expanduser().resolve()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if output_zarr.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_zarr} exists; pass --overwrite to replace it")
        shutil.rmtree(output_zarr)
    output_zarr.parent.mkdir(parents=True, exist_ok=True)

    src_root = zarr.open(str(input_zarr), mode="r")
    episode_ends = src_root["meta"]["episode_ends"][:]
    if args.episodes > len(episode_ends):
        raise ValueError(f"--episodes {args.episodes} exceeds source episode count {len(episode_ends)}")
    sliced_episode_ends = episode_ends[: args.episodes].copy()
    frame_count = int(sliced_episode_ends[-1])

    dst_root = zarr.group(str(output_zarr))
    dst_data = dst_root.create_group("data")
    dst_meta = dst_root.create_group("meta")
    for key in ("state", "action", "point_cloud"):
        copy_frame_array(src_root["data"][key], dst_data, key, frame_count)
    dst_meta.create_dataset(
        "episode_ends",
        data=sliced_episode_ends,
        chunks=(min(args.episodes, src_root["meta"]["episode_ends"].chunks[0]),),
        dtype=src_root["meta"]["episode_ends"].dtype,
        overwrite=True,
        compressor=src_root["meta"]["episode_ends"].compressor,
    )

    manifest_path = input_zarr / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update(
        {
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "source_zarr": str(input_zarr),
            "output_zarr": str(output_zarr),
            "slice_episodes": int(args.episodes),
            "demo_count": int(args.episodes),
            "total_frames": int(frame_count),
            "state_dim": int(src_root["data"]["state"].shape[1]),
            "action_dim": int(src_root["data"]["action"].shape[1]),
            "n_points": int(src_root["data"]["point_cloud"].shape[1]),
        }
    )
    (output_zarr / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"saved sliced zarr to {output_zarr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
