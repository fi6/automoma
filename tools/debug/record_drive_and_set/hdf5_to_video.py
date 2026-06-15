#!/usr/bin/env python3
"""Extract RGB camera observations from an AutoMoMa HDF5 demo into mp4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np


def _sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    def demo_index(key: str) -> int:
        try:
            return int(key.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 10**12

    return sorted(data_group.keys(), key=demo_index)


def _camera_candidates(demo: h5py.Group) -> list[tuple[str, h5py.Dataset]]:
    candidates: list[tuple[str, h5py.Dataset]] = []

    def visit(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if obj.ndim < 4:
            return
        lowered = name.lower()
        if "rgb" not in lowered:
            return
        candidates.append((name, obj))

    demo.visititems(visit)
    return candidates


def _select_camera(candidates: list[tuple[str, h5py.Dataset]], preferred: str) -> tuple[str, h5py.Dataset]:
    if not candidates:
        raise ValueError("No RGB camera datasets found in demo.")
    preferred_lower = preferred.lower()
    for name, dataset in candidates:
        if name.lower().endswith(preferred_lower) or preferred_lower in name.lower():
            return name, dataset
    return candidates[0]


def _frames_to_uint8(array: np.ndarray) -> np.ndarray:
    frames = np.asarray(array)
    if frames.ndim == 5 and frames.shape[1] == 1:
        frames = frames[:, 0]
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with 4 dims after squeeze, got shape={frames.shape}")

    if frames.shape[1] in (1, 3, 4) and frames.shape[-1] not in (1, 3, 4):
        frames = np.moveaxis(frames, 1, -1)
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    if frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB/RGBA frames, got shape={frames.shape}")

    if frames.dtype == np.uint8:
        return np.ascontiguousarray(frames)
    if np.issubdtype(frames.dtype, np.floating):
        if float(np.nanmax(frames)) <= 1.0:
            frames = frames * 255.0
        return np.ascontiguousarray(np.clip(frames, 0, 255).astype(np.uint8))
    return np.ascontiguousarray(np.clip(frames, 0, 255).astype(np.uint8))


def _write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def extract_video(
    hdf5_path: Path,
    output_dir: Path,
    *,
    camera: str,
    demo_key: str,
    fps: int,
    require_width: int | None,
    require_height: int | None,
) -> dict[str, Any]:
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise ValueError(f"No data group in {hdf5_path}")
        data = f["data"]
        if demo_key == "first":
            keys = _sorted_demo_keys(data)
            if not keys:
                raise ValueError(f"No demos in {hdf5_path}")
            demo_key = keys[0]
        if demo_key not in data:
            raise ValueError(f"Demo {demo_key!r} not found in {hdf5_path}")

        demo = data[demo_key]
        candidates = _camera_candidates(demo)
        camera_path, dataset = _select_camera(candidates, camera)
        frames = _frames_to_uint8(dataset[:])

    height, width = int(frames.shape[1]), int(frames.shape[2])
    if require_width is not None and width != require_width:
        raise ValueError(f"Expected video width {require_width}, got {width} from {camera_path}")
    if require_height is not None and height != require_height:
        raise ValueError(f"Expected video height {require_height}, got {height} from {camera_path}")

    stem = hdf5_path.stem
    safe_camera = camera_path.replace("/", "_")
    video_path = output_dir / f"{stem}_{demo_key}_{safe_camera}.mp4"
    _write_video(video_path, frames, fps=fps)
    return {
        "hdf5_path": str(hdf5_path),
        "video_path": str(video_path),
        "demo_key": demo_key,
        "camera_path": camera_path,
        "num_frames": int(frames.shape[0]),
        "width": width,
        "height": height,
        "fps": fps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--camera", default="fix_local_rgb")
    parser.add_argument("--demo_key", default="first")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--require_width", type=int, default=None)
    parser.add_argument("--require_height", type=int, default=None)
    parser.add_argument("--summary_json", type=Path, default=None)
    args = parser.parse_args()

    summary = extract_video(
        args.hdf5,
        args.output_dir,
        camera=args.camera,
        demo_key=args.demo_key,
        fps=args.fps,
        require_width=args.require_width,
        require_height=args.require_height,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

