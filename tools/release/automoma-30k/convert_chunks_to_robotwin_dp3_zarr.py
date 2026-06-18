#!/usr/bin/env python3
"""Convert archived AutoMoMa-30K HDF5 chunks directly to RoboTwin DP3 zarr."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import zarr


REPO_ROOT = Path(__file__).resolve().parents[3]
DP3_ROOT = REPO_ROOT / "third_party" / "RoboTwin" / "policy" / "DP3"
DP3_SCRIPTS = DP3_ROOT / "scripts"
if str(DP3_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DP3_SCRIPTS))

from automoma_dp3_utils import PointCloudConfig, rgbd_to_pointcloud


CAMERA_CHOICES = ("ego_topdown", "ego_wrist", "fix_local")
SCENE_RE = re.compile(r"^scene_(\d+)_seed_(\d+)$")
CHUNK_RE = re.compile(r"^chunk_(\d{6})_(\d{6})\.hdf5$")


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def scene_sort_key(path: Path) -> tuple[int, int, str]:
    match = SCENE_RE.match(path.name)
    if match:
        return int(match.group(1)), int(match.group(2)), path.name
    return 1_000_000_000, 1_000_000_000, path.name


def chunk_sort_key(path: Path) -> tuple[int, int, str]:
    match = CHUNK_RE.match(path.name)
    if match:
        return int(match.group(1)), int(match.group(2)), path.name
    return 1_000_000_000, 1_000_000_000, path.name


def demo_sort_key(name: str) -> tuple[int, str]:
    prefix, _, suffix = name.rpartition("_")
    if prefix == "demo" and suffix.isdigit():
        return int(suffix), name
    return 1_000_000_000, name


def scene_names_from_manifest(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload.get("selected_scene_names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{path} does not contain a string list at selected_scene_names")
    return names


def iter_scene_dirs(
    record_root: Path,
    object_name: str,
    scene_start: int,
    scene_count: int,
    scene_names: list[str] | None,
) -> list[Path]:
    object_root = record_root / object_name
    if not object_root.is_dir():
        raise NotADirectoryError(object_root)
    ordered_names = scene_names or [f"scene_{idx}_seed_{idx}" for idx in range(scene_start, scene_start + scene_count)]
    wanted = set(ordered_names)
    scenes = sorted(
        [path for path in object_root.iterdir() if path.is_dir() and path.name in wanted],
        key=scene_sort_key,
    )
    missing = sorted(wanted - {path.name for path in scenes})
    if missing:
        raise FileNotFoundError(f"Missing record scene directories: {missing[:10]}")
    return sorted(scenes, key=lambda path: ordered_names.index(path.name))


def iter_chunk_files(
    record_root: Path,
    object_name: str,
    scene_start: int,
    scene_count: int,
    scene_names: list[str] | None,
) -> Iterator[Path]:
    for scene_dir in iter_scene_dirs(record_root, object_name, scene_start, scene_count, scene_names):
        chunks = sorted(scene_dir.glob("chunk_*.hdf5"), key=chunk_sort_key)
        if not chunks:
            raise FileNotFoundError(f"No chunk_*.hdf5 files in {scene_dir}")
        yield from chunks


def iter_demos(chunk_file: Path) -> Iterator[tuple[Path, str]]:
    with h5py.File(chunk_file, "r") as root:
        if "data" not in root:
            raise KeyError(f"{chunk_file} missing /data")
        for demo_key in sorted(root["data"].keys(), key=demo_sort_key):
            yield chunk_file, demo_key


def count_frames_and_dims(chunk_files: list[Path], camera_view: str) -> tuple[int, int, int, int]:
    total_frames = 0
    demo_count = 0
    state_dim = 0
    action_dim = 0
    for chunk_file in chunk_files:
        with h5py.File(chunk_file, "r") as root:
            if "data" not in root:
                raise KeyError(f"{chunk_file} missing /data")
            for demo_key in sorted(root["data"].keys(), key=demo_sort_key):
                demo = root["data"][demo_key]
                joint_pos = demo["obs"]["joint_pos"]
                actions = demo["processed_actions"]
                rgb = demo["camera_obs"][f"{camera_view}_rgb"]
                depth = demo["camera_obs"][f"{camera_view}_depth"]
                steps = min(len(joint_pos), len(actions), len(rgb), len(depth))
                if steps <= 0:
                    raise ValueError(f"{chunk_file}:{demo_key} has no usable steps")
                total_frames += steps
                demo_count += 1
                if state_dim == 0:
                    state_dim = int(joint_pos.shape[1])
                    action_dim = int(actions.shape[1])
    if demo_count == 0:
        raise ValueError("No demos found")
    return demo_count, total_frames, state_dim, action_dim


def make_output_zarr(args: argparse.Namespace) -> Path:
    if args.output_zarr:
        return args.output_zarr.expanduser().resolve()
    return (DP3_ROOT / "data" / f"{args.task_name}-{args.task_config}-{args.expert_data_num}.zarr").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--object-name", default="microwave_7221")
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-count", type=int, default=30)
    parser.add_argument("--selection-manifest", type=Path, default=None)
    parser.add_argument("--task-name", default="microwave_7221")
    parser.add_argument("--task-config", default="automoma_30k")
    parser.add_argument("--expert-data-num", type=int, default=30000)
    parser.add_argument("--output-zarr", type=Path, default=None)
    parser.add_argument("--camera-view", choices=CAMERA_CHOICES, default="ego_topdown")
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--random-drop-points", type=int, default=5000)
    parser.add_argument("--use-fps", type=str2bool, default=True)
    parser.add_argument("--use-rgb", type=str2bool, default=False)
    parser.add_argument("--fov-deg", type=float, default=60.0)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-count-mismatch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record_root = args.record_root.expanduser().resolve()
    scene_names = scene_names_from_manifest(args.selection_manifest.expanduser().resolve()) if args.selection_manifest else None
    chunk_files = list(iter_chunk_files(record_root, args.object_name, args.scene_start, args.scene_count, scene_names))
    demo_count, total_frames, state_dim, action_dim = count_frames_and_dims(chunk_files, args.camera_view)
    if demo_count != args.expert_data_num and not args.allow_count_mismatch:
        raise ValueError(f"Found {demo_count} demos, expected {args.expert_data_num}")

    save_dir = make_output_zarr(args)
    if save_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{save_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(save_dir)
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    pc_cfg = PointCloudConfig(
        n_points=args.n_points,
        random_drop_points=args.random_drop_points,
        use_fps=args.use_fps,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        fov_deg=args.fov_deg,
        use_rgb=args.use_rgb,
    )
    rng = np.random.default_rng(0)

    print(f"chunks: {len(chunk_files)}")
    print(f"demos: {demo_count}")
    print(f"frames: {total_frames}")
    print(f"zarr: {save_dir}")

    zarr_root = zarr.group(str(save_dir))
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    chunk_len = min(100, total_frames)
    state_arr = zarr_data.create_dataset(
        "state",
        shape=(total_frames, state_dim),
        chunks=(chunk_len, state_dim),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    action_arr = zarr_data.create_dataset(
        "action",
        shape=(total_frames, action_dim),
        chunks=(chunk_len, action_dim),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    pc_width = 6
    point_cloud_arr = zarr_data.create_dataset(
        "point_cloud",
        shape=(total_frames, args.n_points, pc_width),
        chunks=(chunk_len, args.n_points, pc_width),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    episode_ends = np.zeros((demo_count,), dtype=np.int64)

    frame_offset = 0
    demo_index = 0
    for chunk_file in chunk_files:
        with h5py.File(chunk_file, "r") as root:
            for demo_key in sorted(root["data"].keys(), key=demo_sort_key):
                demo = root["data"][demo_key]
                joint_pos = demo["obs"]["joint_pos"][:].astype(np.float32)
                actions = demo["processed_actions"][:].astype(np.float32)
                rgb = demo["camera_obs"][f"{args.camera_view}_rgb"][:]
                depth = demo["camera_obs"][f"{args.camera_view}_depth"][:]
                steps = min(len(joint_pos), len(actions), len(rgb), len(depth))
                end = frame_offset + steps
                state_arr[frame_offset:end] = joint_pos[:steps]
                action_arr[frame_offset:end] = actions[:steps]
                for idx in range(steps):
                    point_cloud_arr[frame_offset + idx] = rgbd_to_pointcloud(rgb[idx], depth[idx], pc_cfg, rng)
                frame_offset = end
                episode_ends[demo_index] = frame_offset
                demo_index += 1
        print(f"[{demo_index}/{demo_count}] {chunk_file}", flush=True)

    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "record_root": str(record_root),
        "object_name": args.object_name,
        "task_name": args.task_name,
        "task_config": args.task_config,
        "expert_data_num": int(args.expert_data_num),
        "scene_start": int(args.scene_start),
        "scene_count": int(args.scene_count),
        "selected_scene_names": scene_names,
        "chunk_count": len(chunk_files),
        "demo_count": int(demo_count),
        "total_frames": int(total_frames),
        "output_zarr": str(save_dir),
        "camera_view": args.camera_view,
        "n_points": int(args.n_points),
        "use_rgb": bool(args.use_rgb),
    }
    (save_dir / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"saved DP3 zarr to {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
