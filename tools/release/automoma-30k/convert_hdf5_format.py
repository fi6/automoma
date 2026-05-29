#!/usr/bin/env python3
"""Convert legacy per-episode AutoMoMa HDF5 files to the current recorder schema."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "automoma_30scenes"
DEFAULT_INPUT_ROOT = DATA_ROOT / "automoma-30k-sort"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "automoma-30k-convert"
DEFAULT_REFERENCE = REPO_ROOT / "data" / "automoma" / "summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5"

SCENE_RE = re.compile(r"scene_(\d+)_seed_(\d+)$")
EPISODE_RE = re.compile(r"episode(\d+)\.hdf5$")


CAMERAS = ("ego_topdown", "ego_wrist", "fix_local")


def natural_scene_key(path: Path) -> tuple[int, int, str]:
    match = SCENE_RE.fullmatch(path.name)
    if not match:
        return (10**9, 10**9, path.name)
    return (int(match.group(1)), int(match.group(2)), path.name)


def episode_key(path: Path) -> tuple[int, str]:
    match = EPISODE_RE.fullmatch(path.name)
    if not match:
        return (10**9, path.name)
    return (int(match.group(1)), path.name)


def identity_root_pose(length: int) -> np.ndarray:
    root_pose = np.zeros((length, 7), dtype=np.float32)
    root_pose[:, 3] = 1.0
    return root_pose


def require_dataset(group: h5py.Group, path: str) -> h5py.Dataset:
    if path not in group:
        raise KeyError(f"missing required dataset: {path}")
    obj = group[path]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"not a dataset: {path}")
    return obj


def read_joint_pos(src: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(require_dataset(src, "obs/joint/mobile_base"), dtype=np.float32)
    arm = np.asarray(require_dataset(src, "obs/joint/arm"), dtype=np.float32)
    gripper = np.asarray(require_dataset(src, "obs/joint/gripper"), dtype=np.float32)
    if gripper.ndim == 1:
        gripper_2d = np.repeat(gripper[:, None], 2, axis=1)
    elif gripper.ndim == 2 and gripper.shape[1] == 1:
        gripper_2d = np.repeat(gripper, 2, axis=1)
    elif gripper.ndim == 2 and gripper.shape[1] == 2:
        gripper_2d = gripper
    else:
        raise ValueError(f"unsupported gripper shape: {gripper.shape}")
    joint_pos = np.concatenate([base, arm, gripper_2d], axis=1).astype(np.float32)
    return joint_pos, gripper_2d.astype(np.float32)


def build_actions(joint_pos: np.ndarray) -> np.ndarray:
    actions = joint_pos.copy()
    if len(actions) <= 1:
        actions[:, :3] = 0.0
        return actions.astype(np.float32)

    # Current recording uses mobile-base-relative actions. For legacy state-only
    # files, derive base commands from the next recorded base target.
    actions[:-1, :3] = joint_pos[1:, :3] - joint_pos[:-1, :3]
    actions[-1, :3] = actions[-2, :3]
    actions[:-1, 3:] = joint_pos[1:, 3:]
    actions[-1, 3:] = joint_pos[-1, 3:]
    return actions.astype(np.float32)


def read_eef(src: h5py.File, timesteps: int) -> tuple[np.ndarray, np.ndarray]:
    if "obs/eef" not in src:
        return np.zeros((timesteps, 3), dtype=np.float32), np.zeros((timesteps, 4), dtype=np.float32)
    eef = np.asarray(src["obs/eef"], dtype=np.float32)
    if eef.shape != (timesteps, 7):
        raise ValueError(f"expected obs/eef shape {(timesteps, 7)}, found {eef.shape}")
    return eef[:, :3], eef[:, 3:]


def create_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    group.create_dataset(name, data=data)


def copy_camera_obs(src: h5py.File, demo: h5py.Group) -> None:
    camera_obs = demo.create_group("camera_obs")
    for camera in CAMERAS:
        rgb = np.asarray(require_dataset(src, f"obs/rgb/{camera}"), dtype=np.uint8)
        depth = np.asarray(require_dataset(src, f"obs/depth/{camera}"), dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., None]
        create_dataset(camera_obs, f"{camera}_rgb", rgb)
        create_dataset(camera_obs, f"{camera}_depth", depth)


def add_articulation_state(parent: h5py.Group, object_name: str, joint_pos: np.ndarray) -> None:
    articulation = parent.create_group("articulation")

    obj = articulation.create_group(object_name)
    timesteps = joint_pos.shape[0]
    create_dataset(obj, "joint_position", np.zeros((timesteps, 1), dtype=np.float32))
    create_dataset(obj, "joint_velocity", np.zeros((timesteps, 1), dtype=np.float32))
    create_dataset(obj, "root_pose", identity_root_pose(timesteps))
    create_dataset(obj, "root_velocity", np.zeros((timesteps, 6), dtype=np.float32))

    robot = articulation.create_group("robot")
    create_dataset(robot, "joint_position", joint_pos.astype(np.float32))
    create_dataset(robot, "joint_velocity", np.zeros_like(joint_pos, dtype=np.float32))
    create_dataset(robot, "root_pose", identity_root_pose(timesteps))
    create_dataset(robot, "root_velocity", np.zeros((timesteps, 6), dtype=np.float32))


def add_initial_state(demo: h5py.Group, object_name: str, joint_pos: np.ndarray) -> None:
    initial_state = demo.create_group("initial_state")
    articulation = initial_state.create_group("articulation")

    obj = articulation.create_group(object_name)
    create_dataset(obj, "joint_position", np.zeros((1, 1), dtype=np.float32))
    create_dataset(obj, "joint_velocity", np.zeros((1, 1), dtype=np.float32))
    create_dataset(obj, "root_pose", identity_root_pose(1))
    create_dataset(obj, "root_velocity", np.zeros((1, 6), dtype=np.float32))

    robot = articulation.create_group("robot")
    create_dataset(robot, "joint_position", joint_pos[:1].astype(np.float32))
    create_dataset(robot, "joint_velocity", np.zeros((1, joint_pos.shape[1]), dtype=np.float32))
    create_dataset(robot, "root_pose", identity_root_pose(1))
    create_dataset(robot, "root_velocity", np.zeros((1, 6), dtype=np.float32))


def convert_one(src_path: Path, dst_path: Path, object_name: str, env_args: str, overwrite: bool) -> dict[str, object]:
    if dst_path.exists() and not overwrite:
        return {"source": str(src_path), "output": str(dst_path), "status": "skipped_exists"}

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with h5py.File(src_path, "r") as src:
        joint_pos, gripper_2d = read_joint_pos(src)
        timesteps = int(joint_pos.shape[0])
        if timesteps == 0:
            raise ValueError("empty episode")

        actions = build_actions(joint_pos)
        eef_pos, eef_quat = read_eef(src, timesteps)
        with h5py.File(tmp_path, "w") as dst:
            data = dst.create_group("data")
            data.attrs["env_args"] = env_args
            data.attrs["total"] = timesteps

            demo = data.create_group("demo_0")
            demo.attrs["num_samples"] = timesteps
            demo.attrs["success"] = False

            create_dataset(demo, "actions", actions)
            create_dataset(demo, "processed_actions", actions)

            obs = demo.create_group("obs")
            create_dataset(obs, "actions", actions)
            create_dataset(obs, "eef_pos", eef_pos)
            create_dataset(obs, "eef_quat", eef_quat)
            create_dataset(obs, "gripper_pos", gripper_2d)
            create_dataset(obs, "joint_pos", joint_pos)
            create_dataset(obs, "joint_vel", np.zeros_like(joint_pos, dtype=np.float32))

            copy_camera_obs(src, demo)
            add_initial_state(demo, object_name, joint_pos)
            states = demo.create_group("states")
            add_articulation_state(states, object_name, joint_pos)

    os.replace(tmp_path, dst_path)
    return {
        "source": str(src_path.resolve()),
        "output": str(dst_path.resolve()),
        "status": "converted",
        "num_samples": timesteps,
    }


def iter_sources(input_root: Path, scenes: set[str] | None) -> Iterable[tuple[str, Path]]:
    scene_dirs = sorted(
        [path for path in input_root.iterdir() if path.is_dir() and (scenes is None or path.name in scenes)],
        key=natural_scene_key,
    )
    for scene_dir in scene_dirs:
        for episode in sorted(scene_dir.glob("*.hdf5"), key=episode_key):
            yield scene_dir.name, episode


def iter_manifest_sources(manifest_path: Path, scenes: set[str] | None) -> Iterable[tuple[str, Path]]:
    with manifest_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            scene = row["scene"]
            if scenes is not None and scene not in scenes:
                continue
            yield scene, Path(row["source"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Read source paths from organize_hdf5.py manifest.jsonl instead of input-root scene folders.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference-file", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--object-name", default="microwave_7221")
    parser.add_argument("--scene", action="append", help="Scene to convert. May be passed multiple times.")
    parser.add_argument("--start", type=int, default=0, help="Skip this many matching source files.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of files to convert.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--manifest-name",
        default="conversion_manifest.jsonl",
        help="Manifest filename written under output-root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root

    if args.manifest is None and not input_root.exists():
        raise FileNotFoundError(f"input root does not exist: {input_root}")
    if args.manifest is not None and not args.manifest.exists():
        raise FileNotFoundError(f"manifest does not exist: {args.manifest}")

    with h5py.File(args.reference_file, "r") as reference:
        env_args = reference["data"].attrs["env_args"]

    scenes = set(args.scene) if args.scene else None
    if args.manifest is None:
        selected = list(iter_sources(input_root, scenes))
    else:
        selected = list(iter_manifest_sources(args.manifest, scenes))
    selected = selected[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Files selected: {len(selected)}")
    if args.dry_run:
        for scene, src_path in selected[:20]:
            print(f"{scene}: {src_path}")
        if len(selected) > 20:
            print(f"... {len(selected) - 20} more")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / args.manifest_name
    converted = 0
    skipped = 0
    failed = 0
    with manifest_path.open("a") as manifest:
        for scene, src_path in tqdm(selected, desc="convert"):
            dst_path = output_root / scene / src_path.name
            try:
                row = convert_one(src_path, dst_path, args.object_name, env_args, args.overwrite)
            except Exception as exc:
                row = {
                    "source": str(src_path.resolve()),
                    "output": str(dst_path.resolve()),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            manifest.write(json.dumps(row) + "\n")
            manifest.flush()
            if row["status"] == "converted":
                converted += 1
            elif row["status"] == "skipped_exists":
                skipped += 1
            else:
                failed += 1

    summary = {"selected": len(selected), "converted": converted, "skipped": skipped, "failed": failed}
    (output_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
