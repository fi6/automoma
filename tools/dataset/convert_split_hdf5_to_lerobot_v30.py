#!/usr/bin/env python3
"""Convert split AutoMoMa HDF5 episodes to LeRobot v3.0 without merging first."""

from __future__ import annotations

import argparse
from functools import lru_cache
import logging
import re
import sys
from pathlib import Path
from typing import Any

import h5py
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ARENA = REPO_ROOT / "third_party" / "IsaacLab-Arena"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
for import_path in (ISAACLAB_ARENA, LEROBOT_SRC):
    if import_path.exists():
        sys.path.insert(0, str(import_path))

LOGGER = logging.getLogger(__name__)
EPISODE_FILE_RE = re.compile(r"^episode_(\d{6})\.hdf5$")


@lru_cache(maxsize=1)
def _load_converter_dependencies() -> dict[str, Any]:
    from isaaclab_arena_gr00t.config.dataset_config import Gr00tDatasetConfig
    from isaaclab_arena_gr00t.data_utils.convert_hdf5_to_lerobot_v30 import (
        _ensure_output_dir,
        _infer_features,
        _prepare_episode_data,
    )
    from isaaclab_arena_gr00t.data_utils.io_utils import create_config_from_yaml, load_config_from_yaml
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return {
        "Gr00tDatasetConfig": Gr00tDatasetConfig,
        "LeRobotDataset": LeRobotDataset,
        "create_config_from_yaml": create_config_from_yaml,
        "ensure_output_dir": _ensure_output_dir,
        "infer_features": _infer_features,
        "load_config_from_yaml": load_config_from_yaml,
        "prepare_episode_data": _prepare_episode_data,
    }


def _episode_file_sort_key(path: Path) -> tuple[int, str]:
    match = EPISODE_FILE_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return 1_000_000_000, path.name


def _demo_sort_key(demo_name: str) -> tuple[int, str]:
    parts = demo_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1]), demo_name
    return 1_000_000_000, demo_name


def _episode_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    files = sorted(input_dir.glob("episode_*.hdf5"), key=_episode_file_sort_key)
    if not files:
        raise ValueError(f"No episode_*.hdf5 files found in {input_dir}")
    return files


def _single_demo_key(data_group: h5py.Group, source: Path) -> str:
    demo_keys = sorted([key for key in data_group.keys() if key.startswith("demo_")], key=_demo_sort_key)
    if len(demo_keys) != 1:
        raise ValueError(f"{source} must contain exactly one demo_* group, found {len(demo_keys)}")
    return demo_keys[0]


def _load_config(args: argparse.Namespace) -> Any:
    deps = _load_converter_dependencies()
    config_class = deps["Gr00tDatasetConfig"]
    if args.data_root is None and args.hdf5_name is None:
        return deps["create_config_from_yaml"](args.yaml_file, config_class)

    config_data = deps["load_config_from_yaml"](args.yaml_file, config_class)
    if args.data_root is not None:
        config_data["data_root"] = Path(args.data_root)
    if args.hdf5_name is not None:
        config_data["hdf5_name"] = args.hdf5_name
    return config_class(**config_data)


def convert_split_hdf5_to_lerobot_v30(
    config: Any,
    repo_id: str,
    output_dir: Path,
    vcodec: str,
    image_writer_threads: int,
    image_writer_processes: int,
    batch_encoding_size: int,
) -> None:
    deps = _load_converter_dependencies()
    ensure_output_dir = deps["ensure_output_dir"]
    infer_features = deps["infer_features"]
    prepare_episode_data = deps["prepare_episode_data"]
    lerobot_dataset = deps["LeRobotDataset"]

    split_dir = config.hdf5_file_path
    episode_files = _episode_files(split_dir)

    ensure_output_dir(output_dir)

    LOGGER.info("Loading split HDF5 directory: %s", split_dir)
    LOGGER.info("Found %d episode file(s)", len(episode_files))

    dataset: Any | None = None
    for episode_file in tqdm(episode_files, desc="episodes"):
        with h5py.File(episode_file, "r") as hdf5_handler:
            if "data" not in hdf5_handler:
                raise KeyError(f"{episode_file} is missing the HDF5 'data' group")

            hdf5_data = hdf5_handler["data"]
            demo_key = _single_demo_key(hdf5_data, episode_file)
            trajectory = hdf5_data[demo_key]
            episode_data, frames_by_key = prepare_episode_data(trajectory, config)

        if dataset is None:
            example_episode: dict[str, Any] = dict(episode_data)
            video_shapes = {}
            for video_key, frames in frames_by_key.items():
                video_shapes[video_key] = (frames.shape[1], frames.shape[2], frames.shape[3])
                example_episode[video_key] = frames[0]
            features = infer_features(example_episode, video_shapes, config, vcodec)

            dataset = lerobot_dataset.create(
                repo_id=repo_id,
                fps=config.fps,
                features=features,
                root=output_dir,
                robot_type=config.robot_type,
                use_videos=True,
                image_writer_processes=image_writer_processes,
                image_writer_threads=image_writer_threads,
                batch_encoding_size=batch_encoding_size,
                vcodec=vcodec,
            )

        lengths = {frames.shape[0] for frames in frames_by_key.values()}
        if len(lengths) != 1:
            raise ValueError(f"Mismatched video lengths in {episode_file}: {lengths}")
        length = lengths.pop()

        for t in range(length):
            frame = {}
            for key, values in episode_data.items():
                frame[key] = values[t]

            for video_key, frames in frames_by_key.items():
                frame[video_key] = frames[t]
            frame["task"] = config.language_instruction
            dataset.add_frame(frame)

        dataset.save_episode()

    if dataset is not None:
        dataset.finalize()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Convert split HDF5 episodes to LeRobot v3.0 format")
    parser.add_argument("--yaml_file", required=True, help="Path to YAML configuration file")
    parser.add_argument(
        "--data_root",
        default=None,
        help="Override data_root from YAML (directory containing the split HDF5 directory)",
    )
    parser.add_argument(
        "--hdf5_name",
        default=None,
        help="Override hdf5_name from YAML (split HDF5 directory name)",
    )
    parser.add_argument(
        "--repo_id",
        default=None,
        help="Dataset repo id used for metadata (defaults to split directory name)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for v3.0 dataset (defaults to config.lerobot_data_dir)",
    )
    parser.add_argument(
        "--vcodec",
        default="h264",
        help="Video codec to use for encoding (h264, hevc, libsvtav1)",
    )
    parser.add_argument(
        "--image_writer_threads",
        type=int,
        default=0,
        help="Number of threads for async image writing (0 disables async writing)",
    )
    parser.add_argument(
        "--image_writer_processes",
        type=int,
        default=0,
        help="Number of processes for async image writing (0 disables multiprocessing)",
    )
    parser.add_argument(
        "--batch_encoding_size",
        type=int,
        default=1,
        help="Number of episodes to batch before encoding videos",
    )

    args = parser.parse_args()
    config = _load_config(args)

    if not config.hdf5_file_path.is_dir():
        raise NotADirectoryError(
            f"{config.hdf5_file_path} is not a split HDF5 directory; "
            "use the standard merged-HDF5 converter for single .hdf5 files."
        )

    repo_id = args.repo_id
    if repo_id is None:
        repo_id = Path(config.hdf5_name).stem

    output_dir = Path(args.output_dir) if args.output_dir else config.lerobot_data_dir

    convert_split_hdf5_to_lerobot_v30(
        config=config,
        repo_id=repo_id,
        output_dir=output_dir,
        vcodec=args.vcodec,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        batch_encoding_size=args.batch_encoding_size,
    )


if __name__ == "__main__":
    main()
