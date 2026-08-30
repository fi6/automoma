#!/usr/bin/env python3
"""Convert retained demos from accepted.jsonl into finalized LeRobot chunks.

Source HDF5 files are read in place and never merged or rewritten. Manifest
line order, followed by each record's ``episodes`` order, is authoritative.
"""

from __future__ import annotations

import argparse
import json
import logging
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import h5py

from convert_split_hdf5_to_lerobot_v30 import _load_config, _load_converter_dependencies


LOGGER = logging.getLogger(__name__)
PLANNING_METADATA_ATTRS = ("traj_index", "grasp_id", "grasp_pose", "goal_angle", "scene_id", "object_id")


@dataclass(frozen=True)
class AcceptedEpisodeRef:
    hdf5_path: Path
    demo_key: str
    job_id: str
    metadata: dict[str, Any]


def _json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def accepted_episode_refs(manifest: Path, *, hdf5_name: str = "record.hdf5") -> list[AcceptedEpisodeRef]:
    refs: list[AcceptedEpisodeRef] = []
    with manifest.open("r", encoding="utf-8") as stream:
        records = [(line_no, json.loads(line)) for line_no, line in enumerate(stream, 1) if line.strip()]
    for line_no, record in records:
        episodes = record.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"{manifest}:{line_no} must contain a non-empty episodes list")
        accepted_path = Path(record["path"])
        hdf5_path = accepted_path if accepted_path.suffix == ".hdf5" else accepted_path / hdf5_name
        with h5py.File(hdf5_path, "r") as root:
            if "data" not in root:
                raise ValueError(f"{hdf5_path} is missing data group")
            by_traj: dict[int, tuple[str, dict[str, Any]]] = {}
            for demo_key, demo in root["data"].items():
                missing = [key for key in PLANNING_METADATA_ATTRS if key not in demo.attrs]
                if missing:
                    raise ValueError(f"{hdf5_path}:{demo.name} is missing attrs {missing}")
                metadata = {key: _json_value(demo.attrs[key]) for key in PLANNING_METADATA_ATTRS}
                traj_index = int(metadata["traj_index"])
                if traj_index in by_traj:
                    raise ValueError(f"{hdf5_path} contains duplicate traj_index {traj_index}")
                by_traj[traj_index] = (demo_key, metadata)
            actual = set(by_traj)
            expected = {int(index) for index in episodes}
            if actual != expected or len(expected) != len(episodes):
                raise ValueError(
                    f"{manifest}:{line_no} episodes {episodes} do not exactly match {hdf5_path} demos {sorted(actual)}"
                )
            for traj_index in episodes:
                demo_key, metadata = by_traj[int(traj_index)]
                refs.append(
                    AcceptedEpisodeRef(
                        hdf5_path=hdf5_path,
                        demo_key=demo_key,
                        job_id=str(record["job_id"]),
                        metadata=metadata,
                    )
                )
    return refs


def chunks(items: list[AcceptedEpisodeRef], size: int) -> Iterator[list[AcceptedEpisodeRef]]:
    if size < 1:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _install_episode_metadata_hook(dataset: Any) -> dict[str, Any]:
    """Merge pending planning fields into LeRobot's episode parquet row."""
    pending: dict[str, Any] = {}
    original = dataset.meta.save_episode

    def save_episode(
        _self: Any,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict,
        episode_metadata: dict,
    ) -> None:
        merged = dict(episode_metadata)
        merged.update(pending)
        original(episode_index, episode_length, episode_tasks, episode_stats, merged)

    dataset.meta.save_episode = types.MethodType(save_episode, dataset.meta)
    return pending


def convert_chunk(
    refs: list[AcceptedEpisodeRef],
    *,
    config: Any,
    repo_id: str,
    output_dir: Path,
    vcodec: str,
    image_writer_threads: int,
    image_writer_processes: int,
    batch_encoding_size: int,
) -> None:
    deps = _load_converter_dependencies()
    deps["ensure_output_dir"](output_dir)
    dataset = None
    pending_metadata: dict[str, Any] | None = None
    for ref in refs:
        with h5py.File(ref.hdf5_path, "r") as root:
            trajectory = root["data"][ref.demo_key]
            episode_data, frames_by_key = deps["prepare_episode_data"](trajectory, config)

        if dataset is None:
            example_episode = dict(episode_data)
            video_shapes = {}
            for video_key, frames in frames_by_key.items():
                video_shapes[video_key] = (frames.shape[1], frames.shape[2], frames.shape[3])
                example_episode[video_key] = frames[0]
            features = deps["infer_features"](example_episode, video_shapes, config, vcodec)
            dataset = deps["LeRobotDataset"].create(
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
            pending_metadata = _install_episode_metadata_hook(dataset)

        lengths = {frames.shape[0] for frames in frames_by_key.values()}
        if len(lengths) != 1:
            raise ValueError(f"Mismatched video lengths in {ref.hdf5_path}:{ref.demo_key}: {lengths}")
        length = lengths.pop()
        for frame_index in range(length):
            frame = {key: values[frame_index] for key, values in episode_data.items()}
            frame.update({key: frames[frame_index] for key, frames in frames_by_key.items()})
            frame["task"] = config.language_instruction
            dataset.add_frame(frame)

        assert pending_metadata is not None
        pending_metadata.clear()
        pending_metadata.update(ref.metadata)
        pending_metadata["source_job_id"] = ref.job_id
        pending_metadata["source_hdf5"] = str(ref.hdf5_path)
        pending_metadata["source_demo"] = ref.demo_key
        dataset.save_episode()

    if dataset is not None:
        # This is required for the final partial (< chunk_size) chunk too.
        dataset.finalize()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--yaml_file", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--hdf5-name", default="record.hdf5")
    parser.add_argument("--episodes-per-chunk", type=int, default=200)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--image_writer_threads", type=int, default=0)
    parser.add_argument("--image_writer_processes", type=int, default=0)
    parser.add_argument("--batch_encoding_size", type=int, default=1)
    args = parser.parse_args()
    # Reuse the standard converter's config parser without changing HDF5.
    config_args = argparse.Namespace(yaml_file=args.yaml_file, data_root=None, hdf5_name=None)
    config = _load_config(config_args)
    refs = accepted_episode_refs(args.manifest, hdf5_name=args.hdf5_name)
    for chunk_index, chunk_refs in enumerate(chunks(refs, args.episodes_per_chunk)):
        chunk_dir = args.output_dir / f"chunk_{chunk_index:06d}"
        LOGGER.info("Converting retained chunk %d with %d episodes", chunk_index, len(chunk_refs))
        convert_chunk(
            chunk_refs,
            config=config,
            repo_id=f"{args.repo_id}-chunk-{chunk_index:06d}",
            output_dir=chunk_dir,
            vcodec=args.vcodec,
            image_writer_threads=args.image_writer_threads,
            image_writer_processes=args.image_writer_processes,
            batch_encoding_size=args.batch_encoding_size,
        )


if __name__ == "__main__":
    main()
