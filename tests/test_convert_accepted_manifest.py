from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np


TOOLS = Path(__file__).parents[1] / "tools/dataset"
SCRIPT = TOOLS / "convert_accepted_hdf5_to_lerobot_v30.py"


def module():
    sys.path.insert(0, str(TOOLS))
    try:
        spec = importlib.util.spec_from_file_location("convert_accepted", SCRIPT)
        loaded = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = loaded
        spec.loader.exec_module(loaded)
        return loaded
    finally:
        sys.path.remove(str(TOOLS))


def write_job(path: Path, episodes: list[int], scene: str) -> None:
    path.mkdir(parents=True)
    with h5py.File(path / "record.hdf5", "w") as root:
        data = root.create_group("data")
        for demo_index, traj_index in enumerate(reversed(episodes)):
            demo = data.create_group(f"demo_{demo_index}")
            demo.attrs["traj_index"] = traj_index
            demo.attrs["grasp_id"] = traj_index + 2
            demo.attrs["grasp_pose"] = np.arange(7, dtype=np.float32)
            demo.attrs["goal_angle"] = np.float32(1.0)
            demo.attrs["scene_id"] = scene
            demo.attrs["object_id"] = "7221"


def test_manifest_order_drives_cross_job_cross_scene_chunks_and_partial_finalize(tmp_path: Path) -> None:
    tool = module()
    first = tmp_path / "accepted/job_a"
    second = tmp_path / "accepted/job_b"
    write_job(first, [4, 1], "scene_a")
    write_job(second, [8, 3], "scene_b")
    manifest = tmp_path / "accepted.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"job_id": "a", "episodes": [1, 4], "path": str(first)}),
                json.dumps({"job_id": "b", "episodes": [3, 8], "path": str(second)}),
            ]
        )
        + "\n"
    )

    refs = tool.accepted_episode_refs(manifest)
    assert [(ref.job_id, ref.metadata["traj_index"], ref.metadata["scene_id"]) for ref in refs] == [
        ("a", 1, "scene_a"),
        ("a", 4, "scene_a"),
        ("b", 3, "scene_b"),
        ("b", 8, "scene_b"),
    ]
    grouped = list(tool.chunks(refs, 3))
    assert [len(chunk) for chunk in grouped] == [3, 1]


def test_convert_chunk_writes_planning_metadata_once_and_finalizes(tmp_path: Path, monkeypatch) -> None:
    tool = module()
    hdf5_path = tmp_path / "record.hdf5"
    with h5py.File(hdf5_path, "w") as root:
        root.create_group("data").create_group("demo_0")
    ref = tool.AcceptedEpisodeRef(
        hdf5_path=hdf5_path,
        demo_key="demo_0",
        job_id="job-a",
        metadata={
            "traj_index": 7,
            "grasp_id": 3,
            "grasp_pose": [1, 2, 3, 1, 0, 0, 0],
            "goal_angle": 1.23,
            "scene_id": "scene_a",
            "object_id": "7221",
        },
    )

    class Meta:
        def __init__(self):
            self.rows = []

        def save_episode(self, _index, _length, _tasks, _stats, metadata):
            self.rows.append(dict(metadata))

    class Dataset:
        def __init__(self):
            self.meta = Meta()
            self.frames = []
            self.finalized = False

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.meta.save_episode(0, len(self.frames), ["open"], {}, {"data/chunk_index": 0})

        def finalize(self):
            self.finalized = True

    dataset = Dataset()

    class Factory:
        @staticmethod
        def create(**_kwargs):
            return dataset

    monkeypatch.setattr(
        tool,
        "_load_converter_dependencies",
        lambda: {
            "ensure_output_dir": lambda path: path.mkdir(parents=True),
            "prepare_episode_data": lambda _trajectory, _config: (
                {"state": np.asarray([[1.0], [2.0]], dtype=np.float32)},
                {"camera": np.zeros((2, 2, 2, 3), dtype=np.uint8)},
            ),
            "infer_features": lambda *_args: {},
            "LeRobotDataset": Factory,
        },
    )
    config = type("Config", (), {"fps": 30, "robot_type": "summit", "language_instruction": "open"})()

    tool.convert_chunk(
        [ref],
        config=config,
        repo_id="repo",
        output_dir=tmp_path / "chunk",
        vcodec="h264",
        image_writer_threads=0,
        image_writer_processes=0,
        batch_encoding_size=1,
    )

    assert dataset.finalized is True
    assert dataset.meta.rows[0]["traj_index"] == 7
    assert dataset.meta.rows[0]["grasp_pose"] == [1, 2, 3, 1, 0, 0, 0]
    assert all("traj_index" not in frame for frame in dataset.frames)
