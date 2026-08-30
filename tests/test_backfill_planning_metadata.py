from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "tools/dataset/backfill_planning_metadata.py"


def module():
    spec = importlib.util.spec_from_file_location("backfill_planning_metadata", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loaded)
    return loaded


def raw(values: list[float]) -> dict:
    state = torch.zeros(len(values), 11)
    state[:, 0] = torch.tensor(values)
    state[:, -1] = -torch.tensor(values)
    trajectory = torch.stack([state, state + torch.tensor([0] * 10 + [0.1])], dim=1)
    return {
        "start_states": state,
        "goal_states": state,
        "trajectories": trajectory,
        "success": torch.ones(len(values), dtype=torch.bool),
    }


def test_backfill_uses_exact_matches_not_canonical_order(tmp_path: Path) -> None:
    tool = module()
    raw_payload = raw([0.2, 0.4])
    converted = tool.convert_raw(raw_payload, prepend_steps=4, gripper_open=0.04, gripper_closed=0.0)
    canonical = {key: value[torch.tensor([1, 0])] for key, value in converted.items()}
    path = tmp_path / "traj_data_train.pt"
    torch.save(canonical, path)
    candidates = []
    for index in range(2):
        candidates.append(
            {
                "source": f"grasp_000{index}/traj_data.pt:{index}",
                **{key: converted[key][index] for key in tool.TRAJ_KEYS},
                "grasp_id": torch.tensor(index + 2),
                "grasp_pose": torch.tensor([float(index), 0, 0, 1, 0, 0, 0]),
                "goal_angle": torch.tensor([0.2, 0.4][index]),
                "scene_id": "scene_0_seed_0",
                "object_id": "7221",
            }
        )

    result = tool.backfill(path, candidates)
    assert result["grasp_id"].tolist() == [3, 2]
    torch.testing.assert_close(result["goal_angle"], torch.tensor([0.4, 0.2]))


def test_backfill_stops_on_ambiguous_match(tmp_path: Path) -> None:
    tool = module()
    converted = tool.convert_raw(raw([0.2]), prepend_steps=4, gripper_open=0.04, gripper_closed=0.0)
    path = tmp_path / "traj_data_train.pt"
    torch.save(converted, path)
    candidate = {
        "source": "one",
        **{key: converted[key][0] for key in tool.TRAJ_KEYS},
        "grasp_id": torch.tensor(2),
        "grasp_pose": torch.zeros(7),
        "goal_angle": torch.tensor(0.2),
        "scene_id": "scene",
        "object_id": "7221",
    }
    duplicate = dict(candidate, source="two")

    with pytest.raises(ValueError, match="2 exact"):
        tool.backfill(path, [candidate, duplicate])
