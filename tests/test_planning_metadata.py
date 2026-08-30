from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from automoma.core.types import TrajResult
from automoma.planning.io_utils import PlanningIO
from automoma.planning.metadata import (
    cat_planning_metadata,
    index_planning_metadata,
    make_planning_metadata,
    planning_metadata,
)


SCRIPT = Path(__file__).parents[1] / "tools/release/automoma-500k/plan_automoma_500k.py"


def load_release_planner():
    spec = importlib.util.spec_from_file_location("plan_automoma_500k", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def raw_result(values: list[float], success: list[bool]) -> TrajResult:
    base = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    states = torch.cat([base, base + 0.5], dim=1)
    return TrajResult(
        start_states=states,
        goal_states=states + 1,
        trajectories=torch.stack([states, states + 1], dim=1),
        success=torch.tensor(success),
    )


def metadata(values: list[float], *, scene: str = "scene_7_seed_7") -> dict:
    batches = [
        make_planning_metadata(
            1,
            grasp_id=index + 2,
            grasp_pose=[value, 0, 0, 1, 0, 0, 0],
            goal_angle=value,
            scene_id=scene,
            object_id="7221",
        )
        for index, value in enumerate(values)
    ]
    return cat_planning_metadata(batches)


def converted_payload(values: list[float], success: list[bool]) -> dict:
    values_tensor = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    count = len(values)
    payload = {
        "start_robot": values_tensor.repeat(1, 12),
        "start_obj": values_tensor.clone(),
        "goal_robot": (values_tensor + 1).repeat(1, 12),
        "goal_obj": values_tensor + 1,
        "traj_robot": values_tensor.reshape(count, 1, 1).repeat(1, 2, 12),
        "traj_obj": values_tensor.reshape(count, 1, 1).repeat(1, 2, 1),
        "traj_success": torch.tensor(success),
    }
    payload.update(metadata(values))
    return payload


def test_success_filter_and_limit_index_metadata_identically() -> None:
    planner = load_release_planner()
    filtered = planner.filter_successful(converted_payload([0.1, 0.2, 0.3, 0.4], [False, True, True, True]), 2)

    assert filtered["traj_success"].tolist() == [True, True]
    torch.testing.assert_close(filtered["goal_angle"], torch.tensor([0.2, 0.3]))
    assert filtered["grasp_id"].tolist() == [3, 4]
    torch.testing.assert_close(filtered["grasp_pose"][:, 0], torch.tensor([0.2, 0.3]))


def test_round_merge_preserves_metadata_order(tmp_path: Path) -> None:
    planner = load_release_planner()
    canonical = tmp_path / "canonical.pt"
    round_one = tmp_path / "round_one.pt"
    round_two = tmp_path / "round_two.pt"
    torch.save(converted_payload([0.1, 0.2], [True, False]), round_one)
    torch.save(converted_payload([0.3, 0.4], [False, True]), round_two)

    planner.merge_successes(canonical, round_one)
    planner.merge_successes(canonical, round_two)
    merged = torch.load(canonical, weights_only=False)

    torch.testing.assert_close(merged["goal_angle"], torch.tensor([0.1, 0.4]))
    torch.testing.assert_close(merged["traj_robot"][:, 0, 0], torch.tensor([0.1, 0.4]))
    assert merged["scene_id"] == "scene_7_seed_7"
    assert merged["object_id"] == "7221"


def test_raw_resume_append_preserves_metadata_alignment(tmp_path: Path) -> None:
    io = PlanningIO()
    path = tmp_path / "traj_data.pt"
    first = raw_result([0.1, 0.2], [True, False])
    second = raw_result([0.3], [True])

    io.save_traj_with_metadata(first, metadata([0.1, 0.2]), str(path))
    merged, merged_metadata = io.save_traj_with_metadata(second, metadata([0.3]), str(path))

    assert merged.success.tolist() == [True, False, True]
    torch.testing.assert_close(merged.start_states[:, 0], merged_metadata["goal_angle"])
    saved = torch.load(path, weights_only=False)
    torch.testing.assert_close(saved["start_states"][:, 0], saved["goal_angle"])


def test_resume_rejects_metadata_free_per_grasp_file(tmp_path: Path) -> None:
    io = PlanningIO()
    path = tmp_path / "traj_data.pt"
    result = raw_result([0.1], [True])
    torch.save(
        {
            "start_states": result.start_states,
            "goal_states": result.goal_states,
            "trajectories": result.trajectories,
            "success": result.success,
        },
        path,
    )

    with pytest.raises(ValueError, match="backfill"):
        io.save_traj_with_metadata(result, metadata([0.2]), str(path))


def test_metadata_index_helper_tracks_arbitrary_selection() -> None:
    source = metadata([0.1, 0.2, 0.3])
    selected = index_planning_metadata(source, torch.tensor([2, 0]))
    checked = planning_metadata(selected, 2)
    torch.testing.assert_close(checked["goal_angle"], torch.tensor([0.3, 0.1]))
