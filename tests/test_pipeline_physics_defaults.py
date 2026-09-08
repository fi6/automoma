from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_pipeline.sh"


def test_pipeline_exports_f4e200_by_default() -> None:
    env = os.environ.copy()
    for name in (
        "AUTOMOMA_ROBOT_OBJECT_STATIC_FRICTION",
        "AUTOMOMA_ROBOT_OBJECT_DYNAMIC_FRICTION",
        "AUTOMOMA_GRIPPER_EFFORT_LIMIT",
    ):
        env.pop(name, None)

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 1
    assert "Using AUTOMOMA_ROBOT_OBJECT_STATIC_FRICTION=4.0" in result.stdout
    assert "Using AUTOMOMA_ROBOT_OBJECT_DYNAMIC_FRICTION=4.0" in result.stdout
    assert "Using AUTOMOMA_GRIPPER_EFFORT_LIMIT=200.0" in result.stdout


def test_pipeline_preserves_explicit_physics_overrides() -> None:
    env = os.environ.copy()
    env.update(
        {
            "AUTOMOMA_ROBOT_OBJECT_STATIC_FRICTION": "3.0",
            "AUTOMOMA_ROBOT_OBJECT_DYNAMIC_FRICTION": "2.5",
            "AUTOMOMA_GRIPPER_EFFORT_LIMIT": "150.0",
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 1
    assert "Using AUTOMOMA_ROBOT_OBJECT_STATIC_FRICTION=3.0" in result.stdout
    assert "Using AUTOMOMA_ROBOT_OBJECT_DYNAMIC_FRICTION=2.5" in result.stdout
    assert "Using AUTOMOMA_GRIPPER_EFFORT_LIMIT=150.0" in result.stdout
