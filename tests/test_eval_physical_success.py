from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


EVAL_TOOLS = Path(__file__).parents[1] / "tools" / "eval"
sys.path.insert(0, str(EVAL_TOOLS))
from physical_success import classify_physical_success  # noqa: E402
sys.path.pop(0)


@pytest.mark.parametrize(
    ("openness", "distance", "success", "reason"),
    [
        (0.3, 0.1, True, None),
        (0.299, 0.1, False, "not_open"),
        (0.3, 0.101, False, "not_engaged"),
        (0.2, 0.2, False, "not_open_and_not_engaged"),
        (math.nan, 0.05, False, "metrics_error"),
        (0.4, math.inf, False, "metrics_error"),
    ],
)
def test_classify_physical_success_matches_production(
    openness: float,
    distance: float,
    success: bool,
    reason: str | None,
) -> None:
    assert classify_physical_success(openness, distance) == (success, reason)
