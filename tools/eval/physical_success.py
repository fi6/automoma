"""Strict physical-success helpers shared by Microwave evaluation tools."""

from __future__ import annotations

import math


def classify_physical_success(
    final_openness_rad: float,
    final_handle_distance_m: float,
    *,
    openness_threshold_rad: float = 0.3,
    handle_distance_threshold_m: float = 0.1,
) -> tuple[bool, str | None]:
    """Match the production recorder's finite final-frame success contract."""

    if not math.isfinite(final_openness_rad) or not math.isfinite(final_handle_distance_m):
        return False, "metrics_error"
    opened = final_openness_rad >= openness_threshold_rad
    engaged = final_handle_distance_m <= handle_distance_threshold_m
    if opened and engaged:
        return True, None
    if not opened and not engaged:
        return False, "not_open_and_not_engaged"
    return (False, "not_open") if not opened else (False, "not_engaged")
