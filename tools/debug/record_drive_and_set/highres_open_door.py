"""High-resolution Summit Franka open-door environment wrapper.

This module is loaded through IsaacLab-Arena's ``--environment`` extension hook.
It keeps the normal Summit Franka open-door semantics, but overrides camera
sensor sizes from CLI args so debug HDF5 recordings can carry report-quality
frames without patching ``third_party/``.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass

from isaaclab_arena.examples.example_environments.summit_franka_open_door_environment import (
    SummitFrankaOpenDoorEnvironment,
)


class HighResSummitFrankaOpenDoorEnvironment(SummitFrankaOpenDoorEnvironment):
    """Summit Franka open-door environment with CLI-configurable camera sizes."""

    name: str = "summit_franka_open_door_highres"

    def get_env(self, args_cli: argparse.Namespace):
        arena_env = super().get_env(args_cli)
        camera_cfg = getattr(arena_env.embodiment, "camera_config", None)
        if camera_cfg is not None:
            _set_camera_resolution(
                camera_cfg,
                width=int(getattr(args_cli, "camera_width", 1920)),
                height=int(getattr(args_cli, "camera_height", 1080)),
                data_types=_parse_csv(getattr(args_cli, "camera_data_types", "rgb")),
            )
        return arena_env

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        SummitFrankaOpenDoorEnvironment.add_cli_args(parser)
        parser.add_argument(
            "--camera_width",
            type=int,
            default=1920,
            help="Width for all Summit Franka camera sensors used by debug recording.",
        )
        parser.add_argument(
            "--camera_height",
            type=int,
            default=1080,
            help="Height for all Summit Franka camera sensors used by debug recording.",
        )
        parser.add_argument(
            "--camera_data_types",
            default="rgb",
            help="Comma-separated camera modalities to record. Defaults to rgb to avoid large depth HDF5 files.",
        )


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _set_camera_resolution(camera_cfg: object, *, width: int, height: int, data_types: list[str]) -> None:
    if width < 1 or height < 1:
        raise ValueError(f"Camera resolution must be positive, got {width}x{height}.")
    if not data_types:
        raise ValueError("--camera_data_types must contain at least one modality.")
    if not is_dataclass(camera_cfg):
        raise TypeError(f"Expected a dataclass camera config, got {type(camera_cfg)!r}.")

    for field in fields(camera_cfg):
        camera = getattr(camera_cfg, field.name)
        if all(hasattr(camera, attr) for attr in ("width", "height", "data_types")):
            camera.width = width
            camera.height = height
            camera.data_types = list(data_types)
