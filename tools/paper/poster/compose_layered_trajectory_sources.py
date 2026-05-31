#!/usr/bin/env python3
"""Composite poster trajectory layers exported by render_actual_ghost_comparison.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_COLORS = [
    (0, 140, 133),
    (26, 115, 199),
    (235, 122, 20),
    (166, 87, 184),
    (61, 158, 71),
]


def robot_mask(image: Image.Image, threshold: int) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    # Isaac's headless renderer uses a near-white, but not exactly 255, studio
    # background.  Estimate it from the border so we do not tint the whole image.
    border = np.concatenate(
        [
            arr[:12].reshape(-1, 3),
            arr[-12:].reshape(-1, 3),
            arr[:, :12].reshape(-1, 3),
            arr[:, -12:].reshape(-1, 3),
        ]
    )
    background = np.median(border, axis=0)
    if background[1] > 220 and background[0] < 40 and background[2] < 40:
        green_screen = (
            (arr[..., 1] > 120)
            & (arr[..., 1] > arr[..., 0] * 1.25)
            & (arr[..., 1] > arr[..., 2] * 1.25)
        )
        return ~green_screen
    if threshold >= 200:
        delta = max(6, 255 - int(threshold))
        return np.max(np.abs(arr - background), axis=-1) > delta
    return np.any(arr < threshold, axis=-1)


def alpha_layer(image: Image.Image, alpha: float, threshold: int) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = robot_mask(image, threshold)
    layer = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    layer[..., :3] = rgb
    layer[..., 3] = (mask.astype(np.float32) * 255.0 * alpha).astype(np.uint8)
    return Image.fromarray(layer).convert("RGBA")


def tint_layer(image: Image.Image, color: tuple[int, int, int], alpha: float, threshold: int) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = robot_mask(image, threshold)
    layer = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    shaded = (0.35 * rgb.astype(np.float32) + 0.65 * np.asarray(color, dtype=np.float32)).clip(0, 255)
    layer[..., :3] = shaded.astype(np.uint8)
    layer[..., 3] = (mask.astype(np.float32) * 255.0 * alpha).astype(np.uint8)
    return Image.fromarray(layer).convert("RGBA")


def episode_dirs(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_dir()
        and (
            path.name.startswith("episode")
            or path.name.startswith("mobile_traj")
            or path.name.startswith("fixed_traj")
        )
    )


def selected_episode_dirs(source_dir: Path, max_episodes: int) -> list[Path]:
    episodes = episode_dirs(source_dir)
    if max_episodes > 0:
        return episodes[:max_episodes]
    return episodes


def selected_frames(episode_dir: Path, max_frames: int) -> list[Path]:
    frames = sorted(episode_dir.glob("*_robot_only.png"))
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


def composite_view(source_dir: Path, output_dir: Path, args: argparse.Namespace, view: str) -> dict:
    scene_path = source_dir / "scene_objects.png"
    if not scene_path.exists():
        raise FileNotFoundError(f"Missing scene layer: {scene_path}")

    manifest = {"scene": str(scene_path), "episodes": {}, "outputs": {}}
    all_canvas = Image.open(scene_path).convert("RGBA")
    ghost_canvas = Image.open(scene_path).convert("RGBA")

    workspace_path = source_dir / "workspace_robots_only.png"
    if workspace_path.exists() and args.workspace_alpha > 0.0:
        workspace_layer = alpha_layer(Image.open(workspace_path), args.workspace_alpha, args.threshold)
        all_canvas.alpha_composite(workspace_layer)
        ghost_canvas.alpha_composite(workspace_layer)
        manifest["workspace"] = str(workspace_path)

    episodes = selected_episode_dirs(source_dir, args.max_episodes)
    for episode_id, episode_dir in enumerate(episodes):
        color = DEFAULT_COLORS[episode_id % len(DEFAULT_COLORS)]
        frames = selected_frames(episode_dir, args.max_frames)
        manifest["episodes"][episode_dir.name] = [str(path) for path in frames]
        for frame in frames:
            image = Image.open(frame)
            all_canvas.alpha_composite(alpha_layer(image, args.raw_alpha, args.threshold))
            ghost_canvas.alpha_composite(tint_layer(image, color, args.alpha, args.threshold))

    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / f"{view}_all.png"
    ghost_path = output_dir / f"{view}_all_ghost.png"
    all_canvas.convert("RGB").save(all_path)
    ghost_canvas.convert("RGB").save(ghost_path)
    manifest["outputs"]["all"] = str(all_path)
    manifest["outputs"]["ghost"] = str(ghost_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True, help="Panel directory, e.g. .../mobile_base")
    parser.add_argument("--views", nargs="+", default=["overview", "close"])
    parser.add_argument("--raw_alpha", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha for colored ghost composites.")
    parser.add_argument("--workspace_alpha", type=float, default=0.80)
    parser.add_argument("--threshold", type=int, default=246)
    parser.add_argument("--max_episodes", type=int, default=3, help="Maximum episode layers to composite; 0 uses all.")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_root / "composites"
    manifest = {}
    for view in args.views:
        source_dir = input_root / "sources" / view
        manifest[view] = composite_view(source_dir, output_dir, args, view)
        print(f"[compose] wrote {manifest[view]['outputs']['all']}")
        print(f"[compose] wrote {manifest[view]['outputs']['ghost']}")

    manifest_path = output_dir / "compose_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[compose] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
