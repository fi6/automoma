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
    if threshold >= 200:
        delta = max(6, 255 - int(threshold))
        return np.max(np.abs(arr - background), axis=-1) > delta
    return np.any(arr < threshold, axis=-1)


def tint_layer(image: Image.Image, color: tuple[int, int, int], alpha: float, threshold: int) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = robot_mask(image, threshold)
    tinted = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    shaded = (0.35 * rgb.astype(np.float32) + 0.65 * np.asarray(color, dtype=np.float32)).clip(0, 255)
    tinted[..., :3] = shaded.astype(np.uint8)
    tinted[..., 3] = (mask.astype(np.float32) * 255.0 * alpha).astype(np.uint8)
    return Image.fromarray(tinted).convert("RGBA")


def composite_view(
    source_dir: Path,
    output_path: Path,
    alpha: float,
    workspace_alpha: float,
    threshold: int,
    max_frames: int,
) -> dict:
    scene_path = source_dir / "scene_objects.png"
    if not scene_path.exists():
        raise FileNotFoundError(f"Missing scene layer: {scene_path}")
    canvas = Image.open(scene_path).convert("RGBA")

    manifest = {"scene": str(scene_path), "episodes": {}}
    workspace_path = source_dir / "workspace_robots_only.png"
    if workspace_path.exists() and workspace_alpha > 0.0:
        workspace_color = (178, 184, 178)
        workspace_layer = tint_layer(Image.open(workspace_path), workspace_color, workspace_alpha, threshold)
        canvas.alpha_composite(workspace_layer)
        manifest["workspace"] = str(workspace_path)

    episodes = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_dir()
        and (
            path.name.startswith("episode")
            or path.name.startswith("mobile_traj")
            or path.name.startswith("fixed_traj")
        )
    )
    for episode_id, episode_dir in enumerate(episodes):
        color = DEFAULT_COLORS[episode_id % len(DEFAULT_COLORS)]
        frames = sorted(episode_dir.glob("*_robot_only.png"))
        if max_frames > 0:
            frames = frames[:max_frames]
        manifest["episodes"][episode_dir.name] = [str(path) for path in frames]
        for frame in frames:
            canvas.alpha_composite(tint_layer(Image.open(frame), color, alpha, threshold))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)
    manifest["output"] = str(output_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True, help="Panel directory, e.g. .../mobile_base")
    parser.add_argument("--views", nargs="+", default=["overview", "close"])
    parser.add_argument("--alpha", type=float, default=0.34)
    parser.add_argument("--workspace_alpha", type=float, default=0.10)
    parser.add_argument("--threshold", type=int, default=246)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_root / "composites"
    manifest = {}
    for view in args.views:
        source_dir = input_root / "sources" / view
        output_path = output_dir / f"{view}_composite.png"
        manifest[view] = composite_view(
            source_dir,
            output_path,
            args.alpha,
            args.workspace_alpha,
            args.threshold,
            args.max_frames,
        )
        print(f"[compose] wrote {output_path}")

    manifest_path = output_dir / "compose_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[compose] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
