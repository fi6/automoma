#!/usr/bin/env python3
"""Convert poster green-screen PNG layers to transparent-background PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def has_green_screen(arr: np.ndarray, min_fraction: float) -> bool:
    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3] if arr.shape[-1] == 4 else np.full(rgb.shape[:2], 255, dtype=np.uint8)
    visible = alpha > 0
    if not np.any(visible):
        return False

    green = (rgb[..., 1] > 180) & (rgb[..., 0] < 120) & (rgb[..., 2] < 120) & visible
    if float(green.sum()) / float(visible.sum()) >= min_fraction:
        return True

    border = np.zeros(visible.shape, dtype=bool)
    border[:16, :] = True
    border[-16:, :] = True
    border[:, :16] = True
    border[:, -16:] = True
    border_visible = border & visible
    if not np.any(border_visible):
        return False
    return float((green & border).sum()) / float(border_visible.sum()) >= 0.20


def key_green_screen(path: Path, min_fraction: float, dry_run: bool) -> bool:
    image = Image.open(path).convert("RGBA")
    arr = np.asarray(image, dtype=np.uint8).copy()
    if not has_green_screen(arr, min_fraction):
        return False

    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3].astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_rb = np.maximum(r, b)
    green_excess = g - max_rb

    # Smooth matte: exact green becomes transparent, anti-aliased green edges
    # become partially transparent, and de-spill removes the remaining halo.
    key_candidate = (g > 100.0) & (green_excess > 8.0)
    matte = np.ones_like(alpha, dtype=np.float32)
    matte[key_candidate] = np.clip((70.0 - green_excess[key_candidate]) / 62.0, 0.0, 1.0)
    pure_green = (g > 220.0) & (r < 60.0) & (b < 60.0)
    matte[pure_green] = 0.0

    arr[..., 3] = np.round(alpha * matte).astype(np.uint8)
    despill = key_candidate & (arr[..., 3] > 0)
    arr[..., 1][despill] = np.maximum(arr[..., 0][despill], arr[..., 2][despill])

    if not dry_run:
        Image.fromarray(arr).save(path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="PNG file or directory roots to process.")
    parser.add_argument("--min_fraction", type=float, default=0.05)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    candidates: list[Path] = []
    for root_arg in args.roots:
        root = Path(root_arg).expanduser().resolve()
        if root.is_file() and root.suffix.lower() == ".png":
            candidates.append(root)
        elif root.is_dir():
            candidates.extend(sorted(root.rglob("*.png")))

    processed = []
    for path in candidates:
        if key_green_screen(path, args.min_fraction, args.dry_run):
            processed.append(path)
            print(f"[key] {'would write' if args.dry_run else 'wrote'} {path}")

    print(f"[key] processed {len(processed)} / {len(candidates)} PNG files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
