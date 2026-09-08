#!/usr/bin/env python3
"""Render and validate one synthetic RTX frame inside the AutoMoMa image."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/automoma-render-smoke"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    return parser.parse_args()


args = parse_args()

from isaacsim import SimulationApp


app = SimulationApp(
    {
        "headless": True,
        "renderer": "RayTracedLighting",
        "width": args.width,
        "height": args.height,
    }
)

import omni.replicator.core as rep
from PIL import Image, ImageStat


args.output_dir.mkdir(parents=True, exist_ok=True)
camera = rep.create.camera(position=(4.0, 4.0, 3.0), look_at=(0.0, 0.0, 0.0))
render_product = rep.create.render_product(camera, (args.width, args.height))
rep.create.cube(
    position=(0.0, 0.0, 0.0),
    scale=1.5,
    semantics=[("class", "render_smoke_cube")],
)
rep.create.plane(position=(0.0, 0.0, -0.8), scale=10.0)
rep.create.light(light_type="Distant", intensity=3000.0, rotation=(315.0, 0.0, 0.0))

writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=str(args.output_dir), rgb=True)
writer.attach([render_product])
rep.orchestrator.step(rt_subframes=8)
rep.orchestrator.wait_until_complete()
for _ in range(5):
    app.update()

frames = sorted(args.output_dir.glob("rgb_*.png"))
if len(frames) != 1:
    raise RuntimeError(f"expected exactly one RGB frame, found {len(frames)}")

image = Image.open(frames[0])
if image.size != (args.width, args.height):
    raise RuntimeError(f"unexpected image size: {image.size}")
standard_deviation = ImageStat.Stat(image.convert("RGB")).stddev
if max(standard_deviation) < 1.0:
    raise RuntimeError(f"rendered frame is effectively constant: stddev={standard_deviation}")

print(
    "RENDER_SMOKE_PASSED "
    f"frame={frames[0]} size={image.size} "
    f"rgb_stddev={[round(value, 2) for value in standard_deviation]}",
    flush=True,
)
app.close(wait_for_replicator=True)
