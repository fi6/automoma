#!/usr/bin/env python3
# Copyright (c) 2024-2025, AutoMoMa Authors. All rights reserved.
# SPDX-License-Identifier: MIT
"""CLI entry-point for the AutoMoMa planning pipeline.

Uses OmegaConf to load the YAML config and allows command-line overrides::

    # Run with defaults
    python scripts/plan.py

    # Override from CLI
    python scripts/plan.py scene_name=scene_1_seed_1 object_id=11622

    # Use a custom config file
    python scripts/plan.py --config my_custom_plan.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

# Reduce cuRobo noise
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logging.getLogger("curobo").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    # ── Parse args ───────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="AutoMoMa planning pipeline",
        # Let unknown args pass through → OmegaConf overrides
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "plan.yaml"),
        help="Path to the planning config YAML (default: configs/plan.yaml)",
    )
    known, unknown = parser.parse_known_args()

    # ── Load + merge config ──────────────────────────────────────────────
    file_cfg = OmegaConf.load(known.config)
    cli_cfg = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(file_cfg, cli_cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # ── Run pipeline ─────────────────────────────────────────────────────
    from automoma.planning.pipeline import PlanningPipeline

    pipeline = PlanningPipeline(cfg_dict)
    out = pipeline.run(
        scene_name=cfg_dict["scene_name"],
        object_id=str(cfg_dict["object_id"]),
        mode=cfg_dict.get("mode", "train"),
    )

    if out:
        logger.info("Pipeline finished. Output: %s", out)
        return 0
    else:
        logger.error("Pipeline produced no output.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
