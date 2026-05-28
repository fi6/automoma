#!/usr/bin/env python3
"""Fast import/version check used at image build time and for runtime smoke checks."""

from __future__ import annotations

import importlib
import importlib.util
import importlib.metadata as metadata
import sys
from pathlib import Path

PACKAGES = [
    "torch",
    "torchvision",
    "warp",
    "numpy",
    "automoma",
    "curobo",
    "isaaclab",
    "isaaclab_arena",
    "lerobot",
    "datasets",
    "av",
    "h5py",
]

DIST_NAMES = {
    "curobo": "nvidia_curobo",
    "isaaclab_arena": "isaaclab_arena",
    "warp": "warp-lang",
}


def version_for(module_name: str) -> str:
    dist_name = DIST_NAMES.get(module_name, module_name)
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def main() -> int:
    print(f"python={sys.executable}")
    for name in PACKAGES:
        module = importlib.import_module(name)
        print(f"{name}={version_for(name)} file={getattr(module, '__file__', '')}")

    import torch

    print(f"torch_cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
    import warp

    print(f"warp_torch_bridge={hasattr(warp, 'torch')}")

    import isaacsim

    lighting_ext = (
        Path(isaacsim.__file__).resolve().parent
        / "extscache"
        / "omni.kit.viewport.menubar.lighting-107.3.1+107.3.0"
    )
    lighting_usd = lighting_ext / "data" / "usd" / "Grey_Studio.usda"
    if str(lighting_ext) not in sys.path:
        sys.path.append(str(lighting_ext))
    lighting_spec = importlib.util.find_spec("omni.kit.viewport.menubar.lighting")
    print(f"lighting_ext={lighting_ext} exists={lighting_ext.exists()}")
    print(f"grey_studio_usd={lighting_usd} exists={lighting_usd.exists()}")
    print(f"lighting_import_spec={lighting_spec is not None}")
    if not lighting_ext.exists() or not lighting_usd.exists() or lighting_spec is None:
        raise RuntimeError("Isaac Sim viewport lighting extension is not available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
