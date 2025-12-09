#!/usr/bin/env python
"""Debug script to inspect USD prim structure"""

from pathlib import Path
from pxr import Usd
import sys

def inspect_usd(usd_path):
    """Print all Xform prims in USD file"""
    print(f"Inspecting: {usd_path}\n")
    
    stage = Usd.Stage.Open(str(usd_path))
    
    print("All Xform prims in /World/scene:")
    print("=" * 80)
    
    world_scene = stage.GetPrimAtPath("/World/scene")
    if not world_scene:
        print("⚠️  No /World/scene prim found. Looking in /World:")
        world_scene = stage.GetPrimAtPath("/World")
    
    if world_scene:
        for prim in world_scene.Traverse():
            if prim.GetTypeName() == "Xform":
                prim_path = str(prim.GetPath())
                translate_attr = prim.GetAttribute("xformOp:translate")
                if translate_attr:
                    translate_val = translate_attr.Get()
                    print(f"{prim_path}")
                    print(f"  Translation: {translate_val}")
    
    print("\n" + "=" * 80)
    print("Search patterns to try:")
    print("=" * 80)
    print("- StaticCategoryFactory")
    print("- BowlFactory")
    print("- JarFactory")
    print("- WineglassFactory")
    print("- PotFactory")
    print("- Microwave")

if __name__ == "__main__":
    # Test with first scene
    usd_path = Path("/home/xinhai/Documents/automoma/assets/scene/infinigen/kitchen_1130/test/scene_0_seed_0/export/export_scene.blend/export_scene.usdc")
    
    if not usd_path.exists():
        print(f"❌ File not found: {usd_path}")
        sys.exit(1)
    
    inspect_usd(usd_path)
