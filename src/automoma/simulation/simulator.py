"""
Isaac Sim Manager module.

This module provides the IsaacSimManager class which handles the Isaac Sim world
and USD operations. It uses lazy imports to avoid loading Isaac Sim modules
until SimulationApp is initialized.

IMPORTANT: SimulationApp must be initialized before importing this module.
Use automoma.simulation.sim_app_manager.get_simulation_app() first.
"""

from __future__ import annotations
import os
import sys
import logging
from typing import Dict, List, Any, Sequence, Optional, Union


logger = logging.getLogger(__name__)


# Lazy imports - these are populated when SimulationApp is available
_omni_imported = False


def _import_omni_modules():
    """Lazily import omni modules after SimulationApp is initialized."""
    global _omni_imported
    
    if _omni_imported:
        return
    
    # Check if SimulationApp is initialized
    from automoma.utils.sim_utils import require_simulation_app
    require_simulation_app()
    
    # Now safe to import omni modules
    global omni, World, add_reference_to_stage, XFormPrim, Robot, Camera, execute
    global UsdPhysics, Gf, UsdGeom
    global torch, np
    
    import omni.kit.actions.core
    import torch as _torch
    import numpy as _np
    
    from omni.isaac.core.utils.stage import add_reference_to_stage as _add_ref
    from omni.isaac.core import World as _World
    from omni.isaac.core.prims.xform_prim import XFormPrim as _XFormPrim
    from omni.isaac.core.robots import Robot as _Robot
    from omni.isaac.sensor import Camera as _Camera
    from omni.kit.commands import execute as _execute
    
    from pxr import UsdPhysics as _UsdPhysics, Gf as _Gf, UsdGeom as _UsdGeom
    import omni as _omni
    
    omni = _omni
    World = _World
    add_reference_to_stage = _add_ref
    XFormPrim = _XFormPrim
    Robot = _Robot
    Camera = _Camera
    execute = _execute
    UsdPhysics = _UsdPhysics
    Gf = _Gf
    UsdGeom = _UsdGeom
    torch = _torch
    np = _np
    
    _omni_imported = True
    logger.debug("Omni modules imported successfully")


# CuRobo imports are safe without SimulationApp
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util.logger import log_warn, log_info
from curobo.util.usd_helper import UsdHelper, set_prim_transform
from curobo.util_file import get_filename, get_path_of_dir, load_yaml
from curobo.types.state import JointState
from curobo.geom.types import WorldConfig, VoxelGrid
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig


class IsaacSimManager:
    def __init__(self, cfg):
        # Import omni modules lazily
        _import_omni_modules()
        
        # Now import the simulation_app reference
        from automoma.utils.sim_utils import get_simulation_app
        
        self.cfg = cfg
        self.world = None
        self.usd_helper = UsdHelper()
        self.simulation_app = get_simulation_app()
        self.tensor_args = TensorDeviceType()

        self.init_world()
        self.usd_helper.load_stage(self.world.stage)

    def init_world(self):
        self.world = World(stage_units_in_meters=1.0)
        xform = self.world.stage.DefinePrim("/World", "Xform")
        self.world.stage.SetDefaultPrim(xform)
        self.world.clear()

    def init_world_physics(self):
        self.world.initialize_physics()
        self.world.reset()
        
    def set_deactivate_prims(self, prim_paths: List[str] = []):
        """Deactivate prims by name pattern."""
        stage = self.world.stage
        for prim_path in prim_paths:
            for prim in stage.Traverse():
                prim_path_str = str(prim.GetPath())
                if prim_path in prim_path_str:
                    if prim.IsActive():
                        print(f"  ➤ Deactivating: {prim_path_str}")
                        prim.SetActive(False)
                    else:
                        print(f"  ➤ Already inactive: {prim_path_str}")

    def set_isaacsim_collision_free(self, prim_paths: List[str] = []):
        def disable_collision(prim_path):
            UsdPhysics.CollisionAPI.Get(self.world.stage, prim_path).GetCollisionEnabledAttr().Set(False)

        for prim_path in prim_paths:
            disable_collision(prim_path)

        # for prim_path in [
        #     "/World/summit_panda/panda_leftfinger/collisions",
        #     "/World/summit_panda/panda_rightfinger/collisions",
        #     "/World/summit_panda/grasp_frame/collisions",
        #     "/World/summit_panda/panda_hand/collisions",
        #     # "/World/object/partnet_5b2633d960419bb2e5bf1ab8e7d0b/link_0/collisions",
        #     # "/World/object/partnet_5b2633d960419bb2e5bf1ab8e7d0b/link_1/collisions"
        # ]:

    def set_lighting(self, mode: int = 2):
        action_registry = omni.kit.actions.core.get_action_registry()
        action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_rig")
        action.execute(lighting_mode=mode)
        
    def step(self, step=1, render=True):
        """Step the Isaac Sim world"""
        if step == -1:
            # If step is -1, run until the simulation app is running
            while self.simulation_app.is_running():
                self.world.step(render=render)
        for _ in range(step):
            self.world.step(render=render)


    def get_prim_pose(self, prim_path: str) -> Optional[np.ndarray]:
        """Get the world pose of a prim (object) in the scene."""
        from omni.usd import get_world_transform_matrix
        prim = self.world.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"prim {prim_path} not found")
            return None
        return np.array(get_world_transform_matrix(prim)).T
   
    def close(self):
        """Close the Isaac Sim world."""
        self.simulation_app.close()
        logger.info("Isaac Sim application closed.")