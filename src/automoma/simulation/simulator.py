import os
import sys

# Set up Isaac Sim before importing other modules
import isaacsim
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": 1920, "height": 1080})

import omni.kit.actions.core


import torch
import numpy as np
import h5py
from typing import Dict, List, Any, Sequence, Optional, Union
from tqdm import tqdm
from pathlib import Path

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.usd import get_world_transform_matrix, get_context
from omni.isaac.core import World
from omni.isaac.core.objects import sphere
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.importer.urdf import _urdf
from omni.isaac.core.robots import Robot
from omni.kit.commands import execute
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.torch.rotations as rot_utils

from pxr import Gf, UsdGeom, UsdPhysics, UsdLux, UsdShade, Usd, Sdf

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
        self.cfg = cfg
        self.world = None
        self.usd_helper = UsdHelper()
        self.simulation_app = simulation_app
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

    def step(self, step=5, render=True):
        """Step the Isaac Sim world"""
        if step == -1:
            # If step is -1, run until the simulation app is running
            while self.simulation_app.is_running():
                self.world.step(render=render)
        for _ in range(step):
            self.world.step(render=render)
