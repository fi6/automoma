"""
Scene builder for loading scenes, objects, and robots into Isaac Sim.

IMPORTANT: SimulationApp must be initialized before importing this module.
Use automoma.simulation.sim_app_manager.get_simulation_app() first.
"""

from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import torch
import os
import logging


logger = logging.getLogger(__name__)

ISAAC_SIM_23 = False


# CuRobo imports are safe without SimulationApp
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util.logger import log_warn, log_info
from curobo.util.usd_helper import UsdHelper, set_prim_transform
from curobo.util_file import get_filename, get_path_of_dir, load_yaml
from curobo.types.state import JointState
from curobo.geom.types import WorldConfig, VoxelGrid
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

from automoma.utils.math_utils import pose_multiply
from automoma.utils.file_utils import get_abs_path

# Lazy imports for omni modules
_omni_imported = False


def _import_omni_modules():
    """Lazily import omni modules after SimulationApp is initialized."""
    global _omni_imported, ISAAC_SIM_23
    
    if _omni_imported:
        return
    
    # Check if SimulationApp is initialized
    from automoma.simulation.sim_app_manager import require_simulation_app
    require_simulation_app()
    
    # Now safe to import omni modules
    global add_reference_to_stage, XFormPrim, Robot, execute, _urdf, UsdPhysics
    
    from omni.isaac.core.utils.stage import add_reference_to_stage as _add_ref
    from omni.isaac.core.prims.xform_prim import XFormPrim as _XFormPrim
    from omni.isaac.core.robots import Robot as _Robot
    from omni.kit.commands import execute as _execute
    from pxr import UsdPhysics as _UsdPhysics
    
    # Try to import URDF interface (version detection)
    try:
        from omni.isaac.urdf import _urdf as __urdf  # isaacsim 2022.2
        ISAAC_SIM_23 = False
    except ImportError:
        from omni.importer.urdf import _urdf as __urdf  # isaac sim 2023.1+
        ISAAC_SIM_23 = True
    
    add_reference_to_stage = _add_ref
    XFormPrim = _XFormPrim
    Robot = _Robot
    execute = _execute
    _urdf = __urdf
    UsdPhysics = _UsdPhysics
    
    _omni_imported = True
    logger.debug(f"Omni modules for scene_builder imported successfully (Isaac Sim 2023: {ISAAC_SIM_23})")


def add_robot_to_scene(
    robot_config: Dict,
    my_world,
    load_from_usd: bool = False,
    subroot: str = "",
    robot_name: str = "robot",
    position: np.array = np.array([0, 0, 0]),
):
    """Helper function to add robot to scene - requires omni modules.
    
    This is adapted from helper.py in the cuakr codebase.
    """
    _import_omni_modules()
    
    # Load robot YAML config to get URDF path and other settings
    from curobo.util_file import get_assets_path, get_filename, get_path_of_dir, join_path
    
    urdf_interface = _urdf.acquire_urdf_interface()

    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.make_default_prim = False
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 20000
    import_config.default_position_drive_damping = 500
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0
    
    # Get URDF path from robot config
    asset_path = get_assets_path()
    if (
        "external_asset_path" in robot_config["kinematics"]
        and robot_config["kinematics"]["external_asset_path"] is not None
    ):
        asset_path = robot_config["kinematics"]["external_asset_path"]
    full_path = join_path(asset_path, robot_config["kinematics"]["urdf_path"])
    robot_path = get_path_of_dir(full_path)
    filename = get_filename(full_path)
    
    logger.info(f"Loading robot from {full_path} to {subroot}")
    
    imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)
    
    # Fix damping
    for _, joint in imported_robot.joints.items():
        joint.dynamics.damping *= 10
    
    dest_path = subroot
    robot_path = urdf_interface.import_robot(
        robot_path,
        filename,
        imported_robot,
        import_config,
        dest_path,
    )

    base_link_name = robot_config["kinematics"]["base_link"]

    robot_p = Robot(
        prim_path=robot_path + "/" + base_link_name,
        name=robot_name,
    )

    robot_prim = robot_p.prim
    stage = robot_prim.GetStage()
    linkp = stage.GetPrimAtPath(robot_path)
    set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])

    if False and ISAAC_SIM_23:  # this doesn't work in isaac sim 2023.1.1
        robot_p.set_solver_velocity_iteration_count(0)
        robot_p.set_solver_position_iteration_count(44)
        my_world._physics_context.set_solver_type("PGS")

    if ISAAC_SIM_23:  # fix to load robot correctly in isaac sim 2023.1.1
        linkp = stage.GetPrimAtPath(robot_path + "/" + base_link_name)
        mass = UsdPhysics.MassAPI(linkp)
        mass.GetMassAttr().Set(0)
    
    robot = my_world.scene.add(robot_p)
    
    return robot, robot_path

class SceneBuilder:
    def __init__(self, sim, cfg):
        self.sim = sim
        self.cfg = cfg    
    
class InfinigenBuilder(SceneBuilder):
    def __init__(self, sim, cfg=None):
        super().__init__(sim, cfg)
        # Import omni modules when builder is created
        _import_omni_modules()
        
    def init_root_pose(self, object_pose: torch.Tensor | List[float], scene_pose: torch.Tensor | List[float], type: str = "object_center") -> torch.Tensor:
        """
        Calculate the root pose of an object in the scene.

        Args:
            scene_pose (torch.Tensor | List[float]): The pose of the scene 7 dim. 
            object_pose (torch.Tensor | List[float]): The pose of the object 7 dim.
            type (str): Type of root pose calculation.

        Returns:
            torch.Tensor: The calculated root pose 7 dim. [x, y, z, qw, qx, qy, qz]
        """
        # Convert scene_pose to list
        if isinstance(scene_pose, torch.Tensor):
            scene_pose_list = scene_pose.tolist()
        elif isinstance(scene_pose, list):
            scene_pose_list = scene_pose
        else:
            scene_pose_list = list(scene_pose)
            
        # Convert object_pose to list
        if isinstance(object_pose, torch.Tensor):
            object_pose_list = object_pose.tolist()
        elif isinstance(object_pose, list):
            object_pose_list = object_pose
        else:
            object_pose_list = list(object_pose)
            
        self.root_pose = scene_pose_list
        
        if type == "object_center":
            object_Pose = Pose.from_list(object_pose_list)
            object_pose_inverse = object_Pose.inverse().to_list()
            
            object_pose_inverse[2] = 0.0  # ignore z offset for root pose
            self.root_pose = pose_multiply(object_pose_inverse, scene_pose_list)
        
        if not isinstance(self.root_pose, torch.Tensor):
            self.root_pose = torch.tensor(self.root_pose, device="cuda", dtype=torch.float32)
            
        return self.root_pose
    
    def get_world_pose(self, pose: torch.Tensor | List[float]) -> torch.Tensor:
        '''
        Calculate the world pose of an object given its local pose and the root pose.
        
        Args:
            pose (torch.Tensor | List[float]): The local pose of the object 7 dim.
        Returns:
            torch.Tensor: The world pose of the object 7 dim.
        '''
        if not isinstance(pose, torch.Tensor):
            pose = torch.tensor(pose, device="cuda", dtype=torch.float32)
        
        # Ensure root_pose is a tensor
        root_pose = self.root_pose
        if not isinstance(root_pose, torch.Tensor):
            root_pose = torch.tensor(root_pose, device="cuda", dtype=torch.float32)
            
        return pose_multiply(root_pose, pose)
    
    def load_scene(self, scene_cfg: Dict[str, Any], prim_path: str = "/World/Scene") -> None:
        '''
        Load a scene into the simulation.
        
        Args:
            scene_cfg (Dict[str, Any]): The scene configuration.
            prim_path (str): The path to the scene.
        '''
        path = scene_cfg.get("path", None)
        
        path = get_abs_path(path)  # scene usd need abs path
        
        # suppose scene is the relative world pose
        # pose = scene_cfg.get("pose", None)
        # if pose is not None:
        #     pose = self.get_world_pose(pose)
        pose = self.root_pose
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"USD file not found: {path}")
        
        print(f"Loading scene from {path} to {prim_path}")
        add_reference_to_stage(usd_path=path, prim_path=prim_path)
        
        if pose is not None:
            if isinstance(pose, torch.Tensor):
                pose = pose.tolist()
            elif not isinstance(pose, list):
                pose = list(pose)
            set_prim_transform(self.sim.world.stage.GetPrimAtPath(prim_path), pose=pose)
        
    def load_object(self, object_cfg: Dict[str, Any], prim_path: str = "/World/Object") -> None:
        '''
        Load an object into the simulation.
        
        Args:
            object_cfg (Dict[str, Any]): The object configuration.
            prim_path (str): The path to the object.
        '''
        path = object_cfg.get("path", None)
        pose = object_cfg.get("pose", None)
        
        if pose is not None:
            pose = self.get_world_pose(pose)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"USD file not found: {path}")
        
        print(f"Loading object from {path} to {prim_path}")
        
        root, sub_root = prim_path.rsplit('/', 1)
            
        self.sim.usd_helper.add_subroot(root=root, sub_root=sub_root, pose=pose)
        self._import_urdf_to_scene(urdf_full_path=path, subroot=prim_path, unique_name="target_object")
        
    def load_robot(self, robot_cfg: Dict[str, Any], prim_path: str = "/World/Robot", pose: Optional[List[float]] = None) -> Tuple:
        '''
        Load a robot into the simulation.
        
        Args:
            robot_cfg (Dict[str, Any]): The robot configuration.
            prim_path (str): The path to the robot.
            pose (List[float]): The pose of the robot [x, y, z, qw, qx, qy, qz].
        Returns:
            Tuple: (robot, robot_path) - The loaded robot prim and its USD path.
        '''
        if pose is None:
            pose = [0, 0, 0, 1, 0, 0, 0]
        pose = Pose.from_list(pose)
        
        logger.info(f"Loading robot to {prim_path} with pose {pose.to_list()}")
        
        # Add robot to scene using the proper implementation
        robot, robot_path = add_robot_to_scene(
            robot_config=robot_cfg,
            my_world=self.sim.world,
            load_from_usd=False,
            subroot=prim_path,
            robot_name="robot",
            position=pose.position[0].cpu().numpy(),
        )

        self.robot = robot
        self.robot_path = robot_path
        return robot, robot_path
        
    def _import_urdf_to_scene(
        self,
        urdf_full_path: str,
        subroot: str = "",
        position: List = [0, 0, 0, 1, 0, 0, 0],
        unique_name: Optional[str] = None,
    ):
        """Import a URDF file into the scene."""
        urdf_interface = _urdf.acquire_urdf_interface()
        import_config = _urdf.ImportConfig()
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = False
        import_config.fix_base = True
        import_config.make_default_prim = False
        import_config.self_collision = False
        import_config.create_physics_scene = True
        import_config.default_drive_strength = 10000
        import_config.default_position_drive_damping = 100
        import_config.default_drive_type = (
            _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        )
        import_config.distance_scale = 1
        import_config.density = 0.0

        robot_path = get_path_of_dir(urdf_full_path)
        filename = get_filename(urdf_full_path)
        imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)

        robot_prim_path = urdf_interface.import_robot(
            robot_path,
            filename,
            imported_robot,
            import_config,
            subroot,
        )
        robot_name = robot_prim_path.split("/")[-1]
        robot_prim_path_to = subroot + "/" + robot_name
        execute(
            "MovePrim",
            path_from=robot_prim_path,
            path_to=robot_prim_path_to,
            destructive=False,
            keep_world_transform=True,
            stage_or_context=self.sim.world.stage,
        )

        robot_p = Robot(
            prim_path=robot_prim_path_to,
            name=unique_name or f"robot_{id(imported_robot)}",
        )

        linkp = self.sim.world.stage.GetPrimAtPath(robot_prim_path_to)
        set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])
        obstacle = self.sim.world.scene.add(robot_p)
        return obstacle