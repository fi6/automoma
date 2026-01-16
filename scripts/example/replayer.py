"""
Standalone Replayer - visualizes IK and trajectory results in Isaac Sim.
Replays single IK, trajectories, and filtered trajectories for object 7221, grasp 0.
"""

import os
import sys

# Set up Isaac Sim before importing other modules
import isaacsim
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width": 1920,
    "height": 1080
})

import omni.kit.actions.core


import torch
import numpy as np
import h5py
sys.path.append(os.path.join(os.path.dirname(__file__), "../third_party/lerobot/src"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
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

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util.logger import log_warn, log_info
from curobo.util.usd_helper import UsdHelper, set_prim_transform
from curobo.util_file import get_filename, get_path_of_dir, load_yaml
from curobo.types.state import JointState
from curobo.geom.types import WorldConfig, VoxelGrid
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

from cuakr.utils.helper import add_robot_to_scene
from automoma.utils.math_utils import pose_multiply
from cuakr.utils.camera import create_colored_pointcloud, process_point_cloud, PCDProcConfig

from automoma.models.object import ObjectDescription
from automoma.models.robot import RobotDescription
from automoma.utils.data_structures import CameraResult, TrajectoryEvaluationResult
from automoma.utils.transform import euler_to_quat, single_axis_self_rotation, matrix_to_pose

# ============================================================================
# CONFIGURATION MAPPING - Object IDs and Paths
# ============================================================================
OBJECT_CONFIG_MAP = {
    "7221": {
        "asset_type": "Microwave",
        "asset_id": "7221",
        "scale": 0.3562990018302636,
        "urdf_path": "assets/object/Microwave/7221/7221_0_scaling.urdf",
        "handle_link": "link_0",
    },
    "11622": {
        "asset_type": "Dishwasher",
        "asset_id": "11622",
        "scale": 0.6446,
        "urdf_path": "assets/object/Dishwasher/11622/11622_0_scaling.urdf",
        "handle_link": "link_0",
    },
    "103634": {
        "asset_type": "TrashCan",
        "asset_id": "103634",
        "scale": 0.48385408192053975,
        "urdf_path": "assets/object/TrashCan/103634/103634_0_scaling.urdf",
        "handle_link": "link_0",
    },
    "46197": {
        "asset_type": "Cabinet",
        "asset_id": "46197",
        "scale": 0.5113198146209817,
        "urdf_path": "assets/object/Cabinet/46197/46197_0_scaling.urdf",
        "handle_link": "link_0",
    },
    "10944": {
        "asset_type": "Refrigerator",
        "asset_id": "10944",
        "scale": 0.900,
        "urdf_path": "assets/object/Refrigerator/10944/10944_0_scaling.urdf",
        "handle_link": "link_0",
    },
    "101773": {
        "asset_type": "Oven",
        "asset_id": "101773",
        "scale": 0.7231779244463762,
        "urdf_path": "assets/object/Oven/101773/101773_0_scaling.urdf",
        "handle_link": "link_0",
    },
}

OBJECT_PRIM_MAP = {
    "7221": "StaticCategoryFactory_Microwave_7221",
    "11622": "StaticCategoryFactory_Dishwasher_11622",
    "103634": "StaticCategoryFactory_TrashCan_103634",
    "46197": "StaticCategoryFactory_Cabinet_46197",
    "10944": "StaticCategoryFactory_Refrigerator_10944",
    "101773": "StaticCategoryFactory_Oven_101773",
}

class Replayer:
    def __init__(self, simulation_app, robot_cfg, scene_cfg, object_cfg):
        self._init_world()
        
        self.simulation_app = simulation_app
        self.robot_cfg = robot_cfg
        self.scene_cfg = scene_cfg
        self.object_cfg = object_cfg
        self.usd_helper = UsdHelper()
        self.usd_helper.load_stage(self.isaac_world.stage)
        
        self.scene_bounds = {
            'x': [-5.0, 5.0],
            'y': [-5.0, 5.0],
            'z': [-5.0, 15.0]
        }
        
        self.root_pose = [0, 0, 0, 1, 0, 0, 0]
        self._init_root_pose()
        
        self._load_robot(self.robot_cfg)
        self._load_scene(self.scene_cfg)
        self._load_object(self.object_cfg)

        self.set_isaacsim_collision_free()

        self.isaac_world.initialize_physics()
        self.isaac_world.reset()
        self.tensor_args = TensorDeviceType()
    
    def compute_bounds(self, prim_path: str) -> None:
        """
        Set custom scene bounding box for camera positioning.
        """
        min_point_gf, max_point_gf = get_context().compute_path_world_bounding_box(prim_path)
        min_arr = np.array(min_point_gf)
        max_arr = np.array(max_point_gf)
        
        bounding_box = np.concatenate((min_arr, max_arr))
        
        return bounding_box
    def _init_root_pose(self) -> None:
        """Initialize the world  pose"""
        scene_offset = self.scene_cfg["pose"]
        object_pose = self.object_cfg["pose"].copy()
        object_pose_Pose = Pose.from_list(object_pose)
        object_pose_inverse = object_pose_Pose.inverse().to_list()

        object_pose_inverse[2] = 0.0  # ignore z offset for root pose
        self.root_pose = pose_multiply(object_pose_inverse, scene_offset)

    def get_world_pose(self, pose: List[float]) -> List[float]:
        """Get the new world pose"""
        return pose_multiply(self.root_pose, pose)

    def get_prim_pose(self, prim_path: str) -> Optional[np.ndarray]:
        """Get the world pose of a prim (object) in the scene."""
        prim = self.isaac_world.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            log_warn(f"prim {prim_path} not found")
            return None
        return np.array(get_world_transform_matrix(prim)).T

    def _init_world(self):
        self.isaac_world = World(stage_units_in_meters=1.0)
        xform = self.isaac_world.stage.DefinePrim("/World", "Xform")
        self.isaac_world.stage.SetDefaultPrim(xform)
        self.isaac_world.clear()
        
    def set_deactivate_prims(self, prim_name: str) -> None:
        """Deactivate prims by name pattern."""
        stage = self.isaac_world.stage
        for prim in stage.Traverse():
            prim_path_str = str(prim.GetPath())
            if prim_name in prim_path_str:
                if prim.IsActive():
                    print(f"  ➤ Deactivating: {prim_path_str}")
                    prim.SetActive(False)
                else:
                    print(f"  ➤ Already inactive: {prim_path_str}")

    def _load_scene(self, scene_cfg: dict, prim_path: str = "/World/scene") -> None:
        """Load a scene from a USD file."""
        usd_path = scene_cfg["path"]
        if not os.path.exists(usd_path):
            raise FileNotFoundError(f"USD file not found: {usd_path}")
            
        log_info(f"Loading USD scene: {usd_path}")
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        set_prim_transform(self.isaac_world.stage.GetPrimAtPath(prim_path), pose=self.root_pose)
        self.compute_bounds(prim_path)

    def _load_robot(self, robot_cfg: Dict, robot_type: str = "robot", pose: Optional[List[float]] = None) -> XFormPrim:
        """Load a robot into the scene."""
        if pose is None:
            pose = [0, 0, 0, 1, 0, 0, 0]
        pose = Pose.from_list(pose)
            
        # Determine subroot and robot name based on type
        subroot = "/World/robot/"
        if robot_type == "akr_robot":
            robot_name = "robot_akr"
        else:
            robot_name = "robot"

        log_info(f"Loading {robot_type} at position {pose}")
        
        # Add robot to scene
        robot_prim = add_robot_to_scene(
            robot_cfg,
            self.isaac_world,
            subroot=subroot,
            robot_name=robot_name,
            position=pose.position[0].cpu().numpy(),
        )

        self.robot = robot_prim[0]
        return self.robot

    def _load_object(self, object_cfg):
        """Load an object into the scene."""
        if object_cfg is None:
            log_warn("No object configuration provided.")
            return
        
        object_path = object_cfg["path"]
        object_pose = self.get_world_pose(object_cfg["pose"])
        
        print(f"Loading object from {object_path} at pose {object_pose}")

        log_info(f"Loading object at pose {object_pose}")
        
        self.usd_helper.add_subroot(
            "/World",
            "object",
            Pose.from_list(object_pose)
        )

        self._import_urdf_to_scene(
            urdf_full_path=object_path,
            subroot="/World/object",
            unique_name="target_object",
        )

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
            stage_or_context=self.isaac_world.stage,
        )

        robot_p = Robot(
            prim_path=robot_prim_path_to,
            name=unique_name or f"robot_{id(imported_robot)}",
        )

        linkp = self.isaac_world.stage.GetPrimAtPath(robot_prim_path_to)
        set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])
        obstacle = self.isaac_world.scene.add(robot_p)
        return obstacle
    
    def _adjust_pose_for_robot(self, robot_pose, robot_name, gripper_joint_value=0.02, *args):
        """Adjust the robot pose tensor based on the robot model."""
        if robot_name == "summit_franka":
            return torch.cat([
                robot_pose,
                torch.tensor([gripper_joint_value, gripper_joint_value], device=robot_pose.device),
            ])
        elif robot_name == "tiago":
            return torch.cat([
                robot_pose,
                torch.tensor([gripper_joint_value, gripper_joint_value], device=robot_pose.device),
            ])
        elif robot_name == "r1":
            return torch.cat([
                robot_pose[0:3],
                torch.tensor([2.1816, -2.6178, -0.4363, 0.0], device=robot_pose.device),
                robot_pose[3:],
                torch.tensor([gripper_joint_value, gripper_joint_value], device=robot_pose.device),
            ])
        elif robot_name == "summit_franka_fixed_base":
            return torch.cat([
                torch.tensor([-0.9, 1.4, 0.0], device=robot_pose.device),
                robot_pose,
                torch.tensor([gripper_joint_value, gripper_joint_value], device=robot_pose.device),
            ])
        return robot_pose
    
    def _adjust_pose_for_akr_robot(self, robot_pose, robot_name, gripper_joint_value=0.02, *args):
        """Adjust the robot pose tensor for AKR robot."""
        if robot_name == "summit_franka":
            return torch.cat([
                robot_pose[:-1],
                torch.tensor([gripper_joint_value, gripper_joint_value], device=robot_pose.device),
                robot_pose[-1:],
            ])
        elif robot_name == "tiago":
            return torch.cat([
                robot_pose[:-1],
                torch.tensor([gripper_joint_value, gripper_joint_value], device=robot_pose.device),
                robot_pose[-1:],
            ])
        elif robot_name == "r1":
            return torch.cat([
                robot_pose[0:3],
                torch.tensor([2.1816, -2.6178, -0.4363, 0.0], device=robot_pose.device),
                robot_pose[3:-1],
                torch.tensor([gripper_joint_value, gripper_joint_value], device=robot_pose.device),
                robot_pose[-1:],
            ])
        elif robot_name == "summit_franka_fixed_base":
            return torch.cat([
                torch.tensor([-0.9, 1.4, 0.0], device=robot_pose.device),
                robot_pose[:-1],
                torch.tensor([gripper_joint_value, gripper_joint_value], device=robot_pose.device),
                robot_pose[-1:],
            ])
        return robot_pose

    def _isaacsim_step(self, step=10, duration=None, render=True):
        """Step the Isaac Sim world"""
        if duration is not None:
            start_time = time.time()
            while time.time() - start_time < duration:
                self.isaac_world.step(render=render)
        elif step == -1:
            while self.simulation_app.is_running():
                self.isaac_world.step(render=render)
        else:
            for _ in range(step):
                self.isaac_world.step(render=render)

    def set_handle_pose(self, joint_angle):
        """Set the joint angle of the articulated object"""
        joint_angle = abs(joint_angle)
        robot = self.isaac_world.scene.get_object("target_object")
        if not robot._articulation_view.initialized:
            robot.get_articulation_controller()
        robot.set_joint_positions(joint_angle, [self.object_cfg["joint_id"]])
        
    def get_handle_pose(self):
        """Get the current joint angle of the articulated object"""
        robot = self.isaac_world.scene.get_object("target_object")
        if not robot._articulation_view.initialized:
            robot.get_articulation_controller()
        return robot.get_joint_positions()[self.object_cfg["joint_id"]]

    def _init_motion_gen(self):
        """Initialize motion generator for visualization."""
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            WorldConfig(voxel=[VoxelGrid(name="default", pose=[100, 100, 100, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1])]),
            self.tensor_args,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            interpolation_dt=0.05,
            collision_cache={"obb": 30, "mesh": 100},
            optimize_dt=True,
            trajopt_dt=None,
            trajopt_tsteps=32,
            trim_steps=None,
            use_cuda_graph=False,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        
    def _init_motion_gen_akr(self):
        """Initialize AKR motion generator for visualization."""
        motion_gen_config_akr = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            WorldConfig(voxel=[VoxelGrid(name="default", pose=[100, 100, 100, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1])]),
            self.tensor_args,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            interpolation_dt=0.05,
            collision_cache={"obb": 30, "mesh": 100},
            optimize_dt=True,
            trajopt_dt=None,
            trajopt_tsteps=32,
            trim_steps=None,
            use_cuda_graph=False,
            gradient_trajopt_file="gradient_trajopt_fixbase.yml",
        )
        self.motion_gen_akr = MotionGen(motion_gen_config_akr)

    def vis_spheres(self, joint_state, spheres, akr=False):
        """Visualize robot as spheres."""
        if akr:
            sph_list = self.motion_gen_akr.kinematics.get_robot_as_spheres(joint_state.position)
        else:
            sph_list = self.motion_gen.kinematics.get_robot_as_spheres(joint_state.position)
            
        if len(spheres) == 0:
            # create spheres:
            for si, s in enumerate(sph_list[0]):
                sp = sphere.VisualSphere(
                    prim_path="/curobo/robot_sphere_" + ("akr_" if akr else "") + str(si),
                    position=np.ravel(s.position),
                    radius=float(s.radius),
                    color=np.array([0, 0.8, 0.2]),
                )
                spheres.append(sp)
        else:
            for si, s in enumerate(sph_list[0]):
                if not np.isnan(s.position[0]):
                    spheres[si].set_world_pose(position=np.ravel(s.position))
                    spheres[si].set_radius(float(s.radius))
                    
    def replay_single_ik(self, iks, robot_name):
        """Replay IK solutions."""
        self._init_motion_gen()
        
        pose_duration = 1.0
        joint_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        robot = self.robot
        idx_list = [robot.get_dof_index(name) for name in joint_names]
    
        self._isaacsim_step(duration=5, render=True)
        
        print("Starting IK visualization...")
        print("iks shape:", iks.shape)
        
        spheres = []
        # Visualize IK solutions
        for i, pose in enumerate(iks):
            print(f"IK {i}:: {pose.tolist()}")
            robot_pose, handle_pose = pose.split([pose.shape[0] - 1, 1], dim=-1)
            joint_state = JointState.from_position(self.tensor_args.to_device(robot_pose))

            robot_pose = self._adjust_pose_for_robot(robot_pose, robot_name)
            self.vis_spheres(joint_state=joint_state, spheres=spheres)

            robot.set_joint_positions(robot_pose.tolist(), idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000] * len(idx_list)), joint_indices=idx_list,
            )

            self.set_handle_pose(handle_pose.item())
            self._isaacsim_step(duration=pose_duration, render=True)

    def replay_ik(self, start_iks, goal_iks, robot_name):
        """Replay IK solutions."""
        self._init_motion_gen()
        
        pose_duration = 1.0
        joint_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        robot = self.robot
        idx_list = [robot.get_dof_index(name) for name in joint_names]
    
        self._isaacsim_step(duration=5, render=True)
        
        print("Starting IK visualization...")
        print("start_iks shape:", start_iks.shape, "goal_iks shape:", goal_iks.shape)
        
        spheres = []
        # Visualize start IK solutions
        for i, pose in enumerate(start_iks):
            print(f"Start IK {i}:: {pose.tolist()}")
            robot_pose, handle_pose = pose.split([pose.shape[0] - 1, 1], dim=-1)
            joint_state = JointState.from_position(self.tensor_args.to_device(robot_pose))

            robot_pose = self._adjust_pose_for_robot(robot_pose, robot_name)
            self.vis_spheres(joint_state=joint_state, spheres=spheres)

            robot.set_joint_positions(robot_pose.tolist(), idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000] * len(idx_list)), joint_indices=idx_list,
            )

            self.set_handle_pose(handle_pose.item())
            self._isaacsim_step(duration=pose_duration, render=True)

        # Visualize goal IK solutions
        for i, pose in enumerate(goal_iks):
            print(f"Goal IK {i}: {pose.tolist()}")
            robot_pose, handle_pose = pose.split([pose.shape[0] - 1, 1], dim=-1)
            joint_state = JointState.from_position(self.tensor_args.to_device(robot_pose))

            robot_pose = self._adjust_pose_for_robot(robot_pose, robot_name)
            self.vis_spheres(joint_state=joint_state, spheres=spheres)

            robot.set_joint_positions(robot_pose.tolist(), idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000] * len(idx_list)), joint_indices=idx_list,
            )

            self.set_handle_pose(handle_pose.item())
            self._isaacsim_step(duration=pose_duration, render=True)

    def replay_traj(self, start_states, goal_states, trajs, successes, robot_name):
        """Replay trajectory solutions."""
        self._init_motion_gen()
            
        pose_duration = 0.1
        stop_duration = 1.0
        
        joint_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        robot = self.robot
        idx_list = [robot.get_dof_index(name) for name in joint_names]
    
        indices = self._get_indices(successes, all=False)
        self._isaacsim_step(duration=5, render=True)
        
        for idx in indices:
            start_state = start_states[idx]
            goal_state = goal_states[idx]
            traj = trajs[idx]
            success = successes[idx]
            
            print(f"Replaying {idx} with {len(traj)} steps, success: {success}")
            
            # Set to start state
            robot_pose = start_state[:-1]
            handle_pose = start_state[-1:]
            robot_pose = self._adjust_pose_for_robot(robot_pose, robot_name)
            robot.set_joint_positions(robot_pose.tolist(), idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000] * len(idx_list)), joint_indices=idx_list,
            )
            start_time = time.time()
            while time.time() - start_time < pose_duration:
                robot.set_joint_positions(robot_pose.tolist(), idx_list)
                self.set_handle_pose(handle_pose.item())
                self._isaacsim_step(step=1, render=True)
            
            # Step through trajectory
            for step_idx, pose in enumerate(traj):
                robot_pose = pose[:-1]
                handle_pose = pose[-1:]
                robot_pose = self._adjust_pose_for_robot(robot_pose, robot_name)
                start_time = time.time()
                while time.time() - start_time < pose_duration:
                    robot.set_joint_positions(robot_pose.tolist(), idx_list)
                    self.set_handle_pose(handle_pose.item())
                    self._isaacsim_step(step=1, render=True)
                
            # Set to goal state
            # robot_pose = goal_state[:-1]
            # handle_pose = goal_state[-1:]
            # robot_pose = self._adjust_pose_for_robot(robot_pose, robot_name)
            # robot.set_joint_positions(robot_pose.tolist(), idx_list)
            # self.set_handle_pose(handle_pose.item())
            self._isaacsim_step(duration=stop_duration, render=True)
       
    def replay_traj_akr(self, start_states, goal_states, trajs, successes, robot_name):
        """Replay AKR trajectory solutions."""
        self._init_motion_gen_akr()
            
        pose_duration = 0.1
        stop_duration = 1.0
        
        joint_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        robot = self.robot
        idx_list = [robot.get_dof_index(name) for name in joint_names]
    
        indices = self._get_indices(successes, all=True)
        self._isaacsim_step(duration=5, render=True)
        
        spheres = []
        
        for idx in indices:
            start_state = start_states[idx]
            goal_state = goal_states[idx]
            traj = trajs[idx]
            success = successes[idx]
            
            print(f"Replaying AKR {idx} with {len(traj)} steps, success: {success}")
            
            # Set to start state
            robot_pose = start_state
            robot_pose = self._adjust_pose_for_akr_robot(robot_pose, robot_name)
            robot.set_joint_positions(robot_pose.tolist(), idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000] * len(idx_list)), joint_indices=idx_list,
            )
            joint_state = JointState.from_position(self.tensor_args.to_device(robot_pose))
            self.vis_spheres(joint_state=joint_state, spheres=spheres, akr=True)
            self._isaacsim_step(duration=stop_duration, render=True)
            
            # Step through trajectory
            for step_idx, pose in enumerate(traj):
                robot_pose = pose
                robot_pose = self._adjust_pose_for_akr_robot(robot_pose, robot_name)
                robot.set_joint_positions(robot_pose.tolist(), idx_list)
                joint_state = JointState.from_position(self.tensor_args.to_device(robot_pose))
                self.vis_spheres(joint_state=joint_state, spheres=spheres, akr=True)
                self._isaacsim_step(duration=pose_duration, render=True)
                
            # Set to goal state
            robot_pose = goal_state
            robot_pose = self._adjust_pose_for_akr_robot(robot_pose, robot_name)
            robot.set_joint_positions(robot_pose.tolist(), idx_list)
            
    # def replay_traj_record(self, start_states, goal_states, trajs, successes, robot_name):
        

    def _get_indices(self, success, all=False):
        """Get indices for replay."""
        if all:
            return list(range(len(success)))
        else:
            return [i for i, s in enumerate(success) if s]
    
    # ========================================
    # Camera Helper Functions
    # ========================================
    
    def _get_transform(self, transform):
        """Convert camera transform from cuakr format to Isaac Sim format."""
        translate = transform.get("translate", [0, 0, 0])
        orient = transform.get("orient", [0, 0, 0])
        scale = transform.get("scale", [1, 1, 1])

        # Convert Euler angles to quaternion following cuakr pattern
        if len(orient) == 3:
            try:
                rot_quat = rot_utils.euler_angles_to_quats(
                    torch.tensor([orient], dtype=torch.float32), degrees=True, extrinsic=False
                )[0].tolist()
            except:
                # Fallback to our own euler_to_quat if Isaac Sim not available
                rot_quat = euler_to_quat(
                    np.array(orient) * np.pi / 180, order='xyz'
                )
                if hasattr(rot_quat, 'tolist'):
                    rot_quat = rot_quat.tolist()
        elif len(orient) == 4:
            rot_quat = orient
        else:
            rot_quat = [1, 0, 0, 0]  # Identity quaternion
        
        return translate + rot_quat, scale

    # ========================================
    # Camera Setup and Data Collection Methods
    # ========================================
    
    def setup_cameras(self) -> Dict[str, Camera]:
        """
        Setup 3 fixed cameras for data collection:
        1. ego_topdown: attached to robot end effector (panda_hand)
        2. ego_wrist: attached to robot end effector (panda_hand)
        3. fix_local: attached to object
        """
        cameras = {}
        
        # Define camera configurations
        camera_configs = self._get_camera_configurations()
        
        for camera_type, config in camera_configs.items():
            # Create camera prim path based on cuakr structure
            if camera_type in ["ego_topdown", "ego_wrist"]:
                # Ego cameras attach to robot end effector
                camera_prim_path = f"/World/summit_panda/panda_hand/{camera_type}"
            elif camera_type == "fix_local":
                # Fix local camera attaches to object
                camera_prim_path = f"/World/object/{camera_type}"
            else:
                # Fallback for any other cameras
                camera_prim_path = f"/World/cameras/{camera_type}"
            
            # Create camera with matching cuakr resolution [320, 240]
            camera = Camera(
                prim_path=camera_prim_path,
                frequency=30,
                resolution=(320, 240)  # Height x Width - matching cuakr config
            )
            camera.initialize()
            camera.add_motion_vectors_to_frame()
            camera.add_distance_to_image_plane_to_frame()
            
            # Set focal length if specified
            if "focal_length" in config:
                camera.set_focal_length(config["focal_length"])
                print(f"Camera {camera_type} focal length set to {config['focal_length']}")
            
            # Set camera pose based on type
            self._set_camera_pose(camera, camera_type, config)
            
            cameras[camera_type] = camera
            log_info(f"Camera {camera_type} initialized at {camera_prim_path}")
            
        return cameras
    
    def _get_camera_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get camera configurations for the 3 fixed cameras matching original cuakr values."""
        
        # Use original cuakr camera poses from collect_data.py and dataset configs
        camera_transforms = {
            "ego_topdown": {
                "translate": [0, 0, 10],  # From original cuakr configuration
                "orient": [0, 0, -90]  # Identity orientation
            },
            "ego_wrist": {
                "translate": [-0.6, 0, -0.9],  # From collect_data.py
                "orient": [180, -28, 90]  # From collect_data.py
            },
            "fix_local": {
                "translate": [-4.3, 4.7, 6.2],  # From dataset config
                "orient": [-34.0, -26.0, -140.0]  # From dataset config
            }
        }
        
        # Check and mirror fix_local camera if outside scene bounds
        if "fix_local" in camera_transforms:
            camera_transforms["fix_local"] = self._check_and_mirror_camera(
                camera_transforms["fix_local"]
            )
        
        # Convert to Isaac Sim format using cuakr's get_transform
        camera_configs = {}
        for camera_type, transform in camera_transforms.items():
            pose, scale = self._get_transform(transform)
            
            camera_configs[camera_type] = {
                "type": "local",  # Local pose relative to parent
                "local_pose": {
                    "position": pose[:3],  # [x, y, z]
                    "orientation": pose[3:]  # [qw, qx, qy, qz] or [qx, qy, qz, qw]
                }
            }
            
            # Add focal length for ego_wrist as in original code
            if camera_type == "ego_wrist":
                camera_configs[camera_type]["focal_length"] = 1.5
                
        return camera_configs
    
    def _check_and_mirror_camera(self, camera_transform: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if fix_local camera is outside scene bounding box and mirror if needed.
        
        Mirror the camera across the X-Z plane (mirror Y coordinate) if it's outside Y bounds.
        This keeps X and Z unchanged while flipping Y to bring the camera inside the scene.
        
        Args:
            camera_transform: Camera transform dict with 'translate' and 'orient' keys
            
        Returns:
            Modified camera transform if mirroring was needed, otherwise original
        """
        # Get scene bounds
        scene_bounds = self.scene_bounds
        
        translate = camera_transform["translate"].copy() if isinstance(camera_transform["translate"], list) else list(camera_transform["translate"])
        orient = camera_transform["orient"].copy() if isinstance(camera_transform["orient"], list) else list(camera_transform["orient"])
        
        x, y, z = translate
        
        # Check if camera Y coordinate is outside scene bounds
        if y < scene_bounds['y'][0] or y > scene_bounds['y'][1]:
            log_info(f"Camera Y={y:.2f} is outside bounds {scene_bounds['y']}")
            log_info(f"Mirroring fix_local camera from ({x:.2f}, {y:.2f}, {z:.2f})")
            
            # Mirror Y coordinate (across X-Z plane)
            translate[1] = -y
            
            # Mirror the yaw orientation to maintain view toward object
            # For Euler angles [roll, pitch, yaw], mirror the yaw
            orient[2] = -orient[2]
            
            log_info(f"Mirrored to ({translate[0]:.2f}, {translate[1]:.2f}, {translate[2]:.2f}), orient={orient}")
        
        return {
            "translate": translate,
            "orient": orient
        }
    
    def _set_camera_pose(self, camera: Camera, camera_type: str, config: Dict[str, Any]) -> None:
        """Set camera pose based on configuration."""
        if config["type"] == "local":
            # Local pose relative to parent prim (robot end effector or object)
            pose_config = config["local_pose"]
            camera.set_local_pose(
                pose_config["position"],
                pose_config["orientation"],
                camera_axes="usd"
            )
        elif config["type"] == "world":
            # World pose
            pose_config = config["world_pose"] 
            camera.set_world_pose(
                pose_config["position"],
                pose_config["orientation"],
                camera_axes="usd"
            )
    
    def _update_ego_cameras(self, cameras: Dict[str, Camera]) -> None:
        """Update ego camera positions based on current robot pose."""
        # Only ego_topdown needs dynamic updates based on robot pose
        if "ego_topdown" in cameras:
            # Get robot end effector pose using proper prim pose detection
            robot_link_prim_path = "/World/summit_panda/panda_hand"
            robot_link_pose_matrix = self.get_prim_pose(robot_link_prim_path)
            
            if robot_link_pose_matrix is not None:
                # Convert matrix to Pose object and extract position
                robot_link_pose = Pose.from_matrix(robot_link_pose_matrix)
                robot_pos = robot_link_pose.position.cpu().numpy()
                
                # Update camera position (ego_topdown follows end effector)
                camera_configs = self._get_camera_configurations()
                ego_config = camera_configs["ego_topdown"]["local_pose"]
                camera_pos = robot_pos + np.array(ego_config["position"])
                
                cameras["ego_topdown"].set_world_pose(
                    camera_pos.tolist(),
                    ego_config["orientation"],
                    camera_axes="usd"
                )
            else:
                log_warn(f"Failed to get robot pose from {robot_link_prim_path}, skipping ego_topdown camera update")
    
    def _get_camera_observations(self, cameras: Dict[str, Camera]) -> Dict[str, Dict[str, np.ndarray]]:
        """Get RGB, depth, and point cloud data from cameras following collect_data.py format."""
        observations = {
            "rgb": {},
            "depth": {},
            "point_cloud": {}
        }
        
        # Collect RGB and depth data from all cameras
        for camera_type, camera in cameras.items():
            # Get RGB data (remove alpha channel) 
            rgba = camera.get_rgba()
            if rgba is not None:
                observations["rgb"][camera_type] = rgba[:, :, :3]  # Remove alpha
            
            # Get depth data
            depth = camera.get_depth()
            if depth is not None:
                observations["depth"][camera_type] = depth
        
        # Get point cloud data from ego_topdown camera (following collect_data.py pattern)
        if "ego_topdown" in cameras:
            camera = cameras["ego_topdown"]
            pc = camera.get_pointcloud()
            if pc is not None:
                rgb = camera.get_rgba()
                if rgb is not None:
                    rgb = rgb[:, :, :3]  # Remove alpha channel
                    
                    # Create colored point cloud
                    colored_pc = create_colored_pointcloud(pc, rgb)
                    
                    # Process point cloud with configuration matching collect_data.py
                    pcd_config = PCDProcConfig()
                    pcd_config.USE_FPS = False
                    pcd_config.random_drop_points = min(
                        int(1 / 4 * 240 * 320),  # Based on camera resolution
                        2 * 1024,  # 2 * n_points from config
                    )
                    pcd_config.n_points = 1024  # From CollectConfig.POINT_CLOUD_CONFIG
                    
                    processed_pc = process_point_cloud(colored_pc, pcd_config)
                    observations["point_cloud"]["combined"] = processed_pc
                    
        return observations
    
    def _check_episode_exists(self, output_dir: str, episode_idx: int) -> bool:
        """Check if HDF5 file already exists for this episode."""
        camera_data_dir = os.path.join(output_dir, "camera_data")
        episode_filename = f"episode{episode_idx:06d}.hdf5"
        episode_path = os.path.join(camera_data_dir, episode_filename)
        return os.path.exists(episode_path)

    def replay_traj_record(self, start_states: torch.Tensor, goal_states: torch.Tensor, 
                          trajs: torch.Tensor, successes: torch.Tensor, robot_name: str,
                          output_dir: str, scene_id: str, object_id: str, 
                          angle_id: str = "0", pose_id: str = "0", num_episodes = None) -> None:
        """
        Record trajectory data with camera observations following collect_data.py format.
        
        Args:
            start_states: Start states for trajectories
            goal_states: Goal states for trajectories  
            trajs: Trajectory data
            successes: Success flags for each trajectory
            robot_name: Name of the robot
            output_dir: Directory to save data
            scene_id: Scene identifier
            object_id: Object identifier
            angle_id: Angle identifier
            pose_id: Pose identifier
        """
        log_info(f"=== Recording trajectory data for {robot_name} ===")
        
        # Setup cameras
        cameras = self.setup_cameras()
        
        self._init_motion_gen()
        
        self.robot_name = robot_name
        
        # Get successful trajectory indices
        successful_indices = self._get_indices(successes, all=False)
        log_info(f"Recording {len(successful_indices)} successful trajectories")
        
        # Create output directory for camera data
        camera_data_dir = os.path.join(output_dir, "camera_data")
        os.makedirs(camera_data_dir, exist_ok=True)
        
        if num_episodes is None:
            num_episodes = len(successful_indices)
        num_episodes = min(num_episodes, len(successful_indices))

        # --- LeRobot Dataset Initialization ---
        # Ensure robot is available to get DOF info
        robot = self.isaac_world.scene.get_object("robot_0") or self.robot
        if robot is None:
            log_warn("Robot not found, cannot initialize LeRobot dataset")
            return

        joint_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        num_action_joints = len(joint_names)
        
        state_names = robot.dof_names
        num_state_joints = len(state_names)
        
        # Get image resolution from config (assuming all cameras have same resolution for now)
        cam_cfg = list(self.camera_cfg.values())[0]
        width = cam_cfg["width"]
        height = cam_cfg["height"]

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (num_state_joints,),
                "names": state_names
            },
            "action": {
                "dtype": "float32",
                "shape": (num_action_joints,),
                "names": joint_names
            },
        }
        for cam_name in cameras.keys():
            features[f"observation.images.{cam_name}"] = {
                "dtype": "video",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"]
            }

        dataset = LeRobotDataset.create(
            repo_id=os.path.basename(output_dir), # Use output_dir name as repo_id
            root=output_dir,
            fps=15, # Assuming 15 FPS
            features=features,
            robot_type=robot_name,
            use_videos=True
        )
        # --------------------------------------
        
        for i, traj_idx in enumerate(tqdm(successful_indices[:num_episodes], desc="Recording trajectories")):  # Limit to specified number of successful trajectories
            log_info(f"Recording trajectory {i}/{num_episodes}")
            
            # Check if HDF5 file already exists for this episode
            # if self._check_episode_exists(output_dir, i):
            #     log_info(f"Episode {i} already exists, skipping")
            #     continue
            
            # Initialize camera result
            camera_result = CameraResult()
            camera_result.set_env_info(
                scene_id=scene_id,
                robot_name=robot_name,
                object_id=object_id,
                pose_id=pose_id
            )
            
            # Initialize data structures
            camera_result.initialize_joint_structure(robot_name)
            camera_result.initialize_camera_structure(list(cameras.keys()))
            
            # Execute trajectory and record data
            trajectory = trajs[traj_idx]
            start_state = start_states[traj_idx]
            goal_state = goal_states[traj_idx]
            
            # Set robot to start state
            robot = self.isaac_world.scene.get_object("robot_0") or self.robot
            if robot is None:
                log_warn("Robot not found, skipping trajectory recording")
                continue
                
            # Record trajectory execution
            self._record_trajectory_execution(
                robot, trajectory, cameras, camera_result, robot_name,
                dataset=dataset,
                task_description=f"Interact with {object_id}"
            )
            
            # Finalize and save data
            camera_result.finalize()
            # self._save_camera_result(camera_result, camera_data_dir, i)
            
            # Save LeRobot episode
            dataset.save_episode()
            
            # Clean up memory for trajectory data - release camera_result immediately
            del trajectory, start_state, goal_state, camera_result
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        log_info(f"Completed recording {num_episodes} trajectories")
        dataset.finalize()
    
    def _get_robot_joint_names(self, robot_name: str) -> List[str]:
        """Get joint names for the robot based on robot type."""
        # This should match the joint configuration from collect_data.py
        joint_configs = {
            "summit_franka": [
                "base_x", "base_y", "base_z",
                "panda_joint1", "panda_joint2", "panda_joint3", 
                "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",
                "panda_finger_joint1", "panda_finger_joint2"
            ],
            "r1": [
                "base_x", "base_y", "base_z",
                "torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4",
                "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
                "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
                "left_gripper_axis1", "left_gripper_axis2"
            ]
        }
        return joint_configs.get(robot_name, [])
    
    def _record_trajectory_execution(self, robot: Robot, trajectory: torch.Tensor, 
                                   cameras: Dict[str, Camera], camera_result: CameraResult,
                                   robot_name: str, steps=5,
                                   dataset: Optional[LeRobotDataset] = None,
                                   task_description: str = "") -> None:
        """Execute trajectory and record observations at each timestep."""
        
        # Convert trajectory to numpy if needed
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.cpu().numpy()
        
        # Get joint indices like in replay_traj
        joint_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        idx_list = [robot.get_dof_index(name) for name in joint_names]
        
        
        # Execute each step of the trajectory
        for step_idx, step_pose in enumerate(trajectory):
            # Debug trajectory shape
            # print(f"Original step_pose shape: {step_pose.shape}")
            
            # Split pose like in replay_traj: robot_pose (all but last) and handle_pose (last)
            robot_pose = step_pose[:-1]
            handle_pose = step_pose[-1:]
            
            # print(f"After split: robot_pose shape: {robot_pose.shape}, handle_pose shape: {handle_pose.shape}")
            
            # Convert to torch tensor if it's numpy (needed for _adjust_pose_for_robot)
            if isinstance(robot_pose, np.ndarray):
                robot_pose = torch.from_numpy(robot_pose).float()
            
            # Apply adjustment like in replay_traj
            if robot_name.startswith("akr_"):
                robot_pose = self._adjust_pose_for_akr_robot(robot_pose, robot_name)
            else:
                robot_pose = self._adjust_pose_for_robot(robot_pose, robot_name)
                
            # print(f"After adjustment: robot_pose shape: {robot_pose.shape}")
            # print(f"idx_list length: {len(idx_list)}")
            # print(f"Robot DOF: {robot.num_dof}")
            
            # Set joint positions with idx_list like in replay_traj
            robot.set_joint_positions(robot_pose.tolist(), idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000] * len(idx_list)), joint_indices=idx_list,
            )

            # Warm up simulation
            if step_idx == 0:
                set_steps = 50
            else:
                set_steps = 5
            
            for _ in range(set_steps):
                # Set handle pose
                self.set_handle_pose(handle_pose.item())
                
                # Step simulation
                self._isaacsim_step(step=1, render=True)
                
                # Update ego cameras
                self._update_ego_cameras(cameras)
                

            # Collect observations
            observations = self._get_camera_observations(cameras)
            
            # Get current robot state for joint observations
            current_joint_positions = robot.get_joint_positions()
            
            # Get end effector pose
            eef_pose = self._get_end_effector_pose(robot)
            
            # Add grouped joint observation using the new method
            if current_joint_positions is not None:
                camera_result.add_grouped_joint_observation(current_joint_positions, robot_name)
            
            # Add other observations to camera result
            camera_result.add_observation(
                eef_data=eef_pose,
                point_cloud_data=observations["point_cloud"].get("combined"),
                rgb_data=observations["rgb"],
                depth_data=observations["depth"]
            )

            if dataset is not None:
                frame = {}
                # State: current_joint_positions
                frame["observation.state"] = torch.from_numpy(current_joint_positions).float()
                
                # Action: robot_pose (target)
                if isinstance(robot_pose, np.ndarray):
                    frame["action"] = torch.from_numpy(robot_pose).float()
                else:
                    frame["action"] = robot_pose.float()
                
                for cam_name in cameras.keys():
                    frame[f"observation.images.{cam_name}"] = observations['rgb'][cam_name]
                
                frame["task"] = task_description
                dataset.add_frame(frame)
    
    def _get_end_effector_pose(self, robot: Robot) -> Optional[np.ndarray]:
        """Get current end effector pose as 7D array [x, y, z, qw, qx, qy, qz]."""
        # This is a simplified implementation - in practice you'd get the actual EE pose

        # Get end-effector pose
        # joint_angle = self.get_handle_pose()
        # gt_eef_pose = self._get_open_ee_pose(self.object_pose, joint_angle)
        # gt_eef_pose_7d = self.pose_to_7D_numpy(gt_eef_pose)

        # Get model end-effector pose
        robot_pose_np = robot.get_joint_positions()
        # TODO: fix dof bug for fixed base robots
        if "fixed" in self.robot_name:
            robot_pose_np = robot_pose_np[3:]
        js = JointState.from_position(self.tensor_args.to_device(robot_pose_np))
        fk_result = self.motion_gen.ik_solver.fk(js.position)
        model_eef_pose = fk_result.ee_pose
        model_eef_pose_7d = self.pose_to_7D_numpy(model_eef_pose)

        eef_pose_7d = model_eef_pose_7d
        return eef_pose_7d
    
    def pose_to_7D_numpy(self, pose):
        """Convert a pose to a 7D numpy array (position + quaternion)"""
        position = pose.position.flatten().cpu().numpy()
        quaternion = pose.quaternion.flatten().cpu().numpy()
        return np.concatenate([position, quaternion])
    
    def _save_camera_result(self, camera_result: CameraResult, output_dir: str, episode_idx: int) -> None:
        """Save camera result to HDF5 file following collect_data.py format."""
        filename = f"episode{episode_idx:06d}.hdf5"
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            log_info(f"Removed existing file: {filepath}")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with h5py.File(filepath, "w") as f:
            # Save env_info
            env_group = f.create_group("env_info")
            for key, value in camera_result.env_info.items():
                if isinstance(value, str):
                    env_group.create_dataset(key, data=value, dtype=h5py.string_dtype())
                else:
                    env_group.create_dataset(key, data=value)
            
            # Save observation data
            obs_group = f.create_group("obs")
            self._save_obs_group(obs_group, camera_result.obs)
            
        log_info(f"Saved camera result to {filepath}")
    
    def _save_obs_group(self, group: h5py.Group, obs_dict: Dict[str, Any]) -> None:
        """Recursively save observation dictionary to HDF5 group."""
        for key, value in obs_dict.items():
            if isinstance(value, dict):
                sub_group = group.create_group(key)
                self._save_obs_group(sub_group, value)
            elif isinstance(value, list) and len(value) > 0:
                # Convert list to numpy array for saving
                np_array = np.array(value)
                group.create_dataset(key, data=np_array)
            elif not isinstance(value, (list, dict)):
                # Scalar values as attributes
                group.attrs[key] = value
    
    # ========================================
    # Trajectory Evaluation Methods  
    # ========================================
    
    def replay_traj_evaluate(self, policy_model, start_states: torch.Tensor, 
                           goal_states: torch.Tensor, trajs: torch.Tensor, 
                           successes: torch.Tensor, robot_name: str,
                           scene_id: str, object_id: str, 
                           angle_id: str = "0", pose_id: str = "0", 
                           grasp_id: int = 0) -> List[TrajectoryEvaluationResult]:
        """
        Evaluate policy model on trajectories and compare with ground truth.
        
        Args:
            policy_model: Policy model for inference
            start_states: Start states for trajectories
            goal_states: Goal states for trajectories
            trajs: Ground truth trajectory data  
            successes: Success flags for each trajectory
            robot_name: Name of the robot
            scene_id: Scene identifier
            object_id: Object identifier
            angle_id: Angle identifier
            pose_id: Pose identifier
            grasp_id: Grasp identifier
            
        Returns:
            List of TrajectoryEvaluationResult objects
        """
        log_info(f"=== Evaluating policy model on trajectories ===")
        
        # Setup cameras for observation
        cameras = self.setup_cameras()
        
        self.robot_name = robot_name  # Store robot name for pose adjustments
        
        # Get successful trajectory indices
        successful_indices = self._get_indices(successes, all=False)
        log_info(f"Evaluating on {len(successful_indices)} successful trajectories")
        
        evaluation_results = []
        
        for i, traj_idx in enumerate(successful_indices[:5]):  # Limit to first 5 for evaluation
            log_info(f"Evaluating trajectory {i+1}/{min(5, len(successful_indices))}")
            
            # Get ground truth trajectory
            gt_trajectory = trajs[traj_idx]
            start_state = start_states[traj_idx]
            goal_state = goal_states[traj_idx]
            
            # Run policy evaluation
            eval_result = self._evaluate_single_trajectory(
                policy_model, gt_trajectory, start_state, cameras, robot_name
            )
            
            eval_result.trajectory_idx = traj_idx
            eval_result.grasp_id = grasp_id
            evaluation_results.append(eval_result)
        
        log_info(f"Completed evaluation on {len(evaluation_results)} trajectories")
        return evaluation_results
    
    def _evaluate_single_trajectory(self, policy_model, gt_trajectory: torch.Tensor,
                                   start_state: torch.Tensor, cameras: Dict[str, Camera],
                                   robot_name: str) -> TrajectoryEvaluationResult:
        """Evaluate policy model on a single trajectory."""
        
        # Initialize robot to start state
        robot = self.isaac_world.scene.get_object("robot_0") or self.robot
        if robot is None:
            log_warn("Robot not found for evaluation")
            return TrajectoryEvaluationResult(
                eef_poses=torch.empty(0, 2, 7),
                open_angles=torch.empty(0)
            )
        
        # Convert to numpy if needed
        if isinstance(gt_trajectory, torch.Tensor):
            gt_trajectory = gt_trajectory.cpu().numpy()
        if isinstance(start_state, torch.Tensor):
            start_state = start_state.cpu().numpy()
        
        # Set robot to start state
        start_pose = self._adjust_pose_for_robot(start_state, robot_name)
        robot.set_joint_positions(start_pose.tolist())
        
        model_eef_poses = []
        gt_eef_poses = []
        open_angles = []
        
        # Execute trajectory with policy
        for step_idx, gt_step in enumerate(gt_trajectory):
            # Get current observation
            self._update_ego_cameras(cameras) 
            observations = self._get_camera_observations(cameras)
            
            # Format observation for policy (simplified)
            policy_obs = self._format_observation_for_policy(observations, robot)
            
            try:
                # Get policy action (this would depend on the specific policy interface)
                policy_action = self._get_policy_action(policy_model, policy_obs)
                
                # Apply policy action
                if policy_action is not None:
                    policy_pose = self._adjust_pose_for_robot(policy_action, robot_name)
                    robot.set_joint_positions(policy_pose.tolist())
                
                # Get model end effector pose
                model_eef = self._get_end_effector_pose(robot)
                if model_eef is None:
                    model_eef = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                
                # Set robot to ground truth pose to get GT end effector pose
                gt_pose = self._adjust_pose_for_robot(gt_step, robot_name)
                robot.set_joint_positions(gt_pose.tolist())
                gt_eef = self._get_end_effector_pose(robot)
                if gt_eef is None:
                    gt_eef = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                
                model_eef_poses.append(model_eef)
                gt_eef_poses.append(gt_eef)
                
                # Extract handle angle (assuming last dimension is handle angle)
                if len(gt_step) > 7:  # Has handle angle
                    open_angles.append(gt_step[-1])
                else:
                    open_angles.append(0.0)
                
                # Step simulation
                self._isaacsim_step(step=3, render=True)
                
            except Exception as e:
                log_warn(f"Error in trajectory evaluation at step {step_idx}: {e}")
                break
        
        # Convert to tensors
        if len(model_eef_poses) > 0:
            model_eef_tensor = torch.tensor(np.array(model_eef_poses), dtype=torch.float32)
            gt_eef_tensor = torch.tensor(np.array(gt_eef_poses), dtype=torch.float32)
            eef_poses = torch.stack([model_eef_tensor, gt_eef_tensor], dim=1)  # Shape: (T, 2, 7)
        else:
            eef_poses = torch.empty(0, 2, 7)
        
        open_angles_tensor = torch.tensor(open_angles, dtype=torch.float32)
        
        return TrajectoryEvaluationResult(
            eef_poses=eef_poses,
            open_angles=open_angles_tensor,
            success=len(model_eef_poses) == len(gt_trajectory),
            num_steps=len(model_eef_poses)
        )
    
    def _format_observation_for_policy(self, observations: Dict, robot: Robot) -> Dict[str, Any]:
        """Format raw observations for policy input (simplified placeholder)."""
        # This is a simplified version - actual implementation would depend on policy requirements
        return {
            "rgb": observations.get("rgb", {}),
            "depth": observations.get("depth", {}),
            "point_cloud": observations.get("point_cloud", {}),
            "joint_positions": robot.get_joint_positions() if robot else None
        }
    
    def _get_policy_action(self, policy_model, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get action from policy model (placeholder - depends on policy interface)."""
        # This is a placeholder - actual implementation would depend on the specific policy
        # For now, return None to indicate no action
        log_warn("Policy action interface not implemented - placeholder function")
        return None
    
    
    def isaacsim_step(self, step=5, render=True):
        """Step the Isaac Sim world"""
        if step == -1:
            # If step is -1, run until the simulation app is running
            while self.simulation_app.is_running():
                self.isaac_world.step(render=render)
        for _ in range(step):
            self.isaac_world.step(render=render)
            
    def set_isaacsim_collision_free(self):
        from pxr import Gf, UsdGeom, UsdPhysics, UsdLux, UsdShade, Usd, Sdf
        def disable_collision(prim_path):
            UsdPhysics.CollisionAPI.Get(
                self.isaac_world.stage, prim_path
            ).GetCollisionEnabledAttr().Set(False)

        for prim_path in [
            "/World/summit_panda/panda_leftfinger/collisions",
            "/World/summit_panda/panda_rightfinger/collisions",
            "/World/summit_panda/grasp_frame/collisions",
            "/World/summit_panda/panda_hand/collisions",
            # "/World/object/partnet_5b2633d960419bb2e5bf1ab8e7d0b/link_0/collisions",
            # "/World/object/partnet_5b2633d960419bb2e5bf1ab8e7d0b/link_1/collisions"
        ]:
            disable_collision(prim_path)


# ============================================================================
# Helper Functions for Standalone Execution
# ============================================================================

def get_object_config(object_id: str) -> Dict[str, Any]:
    """Get the configuration for an object by ID."""
    if object_id not in OBJECT_CONFIG_MAP:
        raise ValueError(
            f"Unknown object ID: {object_id}. "
            f"Available objects: {list(OBJECT_CONFIG_MAP.keys())}"
        )
    return OBJECT_CONFIG_MAP[object_id]


def create_object(object_id: str) -> ObjectDescription:
    """Create an object description using the configuration map."""
    config = get_object_config(object_id)
    obj = ObjectDescription(
        asset_type=config["asset_type"],
        asset_id=config["asset_id"],
        scale=config["scale"],
        urdf_path=config["urdf_path"],
    )
    obj.set_handle_link(config["handle_link"])
    return obj


def load_scene(scene_path: str, objects: list):
    """Load scene from path."""
    from automoma.pipeline import InfinigenScenePipeline
    
    scene_pipeline = InfinigenScenePipeline()
    scene_result = scene_pipeline.load_scene(scene_path, objects)
    scene_pose = [0, 0, -0.13, 1, 0, 0, 0]
    scene_result.scene.set_pose(scene_pose)
    
    # Set object poses
    for obj in scene_result.valid_objects:
        print(f"Object pose before: {scene_result.scene.get_object_pose(obj)}")
        obj_pose = scene_result.scene.get_object_matrix(obj)
        obj_pose = single_axis_self_rotation(obj_pose, "z", np.pi)
        obj_pose = matrix_to_pose(obj_pose)
        print(f"Object pose after: {obj_pose}")
        obj.set_pose(obj_pose)
    
    return scene_result


def get_output_directory(robot_name: str, scene_name: str, object_id: str, grasp_id: int) -> str:
    """Generate organized output directory path."""
    output_base_dir = "data/collect_1222/traj"
    output_dir = os.path.join(
        output_base_dir,
        robot_name,
        scene_name,
        object_id,
        f"grasp_{grasp_id:04d}"
    )
    return output_dir


def main():
    """
    Main replay function for object 7221, grasp 0.
    Replays single IK, trajectories, and filtered trajectories.
    Directly uses Replayer class without ReplayPipeline.
    """
    print("=" * 80)
    print("STANDALONE REPLAYER - Object 7221, Grasp 0")
    print("=" * 80)
    
    # Configuration
    object_id = "7221"
    robot_name = "summit_franka"
    scene_path = "assets/scene/infinigen/kitchen_1130/scene_0_seed_0"
    grasp_id = 0
    
    print(f"\nConfiguration:")
    print(f"  Object ID: {object_id}")
    print(f"  Robot: {robot_name}")
    print(f"  Scene Path: {scene_path}")
    print(f"  Grasp ID: {grasp_id}\n")
    
    try:
        # Step 1: Create object description
        print("[1/4] Creating object description...")
        obj = create_object(object_id)
        
        # Step 2: Load scene
        print("[2/4] Loading scene...")
        scene_result = load_scene(scene_path, [obj])
        
        # Step 3: Extract scene name and setup configurations
        print("[3/4] Setting up configurations...")
        scene_path_obj = Path(scene_path)
        scene_name = scene_path_obj.parent.parent.parent.name
        
        # Get object configuration
        object_config = get_object_config(object_id)
        
        # Setup scene configuration
        scene_cfg = {
            "path": scene_result.scene.scene_usd_path,
            "pose": scene_result.scene.pose,
        }
        
        # Setup object configuration
        object_cfg = {
            "path": object_config["urdf_path"],
            "asset_type": object_config["asset_type"],
            "asset_id": object_config["asset_id"],
            "pose": obj.pose,
            "joint_id": 0,
        }
        
        # Setup robot configuration - load from file
        robot_cfg_path = f"assets/robot/{robot_name}/{robot_name}.yml"
        robot_cfg = load_yaml(robot_cfg_path)
        
        # Step 4: Initialize Replayer directly
        print("[4/4] Initializing Replayer...")
        replayer = Replayer(
            simulation_app=simulation_app,
            robot_cfg=robot_cfg,
            scene_cfg=scene_cfg,
            object_cfg=object_cfg
        )
        
        # Deactivate scene elements for visualization
        print("\nSetting up visualization...")
        if object_id in OBJECT_PRIM_MAP:
            object_prim = OBJECT_PRIM_MAP[object_id]
            print(f"  - Deactivating object prim: {object_prim}")
            replayer.set_deactivate_prims(object_prim)
        
        # Deactivate ceiling elements
        for prim_name in ["exterior", "ceiling", "Ceiling"]:
            print(f"  - Deactivating prim: {prim_name}")
            replayer.set_deactivate_prims(prim_name)
        
        # Set lighting
        action_registry = omni.kit.actions.core.get_action_registry()
        action = action_registry.get_action(
            "omni.kit.viewport.menubar.lighting",
            "set_lighting_mode_rig"
        )
        action.execute(lighting_mode=2)
        
        print("\n" + "=" * 80)
        print("REPLAY SEQUENCE")
        print("=" * 80)
        
        # Get output directory for loading data
        output_dir = get_output_directory(robot_name, scene_name, object_id, grasp_id)
        
        # 1. Replay single IK
        print(f"\n>>> Running Replay Single IK (Grasp {grasp_id})...")
        start_ik_path = os.path.join(output_dir, "start_ik_data.pt")
        if os.path.exists(start_ik_path):
            ik_data = torch.load(start_ik_path, weights_only=False)
            iks = ik_data["iks"]
            print(f"    Replaying {iks.shape[0]} IK solutions")
            replayer.replay_single_ik(iks, robot_name)
        else:
            print(f"    IK data not found at {start_ik_path}")
        
        # 2. Replay trajectories
        print(f"\n>>> Running Replay Trajectories (Grasp {grasp_id})...")
        traj_path = os.path.join(output_dir, "traj_data.pt")
        if os.path.exists(traj_path):
            traj_data = torch.load(traj_path, weights_only=False)
            start_states = traj_data["start_states"]
            goal_states = traj_data["goal_states"]
            trajectories = traj_data["trajectories"]
            success = traj_data["success"]
            
            successful_count = success.shape[0]
            success = torch.ones(successful_count, dtype=torch.bool)
            print(f"    Replaying {successful_count} trajectories")
            replayer.replay_traj(
                start_states=start_states,
                goal_states=goal_states,
                trajs=trajectories,
                successes=success,
                robot_name=robot_name
            )
        else:
            print(f"    Trajectory data not found at {traj_path}")
        
        # 3. Replay filtered trajectories
        print(f"\n>>> Running Replay Filtered Trajectories (Grasp {grasp_id})...")
        filtered_path = os.path.join(output_dir, "filtered_traj_data.pt")
        if os.path.exists(filtered_path):
            traj_data = torch.load(filtered_path, weights_only=False)
            start_states = traj_data["start_state"]
            goal_states = traj_data["goal_state"]
            trajectories = traj_data["traj"]
            success = traj_data["success"]
            
            successful_count = success.sum().item()
            print(f"    Replaying {successful_count} filtered trajectories")
            replayer.replay_traj(
                start_states=start_states,
                goal_states=goal_states,
                trajs=trajectories,
                successes=success,
                robot_name=robot_name
            )
        else:
            print(f"    Filtered trajectory data not found at {filtered_path}")
        
        print("\n" + "=" * 80)
        print("REPLAY COMPLETE - Keeping Isaac Sim running for visualization")
        print("=" * 80)
        
        # Keep running until window is closed
        replayer.isaacsim_step(step=-1, render=True)
        
    except Exception as e:
        print(f"\nError during replay: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
