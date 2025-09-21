"""
Replayer utility for visualizing IK and trajectory results in Isaac Sim.
Adapted from the existing test_replay.py script.
"""

# Isaac Sim imports
# import isaacsim
# from omni.isaac.kit import SimulationApp

# # Set up simulation application configuration
# simulation_app = SimulationApp(
#     {
#         "headless": False,
#         "width": 1920,
#         "height": 1080
#     }
# )

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core import World
from omni.isaac.core.objects import sphere
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.importer.urdf import _urdf
from omni.isaac.core.robots import Robot
from omni.kit.commands import execute

# curobo imports
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util.logger import log_warn, log_info
from curobo.util.usd_helper import UsdHelper, set_prim_transform
from curobo.util_file import get_filename, get_path_of_dir, load_yaml
from curobo.types.state import JointState
from curobo.geom.types import WorldConfig, VoxelGrid
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

# Local project utils
from cuakr.utils.helper import add_robot_to_scene
from cuakr.utils.math import pose_multiply

# Standard library imports
from typing import Dict, List, Any, Sequence, Optional, Union
import torch
import numpy as np
import os
import time


class Replayer:
    def __init__(self, simulation_app, robot_cfg, scene_cfg, object_cfg):
        self._init_world()
        
        self.simulation_app = simulation_app
        self.robot_cfg = robot_cfg
        self.scene_cfg = scene_cfg
        self.object_cfg = object_cfg
        self.usd_helper = UsdHelper()
        self.usd_helper.load_stage(self.isaac_world.stage)
        
        self._load_robot(self.robot_cfg)
        self._load_scene(self.scene_cfg)
        self._load_object(self.object_cfg)

        self.isaac_world.initialize_physics()
        self.isaac_world.reset()
        self.tensor_args = TensorDeviceType()

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
        set_prim_transform(self.isaac_world.stage.GetPrimAtPath(prim_path), pose=scene_cfg["pose"])

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
        object_pose = object_cfg["pose"]

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

    def replay_ik(self, start_iks, goal_iks, robot_name):
        """Replay IK solutions."""
        self._init_motion_gen()
        
        pose_duration = 0.2
        joint_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        robot = self.robot
        idx_list = [robot.get_dof_index(name) for name in joint_names]
    
        self._isaacsim_step(duration=5, render=True)
        
        print("Starting IK visualization...")
        print("start_iks shape:", start_iks.shape, "goal_iks shape:", goal_iks.shape)
        
        spheres = []
        # Visualize start IK solutions
        for i, pose in enumerate(start_iks):
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
            self.set_handle_pose(handle_pose.item())
            self._isaacsim_step(duration=stop_duration, render=True)
            
            # Step through trajectory
            for step_idx, pose in enumerate(traj):
                robot_pose = pose[:-1]
                handle_pose = pose[-1:]
                robot_pose = self._adjust_pose_for_robot(robot_pose, robot_name)
                robot.set_joint_positions(robot_pose.tolist(), idx_list)
                self.set_handle_pose(handle_pose.item())
                self._isaacsim_step(duration=pose_duration, render=True)
                
            # Set to goal state
            robot_pose = goal_state[:-1]
            handle_pose = goal_state[-1:]
            robot_pose = self._adjust_pose_for_robot(robot_pose, robot_name)
            robot.set_joint_positions(robot_pose.tolist(), idx_list)
            self.set_handle_pose(handle_pose.item())
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

    def _get_indices(self, success, all=False):
        """Get indices for replay."""
        if all:
            return list(range(len(success)))
        else:
            return [i for i, s in enumerate(success) if s]