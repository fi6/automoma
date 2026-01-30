"""
AKR Planner - A motion planning system for articulated kinematic robots

This module provides a comprehensive interface for motion planning with cuRobo,
specifically designed for articulated kinematic robot (AKR) systems. It separates 
planning logic from simulation code and provides a clean API for:

Key Features:
- Complete scene loading and collision world setup from USD files
- Robot configuration and motion generation management  
- Articulated object manipulation with proper transform handling
- IK and trajectory planning for AKR systems with batch processing
- Automatic metadata loading and object pose processing
- Grasp pose loading and scaling utilities
- Collision checking with voxel grids and ESDF generation
- Save/load functionality for IK and trajectory data

Core Components:
1. AKRPlanner: Main planning class with complete pipeline
2. Scene management: USD loading, collision world setup
3. Object handling: URDF loading, pose processing, joint configurations
4. Transform utilities: Complete get_open_ee_pose implementation
5. Planning algorithms: IK clustering, trajectory optimization

Usage Patterns:

1. Factory Method (Recommended):
   ```python
   config = AKRPlanner.create_default_config(
       scene_path="path/to/scene.usdc",
       robot_cfg_path="path/to/robot.yml", 
       object_urdf_path="path/to/object.urdf",
       metadata_path="path/to/metadata.json"
   )
   planner = AKRPlanner.from_config(config)
   ```

2. Manual Configuration:
   ```python
   planner = AKRPlanner(scene_cfg, robot_cfg, object_cfg)
   planner.load_object_from_metadata("metadata.json")
   ```

3. Complete Planning Pipeline:
   ```python
   # Load grasp poses
   grasp_poses = planner.get_grasp_poses(grasp_dir, scaling_factor=0.356)
   
   # Plan IK with transform handling
   start_iks, goal_iks = planner.plan_ik(
       grasp_poses, start_angle=0.0, goal_angle=1.57
   )
   
   # Plan trajectories  
   start_states, goal_states, trajs, success = planner.plan_traj(
       start_iks, goal_iks, akr_robot_cfg_path
   )
   
   # Save results
   planner.save_ik(start_iks, goal_iks, "ik_data.pt")
   planner.save_traj(start_states, goal_states, trajs, success, "traj_data.pt")
   ```

Requirements:
- All transform functions from automoma.utils.transform
- cuRobo for motion planning
- yourdfpy for URDF handling
- USD libraries for scene loading
- Complete implementation of get_open_ee_pose with proper transform chains

Author: Generated for cuAKR project
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

# Third Party
from curobo.util.usd_helper import UsdHelper, set_prim_transform
from curobo.util_file import load_yaml
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.types import WorldConfig, Cuboid, VoxelGrid, Mesh
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.sdf.world_voxel import WorldVoxelCollision
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollisionConfig
from curobo.rollout.rollout_base import Goal
from yourdfpy import URDF
from tqdm import tqdm
from dataclasses import dataclass

# Local imports
from automoma.utils.transform import *
from cuakr.utils.math import pose_multiply, ik_clustering
from cuakr.utils.voxel import mark_cuboid_as_empty, visualize_voxel_grid_with_cuboid, visualize_voxel_grid_with_transparency

DISABLE_COLLISION = False  #TODO: set to False for real planning
@dataclass
class IKResult:
    start_ik: torch.Tensor
    goal_ik: torch.Tensor

@dataclass
class TrajResult:
    start_states: torch.Tensor
    goal_states: torch.Tensor
    trajectories: torch.Tensor
    success: torch.Tensor

class AKRPlanner:
    """
    Articulated Kinematic Robot Planner
    
    A comprehensive motion planning system that handles:
    - Scene loading and collision world setup
    - Robot configuration and motion generation
    - Object manipulation and articulated object handling
    - IK and trajectory planning for AKR systems
    """
    
    def __init__(
        self, 
        scene_cfg: Dict[str, Any] = None, 
        object_cfg: Dict[str, Any] = None,
        robot_cfg: Dict[str, Any] = None
    ):
        """
        Initialize the AKR Planner
        
        Args:
            scene_cfg: Scene configuration containing path, pose, etc.
            robot_cfg: Robot configuration dictionary
            object_cfg: Object configuration containing path, pose, joint info, etc.
        """
        self.scene_cfg = scene_cfg
        self.robot_cfg = robot_cfg
        self.object_cfg = object_cfg
        
        # Initialize core components
        self.tensor_args = TensorDeviceType()
        self.usd_helper = UsdHelper()
        
        # Initialize storage for motion generators
        self.motion_gen = None
        self.motion_gen_akr = None
        
        # Initialize world collision components
        self.world_voxel_collision_ik = None
        self.world_voxel_collision_traj = None
        self.esdf = None
        
        # Object state
        self.object_urdf = None
        self.object_pose = None
        self.object_mesh = None

        self.root_pose = [0, 0, 0, 1, 0, 0, 0]

        # Object pose with scene offset applied
        self.object_center = True    
        
        self._setup_environment()
        
        
    def _setup_environment(self):
        """Setup the complete planning environment"""
        self._init_root_pose()
        self._load_object()
        self._load_scene()
        self._setup_collision_world()
        
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
        
    def _load_object(self) -> None:
        """Load and setup the articulated object"""
        object_path = self.object_cfg["path"]
        if not os.path.exists(object_path):
            raise FileNotFoundError(f"Object URDF not found: {object_path}")
        
        print(f"Loading object: {object_path}")
        
        # Load object pose
        self.object_pose = self.get_world_pose(self.object_cfg["pose"])
        
        # Load URDF
        self.object_urdf = URDF.load(object_path, build_collision_scene_graph=True)
        
        # Create object mesh
        object_trimesh = self.object_urdf.scene.to_mesh()
        # Use safe naming convention: asset_type + asset_id
        object_name = "target_object"
        self.object_mesh = Mesh(
            trimesh=object_trimesh, 
            name=object_name, 
            pose=self.object_pose
        )
    
    def _load_scene(self) -> None:
        """Load the scene from USD file and setup collision world"""
        usd_path = self.scene_cfg["path"]
        if not os.path.exists(usd_path):
            raise FileNotFoundError(f"USD file not found: {usd_path}")
        
        print(f"Loading scene: {usd_path}")
        self.usd_helper.load_stage_from_file(usd_path)
        
        # Apply scene offset if specified
        set_prim_transform(
            self.usd_helper.stage.GetPrimAtPath("/World/scene"), 
            self.root_pose
        )
        
        # Get collision world from scene
        print("Getting collision world from scene...")
        self.collision = self.usd_helper.get_obstacles_from_stage().get_collision_check_world()
        print(f"Number of collision meshes: {len(self.collision.mesh)}")
        
        # Remove empty meshes
        self._clean_collision_meshes()
    
    def _clean_collision_meshes(self):
        """Remove empty meshes from collision world"""
        to_remove = []
        for obj in self.collision.mesh:
            # print(f"Mesh name: {obj.name}, vertices: {len(obj.vertices)}, faces: {len(obj.faces)}")
            if len(obj.vertices) == 0 or len(obj.faces) == 0:
                # print(f"Removing empty mesh: {obj.name}")
                to_remove.append(obj)
        
        for obj in to_remove:
            self.collision.mesh.remove(obj)
            
        print(f"Removed {len(to_remove)} empty meshes. Remaining meshes: {len(self.collision.mesh)}")

    def _setup_collision_world(self):
        """Setup collision checking world with voxel grids"""
        # Create voxel configuration (hyperparameters)
        object_pose_Pose = Pose.from_list(self.object_pose)
        object_pose_inverse = object_pose_Pose.inverse().to_list()
        
        voxel_dims = [5.0, 5.0, 5.0]
        voxel_size = 0.02
        expanded_dim = 0.2
        
        world_model = {
            "voxel": {
                "base": {
                    "dims": voxel_dims,
                    "pose": object_pose_inverse,
                    "voxel_size": voxel_size,
                },
            }
        }
        
        # Create world collision configuration
        world_collision_config = WorldCollisionConfig.load_from_dict(
            {
                "checker_type": CollisionCheckerType.VOXEL,
                "max_distance": 5.0,
                "n_envs": 1,
            },
            world_model,
            self.tensor_args,
        )
                
        # Initialize collision checkers
        world_voxel_collision = WorldVoxelCollision(world_collision_config)
        
        # Create collision support world
        collision_support_world = WorldConfig.create_collision_support_world(
            self.collision
        )
        world_config = WorldCollisionConfig(
            self.tensor_args, world_model=collision_support_world
        )
        world_mesh_collision = WorldMeshCollision(world_config)
        
        # Generate ESDF
        voxel_grid = world_voxel_collision.get_voxel_grid("base")
        self.esdf = world_mesh_collision.get_esdf_in_bounding_box(
            Cuboid(name="base", pose=voxel_grid.pose, dims=voxel_grid.dims),
            voxel_size=voxel_grid.voxel_size,
        )
        
        # Create expanded object cuboid for collision-free space (following test_plan.py)
        expanded_dim = np.array([expanded_dim] * 3)
        expanded_dims = (np.array(self.object_cfg["dimensions"]) + expanded_dim).tolist()
        
        self.expanded_object_cuboid = Cuboid(
            name=self.object_cfg["asset_type"] + self.object_cfg["asset_id"] + "_expanded",
            pose=self.object_pose,
            dims=expanded_dims,
            tensor_args=self.tensor_args,
        )
        # visualize_voxel_grid_with_cuboid(self.expanded_object_cuboid, self.esdf)
        # Mark cuboid as empty space
        self.esdf = mark_cuboid_as_empty(self.esdf, self.expanded_object_cuboid)
        
        # visualize_voxel_grid_with_cuboid(self.expanded_object_cuboid, self.esdf)
        
        # Initialize collision checkers for IK and trajectory planning
        self._init_collision_checkers(world_collision_config, disable_collision=DISABLE_COLLISION)
    
    def _init_collision_checkers(self, world_collision_config, disable_collision=False):
        """Initialize collision checkers for IK and trajectory planning"""
        self.world_voxel_collision_ik = WorldVoxelCollision(world_collision_config)
        self.world_voxel_collision_traj = WorldVoxelCollision(world_collision_config)
        
        if not disable_collision:
            # Update collision checkers with ESDF data
            for collision_checker in [self.world_voxel_collision_ik, self.world_voxel_collision_traj]:
                collision_checker.clear_voxelization_cache()
                collision_checker.clear_cache()
                collision_checker.update_voxel_data(self.esdf)
                torch.cuda.synchronize()
    
    def load_robot(self, robot_cfg_path: Optional[Union[str, Dict]]) -> Dict[str, Any]:
        """
        Load robot configuration
        
        Args:
            robot_cfg_path: Path to robot configuration 
            
        Returns:
            Robot configuration dictionary
        """
        if isinstance(robot_cfg_path, str):
            robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]
        else:
            robot_cfg = robot_cfg_path

        print("Robot configuration loaded successfully")
        return robot_cfg
    
    def load_scene(self, scene_cfg: Optional[Dict[str, Any]] = None) -> None:
        """
        Load or reload scene configuration
        
        Args:
            scene_cfg: New scene configuration. If None, uses current scene_cfg
        """
        if scene_cfg:
            self.scene_cfg = scene_cfg
        
        self._load_scene()
        print("Scene reloaded successfully")
    
    def load_object(self, object_cfg: Optional[Dict[str, Any]] = None) -> None:
        """
        Load or reload object configuration
        
        Args:
            object_cfg: New object configuration. If None, uses current object_cfg
        """
        if object_cfg:
            self.object_cfg = object_cfg
        
        self._load_object()
        print("Object reloaded successfully")

    def _init_motion_gen(self, robot_cfg: Dict) -> MotionGen:
        """Initialize motion generator for standard robot"""
        if robot_cfg is None:
            raise ValueError("Robot configuration path is required")
        
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            WorldConfig(),
            self.tensor_args,
            world_coll_checker=self.world_voxel_collision_ik,
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
        
        motion_gen = MotionGen(motion_gen_config)
        print("Motion generator initialized")
        return motion_gen

    def _init_motion_gen_akr(self, akr_robot_cfg: Dict) -> MotionGen:
        """Initialize motion generator for AKR robot"""
        if akr_robot_cfg is None:
            raise ValueError("AKR robot configuration path is required")
        
        motion_gen_config_akr = MotionGenConfig.load_from_robot_config(
            akr_robot_cfg,
            WorldConfig(),
            self.tensor_args,
            world_coll_checker=self.world_voxel_collision_traj,
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
        
        motion_gen_akr = MotionGen(motion_gen_config_akr)
        print("AKR motion generator initialized")
        return motion_gen_akr

    def _update_world_collision(self, motion_gen: MotionGen, joint_cfg: Dict[str, float], disable_collision: bool = False):
        """Update world collision with current object state"""
        if disable_collision:
            return
        
        if joint_cfg is not None:
            self.object_urdf.update_cfg(joint_cfg)
        
        object_trimesh = self.object_urdf.scene.to_mesh()
        
        object_name = "target_object"
        object_mesh = Mesh(
            trimesh=object_trimesh, 
            name=object_name, 
            pose=self.object_pose
        )
        
        # Add object mesh to stage
        if self.usd_helper.stage.GetPrimAtPath("/World/object"):
            self.usd_helper.stage.RemovePrim("/World/object")
        
        self.usd_helper.add_mesh_to_stage(object_mesh, "/World/object")
        object_meshes = self.usd_helper.get_obstacles_from_stage(
            only_paths=["/World/object"]
        ).get_collision_check_world().mesh
        
        world_collision = WorldConfig(
            mesh=object_meshes,
            voxel=[self.esdf],
        )
        motion_gen.update_world(world_collision)
        self._debug_collision(self.esdf, object_meshes)
        
    def _debug_collision(self, esdf, object_meshes):
        expanded_dim = 0.2
        expanded_dim = np.array([expanded_dim] * 3)
        expanded_dims = (np.array(self.object_cfg["dimensions"]) + expanded_dim).tolist()
        
        expanded_object_cuboid = Cuboid(
            name=self.object_cfg["asset_type"] + self.object_cfg["asset_id"] + "_expanded",
            pose=self.object_pose,
            dims=expanded_dims,
            tensor_args=self.tensor_args,
        )
        # visualize_voxel_grid_with_cuboid(expanded_object_cuboid, esdf, mesh_obstacle=object_meshes[0])
    
    def _solve_ik(
        self, 
        motion_gen: MotionGen, 
        goal_pose: Pose, 
        retract_config: torch.Tensor,
        num_seeds: int = 20000
    ) -> torch.Tensor:
        """Solve inverse kinematics for a given goal pose"""
        result = motion_gen.ik_solver.solve_single(
            goal_pose=goal_pose,
            retract_config=self.tensor_args.to_device(retract_config).unsqueeze(0),
            return_seeds=num_seeds,
            num_seeds=num_seeds,
            link_poses=None,
        ).get_unique_solution()
        
        return result

    def get_open_ee_pose(
        self, 
        object_pose: Pose, 
        grasp_pose: Pose, 
        object_urdf: URDF, 
        handle_link: str, 
        joint_cfg: Dict[str, float], 
        default_joint_cfg: Dict[str, float]
    ) -> Pose:
        """
        Calculate the end-effector pose for a given joint configuration.
        
        This function implements the complete transform chain:
        T_world_ee = T_world_object * T_object_handle * T_handle_ee
        
        Args:
            object_pose: Pose of object in world coordinates
            grasp_pose: Grasp pose (end-effector relative to object base)
            object_urdf: URDF model of the articulated object
            handle_link: Name of the handle link
            joint_cfg: Joint configuration for the articulated object
            default_joint_cfg: Default joint configuration
            
        Returns:
            End-effector pose in world coordinates
        """
        # 1. Update object URDF with joint configuration
        object_urdf.update_cfg(joint_cfg)
        
        # 2. Get handle pose relative to object base
        handle_pose = Pose.from_matrix(
            object_urdf.get_transform(handle_link, "world")
        )
        
        # 3. Calculate grasp pose relative to handle
        # T_handle_grasp = T_base_handle^-1 * T_base_grasp  
        handle_pose_default = Pose.from_matrix(
            object_urdf.get_transform(handle_link, "world")
        )
        
        # Reset to default configuration to get base grasp relationship
        object_urdf.update_cfg(default_joint_cfg)
        handle_pose_default = Pose.from_matrix(
            object_urdf.get_transform(handle_link, "world")  
        )
        
        # Calculate grasp relative to handle in default configuration
        grasp_handle_pose = handle_pose_default.inverse().multiply(grasp_pose)
        
        # 4. Update to target joint configuration and get new handle pose
        object_urdf.update_cfg(joint_cfg)
        handle_pose_new = Pose.from_matrix(
            object_urdf.get_transform(handle_link, "world")
        )
        
        # 5. Calculate new grasp pose relative to base
        # T_base_grasp_new = T_base_handle_new * T_handle_grasp
        grasp_pose_new = handle_pose_new.multiply(grasp_handle_pose)
        
        # 6. Transform to world coordinates
        # T_world_grasp = T_world_object * T_object_grasp
        world_grasp_pose = object_pose.multiply(grasp_pose_new)
        
        return world_grasp_pose

    def plan_ik(
        self, 
        grasp_pose: np.ndarray,
        start_angle: float = 0.0,
        goal_angle: float = 1.57,
        robot_cfg: Optional[Union[str, Dict]] = None,
        clustering_params: Optional[Dict[str, int]] = None,
        handle_link: str = "link_0"
    ) -> IKResult:
        """
        Plan inverse kinematics for a single grasp pose
        
        Args:
            grasp_pose: Single grasp pose array (7D: x, y, z, qw, qx, qy, qz)
            start_angle: Starting joint angle for articulated object
            goal_angle: Goal joint angle for articulated object  
            robot_cfg: Path to robot config file or robot config dict
            clustering_params: Parameters for IK clustering
            handle_link: Name of the handle link in the URDF
            
        Returns:
            IKResult containing (start_iks, goal_iks) tensors
        """
        print("Starting IK planning...")
                        
        # Ensure robot_cfg is a dictionary
        robot_cfg = self.load_robot(robot_cfg)
        
        # Initialize motion generator
        if self.motion_gen is None:
            self.motion_gen = self._init_motion_gen(robot_cfg)
        
        # Set up clustering parameters
        if clustering_params is None:
            clustering_params = {
                "ap_fallback_clusters": 30,
                "ap_clusters_upperbound": 80,
                "ap_clusters_lowerbound": 10
            }
            # clustering_params = {
            #     "ap_fallback_clusters": 300,
            #     "ap_clusters_upperbound": 300,
            #     "ap_clusters_lowerbound": 100
            # }
        
        # Convert grasp pose to Pose object
        grasp_Pose = Pose.from_list(grasp_pose)
        object_Pose = Pose.from_list(self.object_pose)
        print("object_pose: ", self.object_pose)

        # Get robot retract config
        retract_config = self.tensor_args.to_device(
            robot_cfg["kinematics"]["cspace"]["retract_config"]
        )
        
        # Define joint configurations
        start_joint_cfg = {"joint_0": start_angle}
        goal_joint_cfg = {"joint_0": goal_angle}
        default_joint_cfg = {"joint_0": start_angle}
        
        # Plan start IK (closed state)
        print("Planning start IKs (closed state)...")
        self._update_world_collision(self.motion_gen, start_joint_cfg, disable_collision=DISABLE_COLLISION)
        
        start_grasp_Pose = self.get_open_ee_pose(
            object_Pose, grasp_Pose, self.object_urdf, 
            handle_link, start_joint_cfg, default_joint_cfg
        )
        print("start_grasp_Pose: ", start_grasp_Pose.to_list())
        
        start_iks = self._solve_ik(self.motion_gen, start_grasp_Pose, retract_config)
        # Check for empty IK solutions
        if start_iks is None or start_iks.shape[0] == 0:
            print("No IK solutions found, returning fallback None")
            return self.ik_fallback()
        print(f"Start IKs before clustering: {start_iks.shape}")
        
        start_iks = ik_clustering(start_iks, **clustering_params)
        print(f"Start IKs after clustering: {start_iks.shape}")
        
        # Plan goal IK (open state)  
        print("Planning goal IKs (open state)...")
        self._update_world_collision(self.motion_gen, goal_joint_cfg, disable_collision=DISABLE_COLLISION)
        
        goal_grasp_Pose = self.get_open_ee_pose(
            object_Pose, grasp_Pose, self.object_urdf, 
            handle_link, goal_joint_cfg, default_joint_cfg
        )
        print("goal_grasp_Pose: ", goal_grasp_Pose.to_list())
        
        goal_iks = self._solve_ik(self.motion_gen, goal_grasp_Pose, retract_config)
        # Check for empty IK solutions
        if goal_iks is None or goal_iks.shape[0] == 0:
            print("No IK solutions found, returning fallback None")
            return self.ik_fallback()
        print(f"Goal IKs before clustering: {goal_iks.shape}")
        
        goal_iks = ik_clustering(goal_iks, **clustering_params)
        print(f"Goal IKs after clustering: {goal_iks.shape}")
        
        # Add joint angles to IK solutions
        start_iks = self._stack_angle(start_iks, start_angle)
        goal_iks = self._stack_angle(goal_iks, goal_angle)
        
        print("IK planning completed successfully")
        return IKResult(start_ik=start_iks, goal_ik=goal_iks)
    
    def _stack_angle(self, iks: torch.Tensor, angle: float) -> torch.Tensor:
        """Add joint angle information to IK solutions"""
        joint_angle_expand = (
            torch.tensor([angle], device=iks.device)
            .unsqueeze(0)
            .expand(iks.shape[0], -1)
        )
        return torch.cat((iks, joint_angle_expand), dim=1)
    
    def _process_for_akr(self, iks: torch.Tensor) -> torch.Tensor:
        """Process IK solutions for AKR robot (handle negative angles)"""
        if iks[:, -1].min() > 0:
            iks[:, -1] *= -1
        return iks
    
    def plan_traj(
        self,
        ik_result: IKResult,
        akr_robot_cfg: Union[str, Dict],
        batch_size: int = 40
    ) -> TrajResult:
        """
        Plan trajectories between start and goal IK solutions
        
        Args:
            ik_result: IKResult containing start and goal IK solutions
            akr_robot_cfg: Path to AKR robot configuration or config dict
            batch_size: Batch size for trajectory planning
            
        Returns:
            TrajResult containing (start_states, goal_states, trajectories, success_flags)
        """
        print("Starting trajectory planning...")

        # Ensure akr_robot_cfg is a dictionary
        akr_robot_cfg = self.load_robot(akr_robot_cfg)
        
        # Initialize AKR motion generator
        if self.motion_gen_akr is None:
            self.motion_gen_akr = self._init_motion_gen_akr(akr_robot_cfg)
            
        if ik_result.start_ik.shape[0] == 0 or ik_result.goal_ik.shape[0] == 0:
            print("No IK solutions provided, returning empty trajectory result")
            return self.traj_fallback()

        # Extract IK solutions
        start_iks = ik_result.start_ik
        goal_iks = ik_result.goal_ik
        
        # Process IK solutions for AKR robot
        start_iks = self._process_for_akr(start_iks.clone())
        goal_iks = self._process_for_akr(goal_iks.clone())
        
        # Prepare IK data for batch processing
        goal_obj_iks_expanded = goal_iks.repeat(start_iks.shape[0], 1).clone()
        start_obj_iks_expanded = torch.repeat_interleave(
            start_iks, goal_iks.shape[0], dim=0
        ).clone()
        
        goal_iks = goal_obj_iks_expanded
        start_iks = start_obj_iks_expanded
        
        # Set up batch processing
        n_iters = int(np.ceil(start_iks.shape[0] / batch_size))
        all_idxs = np.arange(start_iks.shape[0])
        idxs_list = []
        
        for i in range(n_iters):
            idxs_list.append(
                all_idxs[i * batch_size : min((i + 1) * batch_size, start_iks.shape[0])]
            )
        
        # Process each batch
        all_traj_result = []
        all_goal_state = []
        all_start_state = []
        all_success = []
        
        with tqdm(range(n_iters), desc="Trajectory Planning") as traj_tqdm:
            for i in traj_tqdm:
                idxs = idxs_list[i]
                batch_start_iks = start_iks[idxs]
                batch_goal_iks = goal_iks[idxs]
                
                start_state = JointState.from_position(
                    self.tensor_args.to_device(batch_start_iks)
                )
                goal_state = JointState.from_position(
                    self.tensor_args.to_device(batch_goal_iks)
                )
                
                goal_pose = self.motion_gen_akr.ik_solver.fk(goal_state.position).ee_pose
                goal = Goal(
                    goal_pose=goal_pose,
                    goal_state=goal_state,
                    current_state=start_state,
                )
                
                torch.cuda.synchronize()
                result = self.motion_gen_akr.trajopt_solver.solve_batch(goal)
                
                all_success.append(result.success.detach().clone().cpu())
                all_goal_state.append(goal_state.position.detach().clone().cpu())
                all_start_state.append(start_state.position.detach().clone().cpu())
                
                if result.solution.position.dim() == 2:
                    all_traj_result.append(
                        result.solution.position.unsqueeze(0).detach().clone().cpu()
                    )
                else:
                    all_traj_result.append(
                        result.solution.position.detach().clone().cpu()
                    )
                
                torch.cuda.synchronize()
                
                # Update progress
                success_count = torch.cat(all_success).sum().item()
                traj_tqdm.set_description(
                    f"Batch {i+1}/{n_iters} (Successes: {success_count})"
                )
                
        if len(all_traj_result) == 0:
            print("No trajectories planned, returning empty result")
            return self.traj_fallback()
        
        # Concatenate results
        all_traj_result = torch.cat(all_traj_result, dim=0)
        all_goal_state = torch.cat(all_goal_state, dim=0)
        all_start_state = torch.cat(all_start_state, dim=0)
        all_success = torch.cat(all_success, dim=0).to(torch.bool)
        
        print(f"Trajectory planning completed:")
        print(f"  Trajectories: {all_traj_result.shape}")
        print(f"  Success rate: {all_success.sum().item()}/{len(all_success)}")
        
        return TrajResult(
            start_states=all_start_state,
            goal_states=all_goal_state, 
            trajectories=all_traj_result,
            success=all_success
        )
        
    def ik_fallback(self) -> IKResult:
        """Return fallback IKResult with empty tensors"""
        dof = self.motion_gen.dof+1
        return IKResult(
            start_ik=torch.zeros((0, dof)),
            goal_ik=torch.zeros((0, dof))
        )
        
    def traj_fallback(self) -> TrajResult:
        """Return fallback TrajResult with empty tensors"""
        dof = self.motion_gen_akr.dof
        return TrajResult(
            start_states=torch.zeros((0, dof)),
            goal_states=torch.zeros((0, dof)),
            trajectories=torch.zeros((0, 0, dof)),
            success=torch.zeros((0,), dtype=torch.bool)
        )

    def save_ik(self, ik_result: IKResult, path: str) -> None:
        """Save IK data to file"""
        ik_data = {
            "start_iks": ik_result.start_ik.cpu(),
            "goal_iks": ik_result.goal_ik.cpu(),
        }
        torch.save(ik_data, path)
        print(f"IK data saved to {path}")
    
    def load_ik(self, path: str) -> IKResult:
        """Load IK data from file"""
        ik_data = torch.load(path, weights_only=False)
        start_iks = ik_data["start_iks"]
        goal_iks = ik_data["goal_iks"]
        print(f"IK data loaded from {path}")
        return IKResult(start_ik=start_iks, goal_ik=goal_iks)
    
    def save_traj(self, traj_result: TrajResult, path: str) -> None:
        """Save trajectory data to file"""
        traj_data = {
            "start_state": traj_result.start_states.cpu(),
            "goal_state": traj_result.goal_states.cpu(),
            "traj": traj_result.trajectories.cpu(),
            "success": traj_result.success.cpu(),
        }
        torch.save(traj_data, path)
        print(f"Trajectory data saved to {path}")
    
    def load_traj(self, path: str) -> TrajResult:
        """Load trajectory data from file"""
        traj_data = torch.load(path, weights_only=False)
        start_state = traj_data["start_state"]
        goal_state = traj_data["goal_state"]
        traj = traj_data["traj"]
        success = traj_data["success"]
        print(f"Trajectory data loaded from {path}")
        return TrajResult(
            start_states=start_state,
            goal_states=goal_state,
            trajectories=traj,
            success=success
        )
    
    def get_grasp_poses(
        self, 
        grasp_dir: str, 
        num_grasps: int = 10, 
        scaling_factor: float = 1.0
    ) -> List[np.ndarray]:
        """
        Load and scale grasp poses from directory
        
        Args:
            grasp_dir: Directory containing grasp pose files
            num_grasps: Number of grasp poses to load
            scaling_factor: Scaling factor for grasp poses
            
        Returns:
            List of scaled grasp poses
        """
        def grasp_scale(grasp: np.ndarray, scale: float) -> np.ndarray:
            scaled_grasp = np.copy(grasp)
            scaled_grasp[:3] *= scale
            return scaled_grasp
        
        grasp_poses = []
        for i in range(num_grasps):
            grasp_file = f"{grasp_dir}/{i:04d}.npy"
            if os.path.exists(grasp_file):
                grasp_pose = np.load(grasp_file)
                grasp_poses.append(grasp_scale(grasp_pose, scaling_factor))
            else:
                print(f"Warning: Grasp file {grasp_file} not found")
        
        print(f"Loaded {len(grasp_poses)} grasp poses")
        return grasp_poses
    
    def _akr_fk_filter(self, js: JointState, pose: Pose, 
                      position_tolerance: float = 0.01, 
                      rotation_tolerance: float = 0.01) -> bool:
        """
        Check if the forward kinematics of the joint state is within the specified pose.
        
        Args:
            js: JointState object representing the joint angles
            pose: Pose object representing the target pose
            
        Returns:
            bool: True if the FK is within the specified pose, False otherwise
        """
        # Use provided tolerance parameters
        position_norm = position_tolerance
        rotation_norm = rotation_tolerance
        
        # Perform FK calculation using AKR motion generator
        if self.motion_gen_akr is None:
            raise ValueError("AKR motion generator not initialized. Call plan_traj first.")
            
        fk_result = self.motion_gen_akr.ik_solver.fk(js.position)
        
        # Move tensors to CPU if they are on GPU and convert to numpy
        pose_position_cpu = (
            pose.position.cpu().numpy()
            if pose.position.is_cuda
            else pose.position.numpy()
        )
        pose_quaternion_cpu = (
            pose.quaternion.cpu().numpy()
            if pose.quaternion.is_cuda
            else pose.quaternion.numpy()
        )
        fk_position_cpu = (
            fk_result.ee_pose.position.cpu().numpy()
            if fk_result.ee_pose.position.is_cuda
            else fk_result.ee_pose.position.numpy()
        )
        fk_quaternion_cpu = (
            fk_result.ee_pose.quaternion.cpu().numpy()
            if fk_result.ee_pose.quaternion.is_cuda
            else fk_result.ee_pose.quaternion.numpy()
        )
        
        # Check if the position difference is within the specified norm (tolerance)
        position_diff = np.linalg.norm(pose_position_cpu - fk_position_cpu)
        
        # Check if the angular distance between the quaternions is within the specified norm (tolerance)
        from cuakr.utils.math import quaternion_distance
        rotation_diff = quaternion_distance(pose_quaternion_cpu, fk_quaternion_cpu)
        
        # Store differences for statistics
        if hasattr(self, 'position_diff_list'):
            self.position_diff_list.append(position_diff)
        if hasattr(self, 'rotation_diff_list'):
            self.rotation_diff_list.append(rotation_diff)
        
        # Compare position and orientation against tolerance thresholds
        if position_diff < position_norm and rotation_diff < rotation_norm:
            return True
        return False
    
    def traj_filter(self, traj_result: TrajResult, akr_robot_cfg: Union[str, Dict], 
                   position_tolerance: float = 0.01, rotation_tolerance: float = 0.01) -> TrajResult:
        """
        Filter trajectory data based on success and forward kinematics validation.
        
        Args:
            traj_result: TrajResult containing trajectory data to filter
            akr_robot_cfg: Path to AKR robot configuration or config dict for FK validation
            position_tolerance: Position tolerance for FK validation
            rotation_tolerance: Rotation tolerance for FK validation (radians)
            
        Returns:
            TrajResult: Filtered trajectory result
        """
        print(f"Starting trajectory filtering...")
        
        if traj_result.trajectories.shape[0] == 0:
            print("No trajectory data provided, returning empty result")
            return self.traj_fallback()
        
        
        akr_robot_cfg = self.load_robot(akr_robot_cfg)
        
        # Extract data from TrajResult
        start_state = traj_result.start_states
        goal_state = traj_result.goal_states
        trajectories = traj_result.trajectories
        success = traj_result.success
        
        indices_count = goal_state.shape[0]
        print(f"Loaded {indices_count} trajectories")
        
        # Step 1: Filter the trajectories based on success
        filtered_indices = success.nonzero(as_tuple=True)[0]
        
        if filtered_indices.shape[0] == 0:
            print("Step 1: No successful trajectories to filter, returning empty result")
            return self.traj_fallback()
        
        goal_state = goal_state[filtered_indices]
        start_state = start_state[filtered_indices]
        trajectories = trajectories[filtered_indices]
        success = success[filtered_indices]
        
        step_1_count = goal_state.shape[0]
        print(f"Step 1: Filtered from {indices_count} to {step_1_count} trajectories based on success.")
        
        
        # Initialize AKR motion generator for FK validation
        if self.motion_gen_akr is None:
            self.motion_gen_akr = self._init_motion_gen_akr(akr_robot_cfg)

        # Step 2: Filter the trajectories based on distance of object pose
        # FK calculation
        fk_succ_indices = []
        
        self.position_diff_list = []
        self.rotation_diff_list = []
        
        for i in tqdm(range(goal_state.shape[0]), desc="FK Filtering"):
            # Get start and goal pose
            start_pose = JointState.from_position(
                self.tensor_args.to_device(start_state[i])
            )
            goal_pose = JointState.from_position(
                self.tensor_args.to_device(goal_state[i])
            )
            
            # Get goal object pose for validation
            goal_object_pose = self.motion_gen_akr.ik_solver.fk(
                goal_pose.position
            ).ee_pose
            
            # IK should make sure the object pose is the same between start and goal
            object_pose = goal_object_pose
            
            # Check each waypoint in the trajectory
            trajectory_valid = True
            for j in range(trajectories.shape[1]):
                traj_pose = JointState.from_position(
                    self.tensor_args.to_device(trajectories[i][j])
                )
                if not self._akr_fk_filter(js=traj_pose, pose=object_pose, 
                                          position_tolerance=position_tolerance, 
                                          rotation_tolerance=rotation_tolerance):
                    trajectory_valid = False
                    break
            
            if trajectory_valid:
                fk_succ_indices.append(i)
        
        # Print basic statistics about position and rotation differences
        if self.position_diff_list:
            print(
                f"Position Difference - Mean: {np.mean(self.position_diff_list):.4f}, "
                f"Max: {np.max(self.position_diff_list):.4f}, "
                f"Min: {np.min(self.position_diff_list):.4f}"
            )
        if self.rotation_diff_list:
            print(
                f"Rotation Difference - Mean: {np.mean(self.rotation_diff_list):.4f}, "
                f"Max: {np.max(self.rotation_diff_list):.4f}, "
                f"Min: {np.min(self.rotation_diff_list):.4f}"
            )
        
        print(f"FK success indices: {len(fk_succ_indices)}")
        
        filtered_indices = torch.tensor(fk_succ_indices, device=goal_state.device)
        
        if filtered_indices.shape[0] == 0:
            print("Step 2: No successful trajectories to filter, returning empty result")
            return self.traj_fallback()
        
        goal_state = goal_state[filtered_indices]
        start_state = start_state[filtered_indices]
        trajectories = trajectories[filtered_indices]
        success = success[filtered_indices]
        
        step_2_count = goal_state.shape[0]
        print(f"Step 2: Filtered from {step_1_count} to {step_2_count} trajectories based on FK filtering.")
        
        print(f"Final count: {step_2_count} trajectories")
        
        # Return filtered result as TrajResult
        return TrajResult(
            start_states=start_state,
            goal_states=goal_state,
            trajectories=trajectories,
            success=success
        )
    
    @staticmethod
    def _load_object_metadata(metadata_path: str, object_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Load object metadata from JSON file and find target object"""
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        objects = metadata["static_objects"]
        
        # Find target object (following test_plan.py pattern)
        object_info = None
        for value in objects.values():
            if (value.get("asset_type") == object_cfg.get("asset_type", "Microwave") and 
                value.get("asset_id") == object_cfg.get("asset_id", "7221")):
                object_info = value
                break
        
        if object_info is None:
            raise ValueError(f"Object with asset_type={object_cfg.get('asset_type')} and asset_id={object_cfg.get('asset_id')} not found in metadata")
        
        return object_info

    @staticmethod
    def _process_object_pose(object_matrix: np.ndarray) -> List[float]:
        """
        Process object pose from matrix format with rotations
        
        Args:
            object_matrix: 4x4 transformation matrix
            
        Returns:
            7D pose list [x, y, z, qw, qx, qy, qz]
        """
        from automoma.utils.transform import matrix_to_pose, quat_to_euler, single_axis_self_rotation
        
        object_pose = np.array(object_matrix)
        print("### Before rotation ###")
        print("object_pose matrix:\n", object_pose)
        print("object_pose 7D: ", matrix_to_pose(object_pose))
        print("object_euler: ", quat_to_euler(matrix_to_pose(object_pose)[3:], order='xyz'))

        # Apply rotation (from test_plan.py)
        object_pose = single_axis_self_rotation(object_pose, 'z', np.pi)
        print("### After rotation ###")
        print("object_pose matrix:\n", object_pose)
        print("object_pose 7D: ", matrix_to_pose(object_pose))
        print("object_euler: ", quat_to_euler(matrix_to_pose(object_pose)[3:], order='xyz'))

        # Convert to 7D pose
        object_pose_7d = matrix_to_pose(object_pose).tolist()
        
        return object_pose_7d

    @staticmethod
    def load_object_from_metadata(metadata_path: str, object_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load object configuration from scene metadata following test_plan.py pattern
        
        Args:
            metadata_path: Path to metadata JSON file
            object_cfg: Object configuration
        """
        # Load object metadata
        object_metadata = AKRPlanner._load_object_metadata(metadata_path, object_cfg)
        
        # Process object pose (following test_plan.py exactly)
        processed_pose = AKRPlanner._process_object_pose(object_metadata["matrix"])
        
        # Update object configuration with all metadata information
        object_cfg.update({
            "name": object_metadata["name"],
            "asset_type": object_metadata.get("asset_type", "Microwave"),
            "asset_id": object_metadata.get("asset_id", "7221"),
            "dimensions": object_metadata["dimensions"],  # Load from metadata
            "pose": processed_pose
        })
        print(f"Object loaded from metadata: {object_metadata['name']} with dimensions {object_metadata['dimensions']}")
        
        return object_cfg
def main():
    """
    Main pipeline function with trajectory filtering
    Complete AKR motion planning pipeline from scene to trajectory with filtering
    """
    import os
    import json
    import numpy as np
    import torch
        
    print("=== AKR Motion Planning Pipeline with Filtering ===")
    
    # ===== CONFIGURATION =====
    # Scene configuration
    scene_cfg = {
        "path": "output/infinigen_scene_10/scene_6_seed_6/export/export_scene.blend/export_scene.usdc",
        "pose": [0, 0, -0.12, 1, 0, 0, 0]  # for infinigen: lower -0.12 z-axis
    }
    
    # Robot configuration  
    robot_cfg_path = "assets/robot/summit_franka/summit_franka.yml"
    from automoma.utils.file import process_robot_cfg
    
    robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]
    robot_cfg = process_robot_cfg(robot_cfg)
        
    # Object configuration (URDF path and basic info, dimensions loaded from metadata)
    object_cfg = {
        "path": "third_party/cuakr/tests/7221/7221_0_scaling.urdf",
        "asset_type": "Microwave",
        "asset_id": "7221"
    }
    metadata_path = "output/test/kitchen_0919/scene_6_seed_6/info/metadata.json"
    object_cfg = AKRPlanner.load_object_from_metadata(metadata_path, object_cfg=object_cfg)
    
    # ===== INITIALIZATION =====
    print("\n1. Initializing AKR Planner...")
    planner = AKRPlanner(scene_cfg, object_cfg, robot_cfg)
    
    # ===== LOAD OBJECT FROM METADATA =====  
    print("\n2. Loading object from metadata...")
    
    # ===== LOAD GRASP POSES =====
    print("\n3. Loading grasp poses...")
    scaling_factor = 0.3562990018302636
    grasp_poses = planner.get_grasp_poses(
        grasp_dir="assets/object/Microwave/7221/grasp",
        num_grasps=20,
        scaling_factor=scaling_factor
    )
    
    if not grasp_poses:
        print("ERROR: No grasp poses found!")
        return
        
    print(f"Loaded {len(grasp_poses)} grasp poses")
    
    # ===== IK PLANNING =====
    print("\n4. Planning IK solutions...")
    start_angle = 0.0
    goal_angle = 1.57
    
    ik_result = planner.plan_ik(
        grasp_pose=grasp_poses[0],  # Use first grasp pose
        start_angle=start_angle,
        goal_angle=goal_angle,
        robot_cfg=robot_cfg,
        handle_link="link_0"
    )
    
    print(f"IK Planning completed:")
    print(f"  Start IKs: {ik_result.start_ik.shape}")  
    print(f"  Goal IKs: {ik_result.goal_ik.shape}")
    
    # ===== SAVE IK RESULTS =====
    print("\n5. Saving IK results...")
    output_dir = "third_party/cuakr/tests/output"
    os.makedirs(output_dir, exist_ok=True)
    planner.save_ik(ik_result, f"{output_dir}/ik_data.pt")
    
    # ===== TRAJECTORY PLANNING =====
    print("\n6. Planning trajectories...")
    akr_robot_cfg_path = "assets/object/Microwave/7221/summit_franka_7221_0_grasp_0000.yml"
    
    akr_robot_cfg = load_yaml(akr_robot_cfg_path)["robot_cfg"]
    akr_robot_cfg = process_robot_cfg(akr_robot_cfg)
    
    traj_result = planner.plan_traj(
        ik_result, akr_robot_cfg, batch_size=40
    )
    
    print(f"Trajectory Planning completed:")
    print(f"  Start states: {traj_result.start_states.shape}")
    print(f"  Goal states: {traj_result.goal_states.shape}") 
    print(f"  Trajectories: {traj_result.trajectories.shape}")
    print(f"  Success rate: {traj_result.success.sum().item()}/{len(traj_result.success)} ({100*traj_result.success.sum().item()/len(traj_result.success):.1f}%)")
    
    # ===== SAVE TRAJECTORY RESULTS =====
    print("\n7. Saving trajectory results...")
    traj_path = f"{output_dir}/traj_data.pt"
    planner.save_traj(traj_result, traj_path)
    
    # ===== TRAJECTORY FILTERING =====
    print("\n8. Filtering trajectories...")
    filtered_traj_path = f"{output_dir}/traj_data_filtered.pt"
    filtered_result = planner.traj_filter(traj_result, akr_robot_cfg_path)
    planner.save_traj(filtered_result, filtered_traj_path)
    
    print(f"\n=== PIPELINE WITH FILTERING COMPLETED SUCCESSFULLY ===")
    print(f"Results saved to: {output_dir}")
    print(f"- IK data: ik_data.pt")
    print(f"- Original trajectory data: traj_data.pt")
    print(f"- Filtered trajectory data: traj_data_filtered.pt")



if __name__ == "__main__":
    main()