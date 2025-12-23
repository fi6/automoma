import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union


# Third party
from curobo.util.usd_helper import UsdHelper, set_prim_transform
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from curobo.geom.types import WorldConfig, Cuboid, VoxelGrid, Mesh
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.sdf.world_voxel import WorldVoxelCollision
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollisionConfig
from curobo.rollout.rollout_base import Goal
from yourdfpy import URDF
from tqdm import tqdm
from dataclasses import dataclass

# Local imports
from automoma.core.types import TaskType, StageType, IKResult, TrajResult, PlanResult
from automoma.core.interfaces import MotionPlannerInterface
from automoma.utils.math_utils import (
    _convert_to_list,
    pose_multiply,
    quat_multiply,
    mark_cuboid_as_empty,
    ik_clustering_kmeans_ap_fallback,
    expand_to_pairs,
    quaternion_distance,
    get_open_ee_pose,
    stack_iks_angle,
)
from automoma.utils.file_utils import (
    load_robot_cfg,
    process_robot_cfg,
    save_ik,
    load_ik,
    save_traj,
    load_traj,
    get_grasp_poses,
    load_object_from_metadata,
)


class BasePlanner(MotionPlannerInterface):

    def init_motion_gen(self, robot_cfg: Union[str, Dict], fixed_base: bool=False) -> MotionGen:
        """Initialize motion generator for standard robot"""
        if robot_cfg is None:
            raise ValueError("Robot configuration path is required")
        
        robot_cfg = load_robot_cfg(robot_cfg)
        gradient_trajopt_file = "gradient_trajopt_fixbase.yml" if fixed_base else "gradient_trajopt.yml"
        
        # DEBUG: need new collision checker each time for different motion_gen

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            WorldConfig(),
            TensorDeviceType(),
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            interpolation_dt=0.05,
            collision_cache={"obb": 30, "mesh": 100},
            optimize_dt=True,
            trajopt_dt=None,
            trajopt_tsteps=32,
            trim_steps=None,
            use_cuda_graph=False,
            gradient_trajopt_file=gradient_trajopt_file,
        )

        self.motion_gen = MotionGen(motion_gen_config)
        print("Motion generator initialized")
        return self.motion_gen

class CuroboPlanner(MotionPlannerInterface):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        # Initialize core components
        self.tensor_args = TensorDeviceType()
        self.usd_helper = UsdHelper()

        self.planner_cfg = cfg

    def setup_env(self, scene_cfg: Dict[str, Any], object_cfg: Dict[str, Any]) -> None:
        self.scene_cfg = scene_cfg
        self.object_cfg = object_cfg
        self.root_pose = [0, 0, 0, 1, 0, 0, 0]
        self._init_root_pose()
        self._load_object()
        self._load_scene()
        self._setup_collision_world()

    def plan_ik(self, target_pose: torch.Tensor,
                robot_cfg: Dict[str, Any] = None,
                plan_cfg: Dict[str, Any] = None,
                motion_gen: MotionGen = None) -> IKResult:
        """Plan inverse kinematics to reach the target pose"""
        print("Starting IK planning...")
        
        if plan_cfg is None:
            plan_cfg = {}
        robot_cfg = self._load_robot(robot_cfg)
        
        # Initialize motion generator
        if motion_gen is None:
            motion_gen = self.init_motion_gen(robot_cfg)

        # Update world collision
        joint_cfg = plan_cfg.get("joint_cfg", None)
        enable_collision = plan_cfg.get("enable_collision", True)
        self._update_world_collision(motion_gen, joint_cfg, enable_collision)
        
        # Solve IK
        retract_config = self.tensor_args.to_device(
            robot_cfg["kinematics"]["cspace"]["retract_config"]
        )
        target_Pose = Pose.from_list(_convert_to_list(target_pose))
        iks = self._solve_ik(motion_gen, target_Pose, retract_config)
        
        # Check for empty IK solutions
        if iks is None or iks.shape[0] == 0:
            print("No IK solutions found, returning empty tensor.")
            iks = torch.zeros(0, motion_gen.dof)
        
        target_poses = target_pose.repeat(iks.shape[0], 1)
        
        return IKResult(target_poses=target_poses, iks=iks)
        
    def ik_clustering(self, ik_result: IKResult, **kwargs):
        '''
        Cluster IK solutions
        
        Args:
            ik_result: IKResult object containing IK solutions
            **kwargs: Additional clustering parameters
            
        Returns:
            Clustered IK solutions tensor
        '''
        if ik_result.iks is None or ik_result.iks.shape[0] == 0:
            print("No IK solutions to cluster, returning empty tensor.")
            return ik_result
        
        idxs = ik_clustering_kmeans_ap_fallback(ik_result.iks, **kwargs)[-1]
        return ik_result[idxs]
    
    def plan_traj(self, start_iks: torch.Tensor, goal_iks: torch.Tensor = None,
                 robot_cfg: Dict[str, Any] = None,
                 plan_cfg: Dict[str, Any] = None,
                 motion_gen: MotionGen = None) -> TrajResult:
        
        if plan_cfg is None:
                plan_cfg = {}
        
        # check goal_iks
        if goal_iks is not None:
            if plan_cfg.get("expand_to_pairs", False):
                start_iks, goal_iks = expand_to_pairs(start_iks, goal_iks)
            assert start_iks.shape[0] == goal_iks.shape[0], \
                "Start and goal IK solutions must have the same number of samples"
        
        # Initialize motion generator
        if motion_gen is None:
            motion_gen = self.init_motion_gen(robot_cfg)

        # Mapping stage type to planning function
        stage_type = plan_cfg.get("stage_type", StageType.MOVE)
        
        stage_mapping = {
            StageType.REACH: self.plan_traj_reach,
            StageType.GRASP: self.plan_traj_grasp,
            StageType.LIFT: self.plan_traj_lift,
            StageType.RELEASE: self.plan_traj_release,
            StageType.MOVE: self.plan_traj_move,
            StageType.MOVE_HOLDING: self.plan_traj_move_holding,
            StageType.MOVE_ARTICULATED: self.plan_traj_move_articulated,
            StageType.WAIT: self.plan_traj_wait,
            StageType.HOME: self.plan_traj_move,
        }

        plan_func = stage_mapping.get(stage_type)
        if plan_func is None:
            raise ValueError(f"Unsupported stage type: {stage_type}")
        
        print(f"Starting TrajOpt planning for stage: {stage_type}")
        return plan_func(start_iks, goal_iks, robot_cfg, plan_cfg, motion_gen)

    def filter_traj(
        self,
        traj_result: TrajResult,
        robot_cfg: Dict[str, Any] = None,
        motion_gen: MotionGen = None,
        filter_cfg: Dict[str, Any] = None,
    ) -> TrajResult:
        """
        Filter trajectories based on success and optional forward kinematics validation.
        
        Args:
            traj_result: The trajectory result to filter
            motion_gen: Optional motion generator for FK validation
            filter_cfg: Optional dictionary containing filter configuration parameters
        """
        if traj_result.trajectories is None or traj_result.trajectories.shape[0] == 0:
            print("No trajectories to filter.")
            return traj_result
            
        # Mapping stage type to planning function
        stage_type = filter_cfg.get("stage_type", StageType.MOVE)
        
        stage_mapping = {
            StageType.REACH: self.filter_traj_reach,
            StageType.GRASP: self.filter_traj_grasp,
            StageType.LIFT: self.filter_traj_lift,
            StageType.RELEASE: self.filter_traj_release,
            StageType.MOVE: self.filter_traj_move,
            StageType.MOVE_HOLDING: self.filter_traj_move_holding,
            StageType.MOVE_ARTICULATED: self.filter_traj_move_articulated,
            StageType.WAIT: self.filter_traj_wait,
            StageType.HOME: self.filter_traj_move,
        }

        filter_func = stage_mapping.get(stage_type)
        if filter_func is None:
            raise ValueError(f"Unsupported stage type: {stage_type}")
        
        print(f"Starting trajectory filtering for stage: {stage_type}")
        return filter_func(traj_result, robot_cfg, motion_gen, filter_cfg)

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
            trimesh=object_trimesh, name=object_name, pose=self.object_pose
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
            self.usd_helper.stage.GetPrimAtPath("/World/scene"), self.root_pose
        )

        # Get collision world from scene
        print("Getting collision world from scene...")
        self.collision = (
            self.usd_helper.get_obstacles_from_stage().get_collision_check_world()
        )
        print(f"Number of collision meshes: {len(self.collision.mesh)}")

        # Remove empty meshes
        self._clean_collision_meshes()
        
    def _load_robot(self, robot_cfg_path: Optional[Union[str, Dict]]) -> Dict[str, Any]:
        """Load robot configuration."""
        return load_robot_cfg(robot_cfg_path)

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

        print(
            f"Removed {len(to_remove)} empty meshes. Remaining meshes: {len(self.collision.mesh)}"
        )

    def _setup_collision_world(self, enable_collision: bool = True) -> None:
        """Setup collision world for planning"""
        object_Pose = Pose.from_list(self.object_pose)
        object_pose_inv = object_Pose.inverse().to_list()

        voxel_dims = self.planner_cfg.get("voxel_dims", [5.0, 5.0, 5.0])
        voxel_size = self.planner_cfg.get("voxel_size", 0.02)
        expanded_dims = self.planner_cfg.get("expanded_dims", [1.0, 0.2, 0.2])

        world_model = {
            "voxel": {
                "base": {
                    "dims": voxel_dims,
                    "pose": object_pose_inv,
                    "voxel_size": voxel_size,
                }
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
        
        # Disable collision
        if not enable_collision:
            self.world_collision_config = world_collision_config
            return

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
        expanded_dims = (
            np.array(self.object_cfg["dimensions"]) + expanded_dims
        ).tolist()

        self.expanded_object_cuboid = Cuboid(
            name=self.object_cfg["asset_type"]
            + self.object_cfg["asset_id"]
            + "_expanded",
            pose=self.object_pose,
            dims=expanded_dims,
            tensor_args=self.tensor_args,
        )
        # visualize_voxel_grid_with_cuboid(self.expanded_object_cuboid, self.esdf)
        # Mark cuboid as empty space
        self.esdf = mark_cuboid_as_empty(self.esdf, self.expanded_object_cuboid)
        # visualize_voxel_grid_with_cuboid(self.expanded_object_cuboid, self.esdf)

        self.collision_type = self.planner_cfg.get(
            "collision_checker_type", CollisionCheckerType.VOXEL
        )
        if self.collision_type == CollisionCheckerType.VOXEL:
            pass
        elif self.collision_type == CollisionCheckerType.MESH:
            pitch = voxel_size
            mesh = self.usd_helper.voxel_to_mesh(self.esdf, pitch=pitch)
            self.usd_helper.add_mesh_to_stage(mesh, "/World/esdf")
            # update mesh, clean the old voxel grid
            self.esdf = voxel_grid
            world_collision_config.world_model["mesh"] = [mesh]
        else:
            raise ValueError(f"Invalid collision checker type: {self.collision_type}")
        
        self.world_collision_config = world_collision_config

    def _update_world_collision(
        self,
        motion_gen: MotionGen,
        joint_cfg: Dict[str, float],
        enable_collision: True,
    ):
        """Update world collision with current object state"""
        if not enable_collision:
            return

        if joint_cfg is not None:
            self.object_urdf.update_cfg(joint_cfg)

        object_trimesh = self.object_urdf.scene.to_mesh()

        object_name = "target_object"
        object_mesh = Mesh(
            trimesh=object_trimesh, name=object_name, pose=self.object_pose
        )

        # Add object mesh to stage
        if self.usd_helper.stage.GetPrimAtPath("/World/object"):
            self.usd_helper.stage.RemovePrim("/World/object")

        self.usd_helper.add_mesh_to_stage(object_mesh, "/World/object")
        object_meshes = (
            self.usd_helper.get_obstacles_from_stage(only_paths=["/World/object"])
            .get_collision_check_world()
            .mesh
        )

        meshes = object_meshes
        if self.collision_type == CollisionCheckerType.MESH:
            env_meshes = (
                self.usd_helper.get_obstacles_from_stage(
                    only_paths=["/World/esdf"],
                )
                .get_collision_check_world()
                .mesh
            )
            meshes += env_meshes

        world_collision = WorldConfig(
            mesh=meshes,
            voxel=[self.esdf],
        )
        motion_gen.update_world(world_collision)
        # self._debug_collision(self.esdf, object_meshes)

    def _solve_ik(
        self,
        motion_gen: MotionGen,
        goal_pose: Pose,
        retract_config: torch.Tensor,
        num_seeds: int = 20000,
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
    
    def _init_collision_checkers(
        self, world_collision_config, enable_collision=True
    ) -> None:
        """Initialize collision checkers for IK and trajectory planning"""
        world_coll_checker = WorldVoxelCollision(world_collision_config)

        if enable_collision:
            world_coll_checker.clear_voxelization_cache()
            world_coll_checker.clear_cache()
            world_coll_checker.update_voxel_data(self.esdf)
            torch.cuda.synchronize()
            
        return world_coll_checker
    def init_motion_gen(self, robot_cfg: Dict, fixed_base: bool=False, enable_collision: bool=True) -> MotionGen:
        """Initialize motion generator for standard robot"""
        if robot_cfg is None:
            raise ValueError("Robot configuration path is required")
        
        robot_cfg = self._load_robot(robot_cfg)
        gradient_trajopt_file = "gradient_trajopt_fixbase.yml" if fixed_base else "gradient_trajopt.yml"
        
        # DEBUG: need new collision checker each time for different motion_gen
        world_coll_checker = self._init_collision_checkers(self.world_collision_config, enable_collision)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            WorldConfig(),
            self.tensor_args,
            world_coll_checker=world_coll_checker,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            interpolation_dt=0.05,
            collision_cache={"obb": 30, "mesh": 100},
            optimize_dt=True,
            trajopt_dt=None,
            trajopt_tsteps=32,
            trim_steps=None,
            use_cuda_graph=False,
            gradient_trajopt_file=gradient_trajopt_file,
        )

        motion_gen = MotionGen(motion_gen_config)
        print("Motion generator initialized")
        return motion_gen

    def plan_traj_move(self, start_iks: torch.Tensor, goal_iks: torch.Tensor = None,
                 robot_cfg: Dict[str, Any] = None,
                 plan_cfg: Dict[str, Any] = None,
                 motion_gen: MotionGen = None) -> TrajResult:
        """Plan a trajectory for a move stage"""
        
        batch_size = plan_cfg.get("batch_size", 10)
        stage_type = plan_cfg.get("stage_type", StageType.MOVE)
        
        start_batches = torch.split(start_iks, batch_size)
        goal_batches = torch.split(goal_iks, batch_size)
        
        all_results = []
        success_count = 0
        
        with tqdm(total=len(start_batches), desc=f"{stage_type.name.capitalize()} Stage Planning") as pbar:
            for i, (b_start, b_goal) in enumerate(zip(start_batches, goal_batches)):
                # Convert to CuRobo JointState
                js_start = JointState.from_position(self.tensor_args.to_device(b_start))
                js_goal = JointState.from_position(self.tensor_args.to_device(b_goal))
                
                goal_pose = motion_gen.ik_solver.fk(js_goal.position).ee_pose
                goal = Goal(
                    goal_pose = goal_pose,
                    goal_state=js_goal,
                    current_state=js_start,
                )
                torch.cuda.synchronize()
                # Solve trajectory optimization
                result = motion_gen.trajopt_solver.solve_batch(goal)
                
                # Convert to TrajResult
                b_results = TrajResult(start_states=js_start.position.detach().clone().cpu(),
                                       goal_states=js_goal.position.detach().clone().cpu(),
                                       trajectories=result.solution.position.detach().clone().cpu(),
                                       success=result.success.detach().clone().cpu())
                all_results.append(b_results)
                
                torch.cuda.synchronize()
                success_count += b_results.success.sum().item()
                pbar.set_description(f"{stage_type.name.capitalize()} Batch {i+1}/{len(start_batches)} (Successes: {success_count})")
                pbar.update(1)                
                
        return TrajResult.cat(all_results)
    def _filter_traj_with_success(self, traj_result: TrajResult, stage_name: str) -> TrajResult:
        """Helper to filter trajectories by success mask and log progress."""
        success_mask = traj_result.success.to(torch.bool)
        filtered_result = traj_result[success_mask]
        print(f"Filtered {filtered_result.num_samples}/{traj_result.num_samples} successful trajectories for {stage_name} stage.")
        return filtered_result

    def _fk_filter(self, js: JointState, pose: Pose, 
                      motion_gen: MotionGen,
                      position_tolerance: float = 0.01, 
                      rotation_tolerance: float = 0.01) -> bool:
        """
        Check if the forward kinematics of the joint state is within the specified pose.
        """
        # Perform FK calculation
        fk_result = motion_gen.ik_solver.fk(js.position)
        
        # Get positions and quaternions
        pose_pos = pose.position.detach().cpu().numpy().flatten()
        pose_quat = pose.quaternion.detach().cpu().numpy().flatten()
        fk_pos = fk_result.ee_pose.position.detach().cpu().numpy().flatten()
        fk_quat = fk_result.ee_pose.quaternion.detach().cpu().numpy().flatten()
        
        # Calculate differences
        position_diff = np.linalg.norm(pose_pos - fk_pos)
        rotation_diff = quaternion_distance(pose_quat, fk_quat)
        
        # Store differences for statistics if lists exist
        if hasattr(self, 'position_diff_list'):
            self.position_diff_list.append(position_diff)
        if hasattr(self, 'rotation_diff_list'):
            self.rotation_diff_list.append(rotation_diff)
            
        return position_diff < position_tolerance and rotation_diff < rotation_tolerance

    def filter_traj_move(self, traj_result: TrajResult, robot_cfg: Dict[str, Any] = None,
                 motion_gen: MotionGen = None,
                 filter_cfg: Dict[str, Any] = None) -> TrajResult:
        """Filter trajectories for a move stage based on success."""
        stage_type = filter_cfg.get("stage_type", StageType.MOVE)
        return self._filter_traj_with_success(traj_result, stage_type.name)
    
    def filter_traj_move_articulated(self, traj_result: TrajResult, robot_cfg: Dict[str, Any] = None,
                 motion_gen: MotionGen = None,
                 filter_cfg: Dict[str, Any] = None) -> TrajResult:
        """Filter trajectories for a move_articulated stage based on success and FK validation."""
        stage_type = filter_cfg.get("stage_type", StageType.MOVE_ARTICULATED)
        
        # Step 1: filter by success
        traj_result = self._filter_traj_with_success(traj_result, stage_type.name)
        if traj_result.num_samples == 0:
            return traj_result
            
        # Step 2: filter by FK validation
        print(f"Step 2: Starting FK filtering for {stage_type.name}...")
        
        # Initialize motion generator
        if motion_gen is None:
            motion_gen = self.init_motion_gen(robot_cfg, fixed_base=True)
            
        pos_tol = filter_cfg.get("position_tolerance", 0.01)
        rot_tol = filter_cfg.get("rotation_tolerance", 0.05)
        
        self.position_diff_list = []
        self.rotation_diff_list = []
        
        fk_succ_indices = []
        
        trajectories = traj_result.trajectories
        goal_positions = traj_result.goal_states
        
        for i in tqdm(range(traj_result.num_samples), desc=f"{stage_type.name} FK Filtering"):
            # Get goal EE pose as reference
            goal_js = JointState.from_position(self.tensor_args.to_device(goal_positions[i:i+1]))
            goal_ee_pose = motion_gen.ik_solver.fk(goal_js.position).ee_pose
            
            traj_valid = True
            for j in range(trajectories.shape[1]):
                waypoint_js = JointState.from_position(self.tensor_args.to_device(trajectories[i:i+1, j]))
                if not self._fk_filter(waypoint_js, goal_ee_pose, motion_gen, pos_tol, rot_tol):
                    traj_valid = False
                    break
            
            if traj_valid:
                fk_succ_indices.append(i)
                
        # Log stats
        if self.position_diff_list:
            print(f"Position Diff - Mean: {np.mean(self.position_diff_list):.4f}, Max: {np.max(self.position_diff_list):.4f}")
        if self.rotation_diff_list:
            print(f"Rotation Diff - Mean: {np.mean(self.rotation_diff_list):.4f}, Max: {np.max(self.rotation_diff_list):.4f}")
            
        # Filter traj_result
        filtered_result = traj_result[fk_succ_indices]
        print(f"Step 2: Filtered from {traj_result.num_samples} to {filtered_result.num_samples} based on FK")
        
        return filtered_result
    
    def plan_traj_move_articulated(self, *args, **kwargs): return self.plan_traj_move(*args, **kwargs)
    def plan_traj_reach(self, *args, **kwargs): return self.plan_traj_move(*args, **kwargs)
    def plan_traj_grasp(self, *args, **kwargs): return self.plan_traj_move(*args, **kwargs)
    def plan_traj_lift(self, *args, **kwargs): return self.plan_traj_move(*args, **kwargs)
    def plan_traj_release(self, *args, **kwargs): return self.plan_traj_move(*args, **kwargs)
    def plan_traj_move_holding(self, *args, **kwargs): return self.plan_traj_move(*args, **kwargs)
    def plan_traj_wait(self, *args, **kwargs): return self.plan_traj_move(*args, **kwargs)
    def filter_traj_move_holding(self, *args, **kwargs): return self.filter_traj_move(*args, **kwargs)
    def filter_traj_reach(self, *args, **kwargs): return self.filter_traj_move(*args, **kwargs)
    def filter_traj_grasp(self, *args, **kwargs): return self.filter_traj_move(*args, **kwargs)
    def filter_traj_lift(self, *args, **kwargs): return self.filter_traj_move(*args, **kwargs)
    def filter_traj_release(self, *args, **kwargs): return self.filter_traj_move(*args, **kwargs)
    def filter_traj_wait(self, *args, **kwargs): return self.filter_traj_move(*args, **kwargs)
              
              
def test_planner():
    """
    Main pipeline function with trajectory filtering
    Complete AKR motion planning pipeline from scene to trajectory with filtering
    """
    print("=== AKR Motion Planning Pipeline with Filtering ===")
    
    # ===== CONFIGURATION =====
    # Scene configuration
    scene_cfg = {
        "path": "assets/scene/infinigen/kitchen_1130/scene_0_seed_0/export/export_scene.blend/export_scene.usdc",
        "pose": [0, 0, -0.13, 1, 0, 0, 0]
    }
    
    # Robot configuration  
    robot_cfg_path = "assets/robot/summit_franka/summit_franka.yml"
    robot_cfg = load_robot_cfg(robot_cfg_path)
    robot_cfg = process_robot_cfg(robot_cfg)
        
    # Object configuration (URDF path and basic info, dimensions loaded from metadata)
    object_cfg = {
        "path": "assets/object/Microwave/7221/7221_0_scaling.urdf",
        "asset_type": "Microwave",
        "asset_id": "7221"
    }
    metadata_path = "assets/scene/infinigen/kitchen_1130/scene_0_seed_0/info/metadata.json"
    object_cfg = load_object_from_metadata(metadata_path, object_cfg=object_cfg)
    
    # ===== INITIALIZATION =====
    print("\n1. Initializing Curobo Planner...")
    planner_cfg = {
        "voxel_dims": [5.0, 5.0, 5.0],
        "voxel_size": 0.02,
        "expanded_dims": [1.0, 0.2, 0.2],
        "collision_checker_type": CollisionCheckerType.VOXEL
    }
    planner = CuroboPlanner(planner_cfg)
    planner.setup_env(scene_cfg, object_cfg)
    
    
    # ===== LOAD GRASP POSES =====
    print("\n2. Loading grasp poses...")
    scaling_factor = 0.3562990018302636
    grasp_poses = get_grasp_poses(
        grasp_dir="assets/object/Microwave/7221/grasp",
        num_grasps=20,
        scaling_factor=scaling_factor
    )
    
    if not grasp_poses:
        print("ERROR: No grasp poses found!")
        return
        
    print(f"Loaded {len(grasp_poses)} grasp poses")
    
    motion_gen = planner.init_motion_gen(robot_cfg)
    
    clustering_params = {
        "ap_fallback_clusters": 30,
        "ap_clusters_upperbound": 80,
        "ap_clusters_lowerbound": 10
    }
    
    # ===== PROCESS EACH GRASP POSE =====
    for grasp_id, grasp_pose in enumerate(grasp_poses):
        print(f"\n{'='*80}")
        print(f"=== Processing Grasp Pose {grasp_id+1}/{len(grasp_poses)} ===")
        print(f"{'='*80}")
        
        # ===== IK PLANNING =====
        print("\n3. Planning IK solutions...")
        start_angles = [0.0]
        # goal_angles = [1.333088176515062, 1.2300752695249741, 1.0739553834158821, 0.9968029606353053]
        goal_angles = [0.9968029606353053]   
        
        # IK collection limits
        IK_LIMITS = {
            StageType.MOVE: [50, 50],
            StageType.MOVE_ARTICULATED: [50, 100],
        }
        
        start_ik_result = []
        goal_ik_result = []
        
        def plan_single_ik(angle):
            default_joint_cfg = {"joint_0": 0.0}
            joint_cfg={"joint_0": angle}
            target_Pose = get_open_ee_pose(
                object_pose=Pose.from_list(planner.object_pose),
                grasp_pose=Pose.from_list(grasp_pose),
                object_urdf=planner.object_urdf,
                handle="link_0",
                joint_cfg=joint_cfg,
                default_joint_cfg=default_joint_cfg
            )
            target_pose = torch.tensor(target_Pose.to_list())
            
            ik_result = planner.plan_ik(
                target_pose=target_pose,
                robot_cfg=robot_cfg,
                plan_cfg={
                    "joint_cfg": joint_cfg,
                    "enable_collision": True,
                },
                motion_gen=motion_gen,
            )
            # Stack angle
            ik_result.iks = stack_iks_angle(ik_result.iks, -angle) # TODO: negative
            
            print(f"  Angle {angle:.4f} rad: Found {ik_result.iks.shape[0]} IK solutions")
            
            # IK clustering
            ik_result = planner.ik_clustering(ik_result, **clustering_params)
            print(f"    After clustering: {ik_result.iks.shape[0]} IK solutions")
            return ik_result
        
        for angle in start_angles:
            ik_result = plan_single_ik(angle)
            start_ik_result.append(ik_result)
            
        start_ik_result = IKResult.cat(start_ik_result)
    
        for angle in goal_angles:
            ik_result = plan_single_ik(angle)
            goal_ik_result.append(ik_result)
        
        goal_ik_result = IKResult.cat(goal_ik_result)
        
        print(f"IK Planning completed:")
        print(f"  Start IKs: {start_ik_result.iks.shape}")  
        print(f"  Goal IKs: {goal_ik_result.iks.shape}")
        
        # ===== SAVE IK RESULTS =====
        print("\n4. Saving IK results...")
        base_dir = f"data/collect_1222/traj/summit_franka/scene_0_seed_0/7221/grasp_{grasp_id:04d}"
        os.makedirs(base_dir, exist_ok=True)
        save_ik(start_ik_result, f"{base_dir}/start_iks.pt")
        save_ik(goal_ik_result, f"{base_dir}/goal_iks.pt")
        
        if start_ik_result.iks.shape[0] == 0 or goal_ik_result.iks.shape[0] == 0:
            print("No IK solutions found for start or goal, skipping trajectory planning.")
            continue
        
        # ===== TRAJECTORY PLANNING =====
        print("\n5. Planning trajectories...")
        plan_cfg = {
            "stage_type": StageType.MOVE_ARTICULATED,
            "batch_size": 10,
            "expand_to_pairs": True,
        }
        
        akr_robot_cfg_path = f"assets/object/Microwave/7221/summit_franka_7221_0_grasp_{grasp_id:04d}.yml"
        
        akr_robot_cfg = load_robot_cfg(akr_robot_cfg_path)
        akr_robot_cfg = process_robot_cfg(akr_robot_cfg)
        
        motion_gen_akr = planner.init_motion_gen(akr_robot_cfg, fixed_base=True)
        
        traj_result = planner.plan_traj(
            start_iks=start_ik_result.iks,
            goal_iks=goal_ik_result.iks,
            robot_cfg=akr_robot_cfg,
            plan_cfg=plan_cfg,
            motion_gen=motion_gen_akr,
        )
        
        print(f"Trajectory Planning completed:")
        print(f"  Trajectories: {traj_result.trajectories.shape}")
        print(f"  Success rate: {traj_result.success.sum().item()}/{len(traj_result.success)}")
        
        # ===== SAVE TRAJECTORY RESULTS =====
        print("\n6. Saving trajectory results...")
        save_traj(traj_result, f"{base_dir}/traj_data.pt")
        
        # ===== TRAJECTORY FILTERING =====
        print("\n7. Filtering trajectories...")
        filter_cfg = {
            "stage_type": StageType.MOVE_ARTICULATED,
            "position_tolerance": 0.01,
            "rotation_tolerance": 0.05
        }
        filtered_result = planner.filter_traj(traj_result, robot_cfg=akr_robot_cfg, filter_cfg=filter_cfg, motion_gen=motion_gen_akr)
        
        print(f"Trajectory Filtering completed: {filtered_result.num_samples} trajectories")
        save_traj(filtered_result, f"{base_dir}/filtered_traj_data.pt")
        
        print(f"\n=== Processing for Grasp {grasp_id} Completed ===")

    print(f"\n=== ALL GRASP POSES PROCESSED ===")


if __name__ == "__main__":
    test_planner()