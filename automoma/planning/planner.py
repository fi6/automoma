# Copyright (c) 2024-2025, AutoMoMa Authors. All rights reserved.
# SPDX-License-Identifier: MIT
"""CuroboPlanner — cuRobo-based motion planner for mobile manipulation.

Uses V1's correct planning logic with V2's clean class structure.
All hyper-parameters are supplied via ``planner_cfg`` (from YAML config).
"""

from __future__ import annotations

import gc
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from yourdfpy import URDF

from curobo.geom.sdf.world import CollisionCheckerType, WorldCollisionConfig
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.sdf.world_voxel import WorldVoxelCollision
from curobo.geom.types import Cuboid, Mesh, WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.usd_helper import UsdHelper, set_prim_transform
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

from automoma.core.types import IKResult, StageType, TrajResult
from automoma.utils.math_utils import (
    _convert_to_list,
    expand_to_pairs,
    ik_clustering,
    mark_cuboid_as_empty,
    pose_multiply,
    quaternion_distance,
    stack_iks_angle,
)
from automoma.utils.file_utils import load_robot_cfg, process_robot_cfg


class CuroboPlanner:
    """CuRobo-based motion planner for articulated-object manipulation.

    Lifecycle::

        planner = CuroboPlanner(planner_cfg)
        planner.setup_env(scene_cfg, object_cfg)
        ik = planner.plan_ik(target_pose, robot_cfg)
        mask = planner.cluster_ik(ik)
        traj = planner.plan_traj(start_iks, goal_iks, robot_cfg)
        traj = planner.filter_traj(traj, robot_cfg)
    """

    def __init__(self, planner_cfg: Dict[str, Any]):
        self.cfg = planner_cfg
        self.tensor_args = TensorDeviceType()
        self.usd_helper = UsdHelper()

        # Populated by setup_env
        self.root_pose: List[float] = [0, 0, 0, 1, 0, 0, 0]
        self.object_pose: Optional[List[float]] = None
        self.object_urdf: Optional[URDF] = None
        self.object_mesh: Optional[Mesh] = None
        self.collision = None
        self.esdf = None
        self.world_collision_config = None
        self.collision_type = None
        self.expanded_object_cuboid = None
        self.world_voxel_collision_ik = None
        self.world_voxel_collision_traj = None
        self.motion_gen = None
        self.motion_gen_akr = None

    # ====================================================================
    # Environment setup
    # ====================================================================

    def setup_env(
        self,
        scene_cfg: Dict[str, Any],
        object_cfg: Dict[str, Any],
    ) -> None:
        """Load scene, object, and build the collision world."""
        self.scene_cfg = scene_cfg
        self.object_cfg = object_cfg
        self.motion_gen = None
        self.motion_gen_akr = None

        self._init_root_pose(object_cfg["pose"], scene_cfg["pose"])
        self._load_object()
        self._load_scene()
        self._setup_collision_world()

    def _init_root_pose(
        self,
        object_pose: List[float],
        scene_pose: List[float],
    ) -> None:
        """Set root pose so the object-centred local frame keeps object-local pose strict.

        The required invariant is: after transforming poses into the local planning
        frame, the object's own local pose must remain exactly ``(0, 0, z, 1, 0, 0, 0)``.
        To preserve that, we must follow the reference cuAKR order exactly: first
        invert the full object pose, then zero the resulting Z translation, and only
        then compose with the scene pose. Zeroing Z before inversion is not
        equivalent.
        """
        object_pose_inverse = Pose.from_list(object_pose).inverse().to_list()
        object_pose_inverse[2] = 0.0
        self.root_pose = pose_multiply(object_pose_inverse, scene_pose)

    def _get_world_pose(self, pose) -> List[float]:
        return pose_multiply(self.root_pose, pose)

    # -- Object --------------------------------------------------------------

    def _load_object(self) -> None:
        obj_path = self.object_cfg["path"]
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Object URDF not found: {obj_path}")
        print(f"Loading object: {obj_path}")

        # The pose from object_cfg is already correct (processed by
        # prepare_scene.py + load_object_from_metadata).  No extra rotation.
        self.object_pose = self._get_world_pose(self.object_cfg["pose"])
        self.object_urdf = URDF.load(obj_path, build_collision_scene_graph=True)
        trimesh = self.object_urdf.scene.to_mesh()
        self.object_mesh = Mesh(
            trimesh=trimesh, name="target_object", pose=self.object_pose
        )

    # -- Scene ---------------------------------------------------------------

    def _load_scene(self) -> None:
        usd_path = self.scene_cfg["path"]
        if not os.path.exists(usd_path):
            raise FileNotFoundError(f"USD file not found: {usd_path}")
        print(f"Loading scene: {usd_path}")

        # USD keeps root layers cached in-process.  This planner mutates
        # /World/scene below to enter the object-centered planning frame; reload
        # the layer so a second CuroboPlanner does not start from the first
        # planner's unsaved transform edits.
        from pxr import Sdf
        layer = Sdf.Layer.FindOrOpen(os.path.abspath(usd_path))
        if layer is not None:
            layer.Reload()

        self.usd_helper.load_stage_from_file(usd_path)
        
        # PRESERVE existing transform in the USD (e.g. Z = -0.12 from prepare_scene.py)
        # Otherwise, the object would be 0.12m too low relative to the table.
        from automoma.utils.math_utils import matrix_to_pose
        orig_mat = self.usd_helper.get_pose("/World/scene")
        orig_pose = matrix_to_pose(orig_mat).tolist()
        final_scene_pose = pose_multiply(self.root_pose, orig_pose)

        set_prim_transform(
            self.usd_helper.stage.GetPrimAtPath("/World/scene"),
            final_scene_pose,
        )
        print("Getting collision world from scene...")
        self.collision = (
            self.usd_helper.get_obstacles_from_stage().get_collision_check_world()
        )
        print(f"Collision meshes: {len(self.collision.mesh)}")
        self._clean_collision_meshes()

    def _clean_collision_meshes(self) -> None:
        to_remove = [
            m for m in self.collision.mesh
            if len(m.vertices) == 0 or len(m.faces) == 0
        ]
        for m in to_remove:
            self.collision.mesh.remove(m)
        print(f"Removed {len(to_remove)} empty meshes. Remaining: {len(self.collision.mesh)}")

    # -- Collision world -----------------------------------------------------

    def _setup_collision_world(self) -> None:
        ctype = self.cfg.get("collision_checker_type", "VOXEL")
        self.collision_type = getattr(CollisionCheckerType, ctype) if isinstance(ctype, str) else ctype

        obj_Pose = Pose.from_list(self.object_pose)
        obj_inv = obj_Pose.inverse().to_list()

        voxel_dims = self.cfg.get("voxel_dims", [5.0, 5.0, 5.0])
        voxel_size = self.cfg.get("voxel_size", 0.02)
        expanded_dims = self.cfg.get("expanded_dims", [1.0, 0.2, 0.2])

        world_model = {
            "voxel": {
                "base": {
                    "dims": voxel_dims,
                    "pose": obj_inv,
                    "voxel_size": voxel_size,
                }
            }
        }

        world_collision_config = WorldCollisionConfig.load_from_dict(
            {"checker_type": self.collision_type, "max_distance": 5.0, "n_envs": 1},
            world_model,
            self.tensor_args,
        )

        self.world_collision_config = world_collision_config
        enable_collision = self.cfg.get("enable_collision", True)

        if enable_collision:
            # Build ESDF once, then clone it into two independent voxel checkers
            # for IK and trajectory planning. This mirrors tests/planner_cvpr26.py.
            world_voxel = WorldVoxelCollision(world_collision_config)
            support = WorldConfig.create_collision_support_world(self.collision)
            mesh_cfg = WorldCollisionConfig(self.tensor_args, world_model=support)
            world_mesh = WorldMeshCollision(mesh_cfg)

            voxel_grid = world_voxel.get_voxel_grid("base")
            self.esdf = world_mesh.get_esdf_in_bounding_box(
                Cuboid(name="base", pose=voxel_grid.pose, dims=voxel_grid.dims),
                voxel_size=voxel_grid.voxel_size,
            )

            # Expanded cuboid — mark object region as free space
            exp = (np.array(self.object_cfg["dimensions"]) + expanded_dims).tolist()
            self.expanded_object_cuboid = Cuboid(
                name=f"{self.object_cfg['asset_type']}_{self.object_cfg['asset_id']}_expanded",
                pose=self.object_pose,
                dims=exp,
                tensor_args=self.tensor_args,
            )

            # Optional collision visualization (before clearing object region)
            if self.cfg.get("visualize_collision", False):
                from automoma.utils.visual_utils import visualize_voxel_grid_with_cuboid
                print("Visualizing collision BEFORE marking cuboid as empty...")
                visualize_voxel_grid_with_cuboid(
                    self.expanded_object_cuboid, self.esdf,
                    mesh_obstacle=self.object_mesh,
                )

            self.esdf = mark_cuboid_as_empty(self.esdf, self.expanded_object_cuboid)

            # Optional collision visualization (after clearing object region)
            if self.cfg.get("visualize_collision", False):
                from automoma.utils.visual_utils import visualize_voxel_grid_with_cuboid
                print("Visualizing collision AFTER marking cuboid as empty...")
                visualize_voxel_grid_with_cuboid(
                    self.expanded_object_cuboid, self.esdf,
                    mesh_obstacle=self.object_mesh,
                )

            if self.collision_type == CollisionCheckerType.MESH:
                mesh = self.usd_helper.voxel_to_mesh(self.esdf, pitch=voxel_size)
                self.usd_helper.add_mesh_to_stage(mesh, "/World/esdf")
                self.esdf = voxel_grid
                world_collision_config.world_model["mesh"] = [mesh]

        self._init_collision_checkers(
            world_collision_config,
            disable_collision=not enable_collision,
        )

    # ====================================================================
    # Motion-gen initialisation
    # ====================================================================

    def _init_collision_checkers(
        self,
        world_collision_config: WorldCollisionConfig,
        *,
        disable_collision: bool = False,
    ) -> None:
        """Initialise independent voxel checkers for IK and trajectory planning."""
        self.world_voxel_collision_ik = WorldVoxelCollision(world_collision_config)
        self.world_voxel_collision_traj = WorldVoxelCollision(world_collision_config)

        if disable_collision or self.esdf is None:
            return

        for checker in (self.world_voxel_collision_ik, self.world_voxel_collision_traj):
            checker.clear_voxelization_cache()
            if getattr(checker, "cache", None) is not None:
                checker.clear_cache()
            checker.update_voxel_data(self.esdf)
        torch.cuda.synchronize()

    def _get_collision_checker(
        self,
        *,
        fixed_base: bool,
        enable_collision: bool,
        collision_role: Optional[str] = None,
    ) -> WorldVoxelCollision:
        if self.world_collision_config is None:
            raise RuntimeError("setup_env() must be called before init_motion_gen().")

        if not enable_collision:
            return WorldVoxelCollision(self.world_collision_config)

        if collision_role is None:
            collision_role = "traj" if fixed_base else "ik"
        checker = self.world_voxel_collision_traj if collision_role == "traj" else self.world_voxel_collision_ik
        if checker is None:
            checker = WorldVoxelCollision(self.world_collision_config)
            if self.esdf is not None:
                checker.clear_voxelization_cache()
                if getattr(checker, "cache", None) is not None:
                    checker.clear_cache()
                checker.update_voxel_data(self.esdf)
                torch.cuda.synchronize()
        return checker

    def free_cuda_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def init_motion_gen(
        self,
        robot_cfg: Dict,
        *,
        fixed_base: bool = False,
        enable_collision: bool = True,
        collision_role: Optional[str] = None,
    ) -> MotionGen:
        """Create a cuRobo MotionGen instance with the current collision world."""
        robot_cfg = load_robot_cfg(robot_cfg)

        traj_cfg = self.cfg.get("traj", {})
        grad_file = traj_cfg.get(
            "gradient_trajopt_file",
            "gradient_trajopt_fixbase.yml" if fixed_base else "gradient_trajopt.yml",
        )
        coll = self._get_collision_checker(
            fixed_base=fixed_base,
            enable_collision=enable_collision,
            collision_role=collision_role,
        )

        mg_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            WorldConfig(),
            self.tensor_args,
            world_coll_checker=coll,
            num_trajopt_seeds=traj_cfg.get("num_trajopt_seeds", 12),
            num_graph_seeds=traj_cfg.get("num_graph_seeds", 12),
            interpolation_dt=traj_cfg.get("interpolation_dt", 0.05),
            collision_cache={"obb": 30, "mesh": 100},
            optimize_dt=True,
            trajopt_dt=None,
            trajopt_tsteps=traj_cfg.get("trajopt_tsteps", 32),
            trim_steps=None,
            use_cuda_graph=False,
            gradient_trajopt_file=grad_file,
        )
        mg = MotionGen(mg_config)
        print("Motion generator initialised")
        return mg

    def _update_world_collision(
        self,
        motion_gen: MotionGen,
        joint_cfg: Optional[Dict[str, float]],
        enable_collision: bool = True,
        include_object_mesh: bool = True,
    ) -> None:
        """Re-mesh the object with a new joint config and push to MotionGen."""
        if not enable_collision:
            return
        if joint_cfg is not None:
            self.object_urdf.update_cfg(joint_cfg)

        meshes = []
        if include_object_mesh:
            trimesh = self.object_urdf.scene.to_mesh()
            obj_mesh = Mesh(trimesh=trimesh, name="target_object", pose=self.object_pose)

            if self.usd_helper.stage.GetPrimAtPath("/World/object"):
                self.usd_helper.stage.RemovePrim("/World/object")
            self.usd_helper.add_mesh_to_stage(obj_mesh, "/World/object")

            obj_meshes = (
                self.usd_helper.get_obstacles_from_stage(only_paths=["/World/object"])
                .get_collision_check_world()
                .mesh
            )
            meshes.extend(list(obj_meshes))
        elif self.usd_helper.stage.GetPrimAtPath("/World/object"):
            self.usd_helper.stage.RemovePrim("/World/object")

        if self.collision_type == CollisionCheckerType.MESH:
            env_meshes = (
                self.usd_helper.get_obstacles_from_stage(only_paths=["/World/esdf"])
                .get_collision_check_world()
                .mesh
            )
            meshes += list(env_meshes)
        motion_gen.update_world(WorldConfig(mesh=meshes, voxel=[self.esdf]))

    # ====================================================================
    # IK planning
    # ====================================================================

    def plan_ik(
        self,
        target_pose: torch.Tensor,
        robot_cfg: Union[str, Dict],
        *,
        plan_cfg: Optional[Dict[str, Any]] = None,
        motion_gen: Optional[MotionGen] = None,
    ) -> IKResult:
        """Solve IK for a single 7-D ``target_pose``.

        Returns an :class:`IKResult` with *all* unique solutions (before
        clustering).
        """
        if plan_cfg is None:
            plan_cfg = {}
        robot_cfg = load_robot_cfg(robot_cfg)

        if motion_gen is None:
            if self.motion_gen is None:
                self.motion_gen = self.init_motion_gen(
                    robot_cfg,
                    fixed_base=False,
                    collision_role="ik",
                    enable_collision=plan_cfg.get(
                        "enable_collision",
                        self.cfg.get("enable_collision", True),
                    ),
                )
            motion_gen = self.motion_gen

        joint_cfg = plan_cfg.get("joint_cfg")
        enable_coll = plan_cfg.get("enable_collision", self.cfg.get("enable_collision", True))
        self._update_world_collision(motion_gen, joint_cfg, enable_coll, include_object_mesh=True)

        retract = self.tensor_args.to_device(
            robot_cfg["kinematics"]["cspace"]["retract_config"]
        )
        goal = Pose.from_list(_convert_to_list(target_pose))
        ik_seeds = self.cfg.get("ik_seeds", 20000)
        iks = self._solve_ik(motion_gen, goal, retract, num_seeds=ik_seeds)

        if iks is None or iks.shape[0] == 0:
            print("No IK solutions found.")
            return IKResult(
                target_poses=target_pose.unsqueeze(0),
                iks=torch.zeros(0, motion_gen.dof),
            )

        poses = target_pose.unsqueeze(0).expand(iks.shape[0], -1)
        return IKResult(target_poses=poses, iks=iks)

    def _solve_ik(
        self,
        motion_gen: MotionGen,
        goal_pose: Pose,
        retract_config: torch.Tensor,
        num_seeds: int = 20000,
    ) -> torch.Tensor:
        return motion_gen.ik_solver.solve_single(
            goal_pose=goal_pose,
            retract_config=self.tensor_args.to_device(retract_config).unsqueeze(0),
            return_seeds=num_seeds,
            num_seeds=num_seeds,
            link_poses=None,
        ).get_unique_solution()

    # ====================================================================
    # IK clustering  (returns mask)
    # ====================================================================

    def cluster_ik(
        self,
        ik_result: IKResult,
        cluster_cfg: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Cluster IK solutions and return a boolean mask.

        All hyper-parameters come from ``cluster_cfg`` (or planner defaults).
        """
        if ik_result.iks.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        cfg = cluster_cfg or self.cfg.get("clustering", {})
        return ik_clustering(
            ik_result.iks,
            kmeans_clusters=cfg.get("kmeans_clusters", 500),
            ap_fallback_clusters=cfg.get("ap_fallback_clusters", 30),
            ap_clusters_upperbound=cfg.get("ap_clusters_upperbound", 80),
            ap_clusters_lowerbound=cfg.get("ap_clusters_lowerbound", 10),
        )

    # ====================================================================
    # Trajectory planning
    # ====================================================================

    def plan_traj(
        self,
        start_iks: torch.Tensor,
        goal_iks: torch.Tensor,
        robot_cfg: Union[str, Dict],
        *,
        plan_cfg: Optional[Dict[str, Any]] = None,
        motion_gen: Optional[MotionGen] = None,
    ) -> TrajResult:
        """Plan trajectories (batched) from *start_iks* to *goal_iks*.

        ``plan_cfg`` keys:
            - ``batch_size`` (int): GPU batch size.
            - ``expand_to_pairs`` (bool): create Cartesian product of start/goal.
            - ``joint_cfg`` / ``enable_collision``: passed to optional world update.
        """
        if plan_cfg is None:
            plan_cfg = {}

        if start_iks.shape[0] == 0:
            print("No IK solutions to plan trajectories for.")
            return TrajResult.fallback()

        if plan_cfg.get("expand_to_pairs", False):
            start_iks, goal_iks = expand_to_pairs(start_iks, goal_iks)
        assert start_iks.shape[0] == goal_iks.shape[0]

        robot_cfg = load_robot_cfg(robot_cfg)
        joint_cfg = plan_cfg.get("joint_cfg")
        enable_coll = plan_cfg.get("enable_collision", self.cfg.get("enable_collision", True))
        include_object_mesh = plan_cfg.get("include_object_mesh", False)

        if start_iks.shape[1] > 0 and start_iks[:, -1].min() > 0:
            start_iks = start_iks.clone()
            start_iks[:, -1] *= -1
        if goal_iks.shape[1] > 0 and goal_iks[:, -1].min() > 0:
            goal_iks = goal_iks.clone()
            goal_iks[:, -1] *= -1
        print(
            "Traj prep:",
            f"start_angle_range=[{start_iks[:, -1].min().item():.4f}, {start_iks[:, -1].max().item():.4f}]",
            f"goal_angle_range=[{goal_iks[:, -1].min().item():.4f}, {goal_iks[:, -1].max().item():.4f}]",
            f"include_object_mesh={include_object_mesh}",
        )

        if motion_gen is None:
            # AKR robot configs are grasp-specific, so initialise the trajectory
            # MotionGen for every plan_traj call and reuse it for the matching
            # filter_traj call. This mirrors tests/planner_cvpr26.py.
            fixed_base = plan_cfg.get(
                "fixed_base",
                self.cfg.get("traj", {}).get("fixed_base", False),
            )
            self.motion_gen_akr = self.init_motion_gen(
                robot_cfg,
                fixed_base=fixed_base,
                enable_collision=enable_coll,
                collision_role="traj",
            )
            motion_gen = self.motion_gen_akr

        if plan_cfg.get("update_world_collision", False) or include_object_mesh:
            self._update_world_collision(
                motion_gen,
                joint_cfg,
                enable_coll,
                include_object_mesh=include_object_mesh,
            )

        batch_size = plan_cfg.get("batch_size", self.cfg.get("traj", {}).get("batch_size", 20))
        start_batches = torch.split(start_iks, batch_size)
        goal_batches = torch.split(goal_iks, batch_size)

        all_results: List[TrajResult] = []
        success_count = 0

        with tqdm(total=len(start_batches), desc="TrajOpt") as pbar:
            for i, (b_s, b_g) in enumerate(zip(start_batches, goal_batches)):
                js_s = JointState.from_position(self.tensor_args.to_device(b_s))
                js_g = JointState.from_position(self.tensor_args.to_device(b_g))
                ee = motion_gen.ik_solver.fk(js_g.position).ee_pose
                goal = Goal(goal_pose=ee, goal_state=js_g, current_state=js_s)
                torch.cuda.synchronize()

                result = motion_gen.trajopt_solver.solve_batch(goal)
                trajectories = result.solution.position.detach().clone().cpu()
                success = result.success.detach().clone().cpu()

                # cuRobo may squeeze the batch axis for single-sample solves.
                if trajectories.ndim == 2:
                    trajectories = trajectories.unsqueeze(0)
                if success.ndim == 0:
                    success = success.unsqueeze(0)

                batch_res = TrajResult(
                    start_states=js_s.position.detach().clone().cpu(),
                    goal_states=js_g.position.detach().clone().cpu(),
                    trajectories=trajectories,
                    success=success,
                )
                all_results.append(batch_res)
                torch.cuda.synchronize()
                success_count += batch_res.success.sum().item()
                pbar.set_description(
                    f"TrajOpt {i + 1}/{len(start_batches)} (ok: {success_count})"
                )
                pbar.update(1)

        return TrajResult.cat(all_results)

    # ====================================================================
    # Trajectory filtering
    # ====================================================================

    def filter_traj(
        self,
        traj_result: TrajResult,
        robot_cfg: Union[str, Dict],
        *,
        filter_cfg: Optional[Dict[str, Any]] = None,
        motion_gen: Optional[MotionGen] = None,
    ) -> TrajResult:
        """Filter trajectories using cuAKR-style success + waypoint FK checks."""
        if traj_result.num_samples == 0:
            return TrajResult.fallback(
                robot_dof=traj_result.start_states.shape[-1] if traj_result.start_states.ndim == 2 else 0
            )

        cfg = filter_cfg or self.cfg.get("filter", {})
        pos_tol = cfg.get("position_tolerance", 0.01)
        rot_tol = cfg.get("rotation_tolerance", 0.05)

        robot_cfg = load_robot_cfg(robot_cfg)
        if motion_gen is None:
            if self.motion_gen_akr is None:
                self.motion_gen_akr = self.init_motion_gen(
                    robot_cfg,
                    fixed_base=self.cfg.get("traj", {}).get("fixed_base", False),
                    enable_collision=self.cfg.get("enable_collision", True),
                    collision_role="traj",
                )
            motion_gen = self.motion_gen_akr

        start_state = traj_result.start_states
        goal_state = traj_result.goal_states
        trajectories = traj_result.trajectories
        success = traj_result.success

        indices_count = goal_state.shape[0]
        print(f"Loaded {indices_count} trajectories")

        # Step 1: keep only trajopt-successful trajectories.
        filtered_indices = success.nonzero(as_tuple=True)[0]
        if filtered_indices.shape[0] == 0:
            print("Step 1: No successful trajectories to filter, returning empty result")
            return TrajResult.fallback(robot_dof=start_state.shape[-1])

        goal_state = goal_state[filtered_indices]
        start_state = start_state[filtered_indices]
        trajectories = trajectories[filtered_indices]
        success = success[filtered_indices]

        step_1_count = goal_state.shape[0]
        print(f"Step 1: Filtered from {indices_count} to {step_1_count} trajectories based on success.")

        # Step 2: waypoint-level FK validation against the goal EE pose.
        pos_diffs, rot_diffs = [], []
        fk_succ_indices = []
        for i in tqdm(range(goal_state.shape[0]), desc="FK filter"):
            goal_js = JointState.from_position(
                self.tensor_args.to_device(goal_state[i : i + 1])
            )
            goal_ee = motion_gen.ik_solver.fk(goal_js.position).ee_pose
            trajectory_valid = True
            for j in range(trajectories.shape[1]):
                wp_js = JointState.from_position(
                    self.tensor_args.to_device(trajectories[i : i + 1, j])
                )
                fk = motion_gen.ik_solver.fk(wp_js.position).ee_pose
                pd = np.linalg.norm(
                    goal_ee.position.cpu().numpy().flatten()
                    - fk.position.cpu().numpy().flatten()
                )
                rd = quaternion_distance(
                    goal_ee.quaternion.cpu().numpy().flatten(),
                    fk.quaternion.cpu().numpy().flatten(),
                )
                pos_diffs.append(pd)
                rot_diffs.append(rd)
                if pd >= pos_tol or rd >= rot_tol:
                    trajectory_valid = False
                    break

            if trajectory_valid:
                fk_succ_indices.append(i)

        if pos_diffs:
            print(f"Position diff — mean: {np.mean(pos_diffs):.4f}, max: {np.max(pos_diffs):.4f}")
        if rot_diffs:
            print(f"Rotation diff — mean: {np.mean(rot_diffs):.4f}, max: {np.max(rot_diffs):.4f}")
        print(f"FK success indices: {len(fk_succ_indices)}")

        filtered_indices = torch.tensor(fk_succ_indices, device=goal_state.device, dtype=torch.long)
        if filtered_indices.shape[0] == 0:
            print("Step 2: No successful trajectories to filter, returning empty result")
            return TrajResult.fallback(robot_dof=start_state.shape[-1])

        goal_state = goal_state[filtered_indices]
        start_state = start_state[filtered_indices]
        trajectories = trajectories[filtered_indices]
        success = success[filtered_indices]

        step_2_count = goal_state.shape[0]
        print(f"Step 2: Filtered from {step_1_count} to {step_2_count} trajectories based on FK filtering.")

        return TrajResult(
            start_states=start_state,
            goal_states=goal_state,
            trajectories=trajectories,
            success=success,
        )
