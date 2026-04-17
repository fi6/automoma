# Copyright (c) 2024-2025, AutoMoMa Authors. All rights reserved.
# SPDX-License-Identifier: MIT
"""Planning pipeline — orchestrates IK → Traj → Filter → 12-DOF output.

This module replaces the combination of ``pipeline_plan.py`` +
``prepare_traj.py`` by producing IsaacLab-Arena–compatible 12-DOF trajectory
files directly.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from curobo.types.math import Pose

from automoma.core.types import IKResult, TrajResult, aggregate_grasp_goal_results
from automoma.planning.planner import CuroboPlanner
from automoma.utils.file_utils import (
    get_grasp_poses,
    load_object_from_metadata,
    load_robot_cfg,
    load_traj,
    process_robot_cfg,
    save_ik,
    save_traj,
)
from automoma.utils.math_utils import get_open_ee_pose, stack_iks_angle


class PlanningPipeline:
    """End-to-end planning from scene metadata → IsaacLab 12-DOF ``.pt`` file.

    Usage::

        pipe = PlanningPipeline(cfg)
        pipe.run(scene_name="scene_0_seed_0", object_id="7221")
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        planner_cfg = cfg.get("planner", {})
        self.planner = CuroboPlanner(planner_cfg)

        self.output_cfg = planner_cfg.get("output", {})

    # ====================================================================
    # Public API
    # ====================================================================

    def run(
        self,
        scene_name: str,
        object_id: str,
        *,
        output_dir: Optional[str] = None,
        mode: str = "train",
    ) -> str:
        """Execute the full planning pipeline and save results.

        Returns the path of the saved ``.pt`` file.
        """
        # ─── resolve configs ─────────────────────────────────────────────
        cfg = self.cfg
        scene_dir = cfg.get("scene_dir", "")
        scene_cfg = self._build_scene_cfg(scene_dir, scene_name)
        obj_cfg = self._build_object_cfg(object_id, scene_cfg)
        robot_cfg_path = cfg["robot"][cfg.get("robot_name", "summit_franka")]["config_path"]
        robot_cfg = process_robot_cfg(load_robot_cfg(robot_cfg_path))

        if output_dir is None:
            obj_name = f"{obj_cfg['asset_type'].lower()}_{object_id}"
            output_dir = os.path.join(
                cfg.get("output_dir", "data/trajs"),
                cfg.get("robot_name", "summit_franka"),
                obj_name,
                scene_name,
                mode,  # train/test subdirectory for full isolation
            )
        os.makedirs(output_dir, exist_ok=True)

        resume = cfg.get("resume", True)

        # ─── setup planner ───────────────────────────────────────────────
        print(f"\n{'=' * 60}")
        print(f"Planning: scene={scene_name}  object={object_id}")
        print(f"{'=' * 60}")

        self.planner.setup_env(scene_cfg, obj_cfg)
        traj_planner = CuroboPlanner(self.cfg.get("planner", {}))
        traj_planner.setup_env(scene_cfg, obj_cfg)

        # ─── per-grasp loop ──────────────────────────────────────────────
        obj_meta = cfg["objects"][object_id]
        grasp_ids: List[int] = obj_meta.get("grasp_ids", list(range(20)))
        goal_angles: List[float] = obj_meta.get("goal_angle", [1.57])

        handle_link = obj_meta.get("handle_link", "link_0")
        joint_name = obj_meta.get("joint_name", "joint_0")
        scale = obj_meta.get("scale", 1.0)
        grasp_dir = os.path.join(
            os.path.dirname(obj_meta["urdf_path"]), "grasp"
        )

        # AKR robot config template
        akr_template = obj_meta.get("akr_template", "")
        robot_name = cfg.get("robot_name", "summit_franka")

        all_raw: List[TrajResult] = []

        for g_idx in grasp_ids:
            grasp_output = os.path.join(output_dir, f"grasp_{g_idx:04d}")
            os.makedirs(grasp_output, exist_ok=True)

            # Resume support
            final_pt = os.path.join(grasp_output, "traj_data.pt")
            if resume and os.path.exists(final_pt):
                print(f"\n--- Grasp {g_idx}: resuming from {final_pt}")
                raw = load_traj(final_pt)
                all_raw.append(raw)
                continue

            print(f"\n--- Grasp {g_idx} ---")

            # Load grasp pose
            grasp_file = os.path.join(grasp_dir, f"{g_idx:04d}.npy")
            if not os.path.exists(grasp_file):
                print(f"  Grasp file not found: {grasp_file}, skipping")
                continue
            grasp_raw = np.load(grasp_file).copy()
            grasp_raw[:3] *= scale
            grasp_pose = Pose.from_list(grasp_raw.tolist())

            # AKR robot config for this grasp
            akr_path = akr_template.format(
                object_id=object_id,
                robot_name=robot_name,
                grasp_id=g_idx,
            )
            if not os.path.exists(akr_path):
                print(f"  AKR cfg not found: {akr_path}, skipping")
                continue
            akr_robot_cfg = process_robot_cfg(load_robot_cfg(akr_path))

            object_Pose = Pose.from_list(self.planner.object_pose)
            default_joint_cfg = {joint_name: 0.0}
            ik_path = os.path.join(grasp_output, "ik_data.pt")
            goal_ik_path = os.path.join(grasp_output, "ik_goal_data.pt")
            grasp_start_iks: List[IKResult] = []
            grasp_goal_iks: List[IKResult] = []
            grasp_trajs: List[TrajResult] = []

            for goal_angle in goal_angles:
                # --- IK planning ---
                # Start IK (closed state)
                start_target = get_open_ee_pose(
                    object_Pose,
                    grasp_pose,
                    self.planner.object_urdf,
                    handle_link,
                    default_joint_cfg,
                    default_joint_cfg,
                )
                start_ik = self.planner.plan_ik(
                    torch.tensor(start_target.to_list()),
                    robot_cfg,
                    plan_cfg={
                        "joint_cfg": default_joint_cfg,
                        "enable_collision": self.cfg.get("planner", {}).get("enable_collision", True),
                    },
                )
                print(f"  Start IK: {len(start_ik)} solutions")

                # Goal IK (open state)
                goal_target = get_open_ee_pose(
                    object_Pose,
                    grasp_pose,
                    self.planner.object_urdf,
                    handle_link,
                    {joint_name: goal_angle},
                    default_joint_cfg,
                )
                goal_ik = self.planner.plan_ik(
                    torch.tensor(goal_target.to_list()),
                    robot_cfg,
                    plan_cfg={
                        "joint_cfg": {joint_name: goal_angle},
                        "enable_collision": self.cfg.get("planner", {}).get("enable_collision", True),
                    },
                )
                print(f"  Goal IK: {len(goal_ik)} solutions")
                self.planner.free_cuda_cache()

                if len(start_ik) == 0 or (goal_ik is not None and len(goal_ik) == 0):
                    print(f"  No IK solutions, skipping grasp {g_idx}")
                    continue

                # --- Clustering ---
                s_mask = self.planner.cluster_ik(start_ik)
                g_mask = self.planner.cluster_ik(goal_ik) if goal_ik is not None else s_mask

                start_clustered = start_ik.iks[s_mask]
                goal_clustered = goal_ik.iks[g_mask] if goal_ik is not None else start_clustered
                print(f"  Clustered: start={start_clustered.shape[0]}, goal={goal_clustered.shape[0]}")

                # Add object joint angle column
                # The articulated joint should be recorded as a negative value per AKR mechanical convention
                start_with_angle = stack_iks_angle(start_clustered, 0.0)
                goal_with_angle = stack_iks_angle(goal_clustered, -goal_angle)

                # --- Trajectory planning ---
                traj_cfg_plan = {
                    "expand_to_pairs": True,
                    "batch_size": self.cfg.get("planner", {}).get("traj", {}).get("batch_size", 20),
                    "joint_cfg": {joint_name: goal_angle},
                    "enable_collision": self.cfg.get("planner", {}).get("enable_collision", True),
                }
                traj_result = traj_planner.plan_traj(
                    start_with_angle, goal_with_angle,
                    akr_robot_cfg,
                    plan_cfg=traj_cfg_plan,
                )
                print(f"  TrajOpt raw: {traj_result.success.sum().item()}/{traj_result.num_samples} ok")

                # --- Filtering ---
                traj_result = traj_planner.filter_traj(
                    traj_result, akr_robot_cfg,
                )
                print(f"  TrajOpt filtered: {traj_result.success.sum().item()}/{traj_result.num_samples} ok")

                grasp_start_iks.append(start_ik)
                grasp_goal_iks.append(goal_ik)
                grasp_trajs.append(traj_result)
                traj_planner.free_cuda_cache()

            if not grasp_trajs:
                print(f"  No valid trajectories for grasp {g_idx}, skipping save")
                continue

            merged_start_ik, merged_goal_ik, merged_traj = aggregate_grasp_goal_results(
                grasp_start_iks,
                grasp_goal_iks,
                grasp_trajs,
            )
            save_ik(merged_start_ik, ik_path)
            save_ik(merged_goal_ik, goal_ik_path)
            save_traj(merged_traj, final_pt)
            all_raw.append(merged_traj)

        # ─── merge + 12D conversion ──────────────────────────────────────
        if not all_raw:
            print("\nNo trajectories produced. Aborting.")
            return ""

        merged = TrajResult.cat(all_raw)
        print(f"\nMerged: {merged.num_samples} trajectories "
              f"({merged.success.sum().item()} successful)")

        converted = self._convert_to_12d(merged)
        out_path = os.path.join(output_dir, f"traj_data_{mode}.pt")
        torch.save(converted, out_path)
        print(f"Saved: {out_path}")
        self._verify(converted)
        return out_path

    # ====================================================================
    # 11-D → 12-D conversion  (from prepare_traj.py)
    # ====================================================================

    def _convert_to_12d(self, merged: TrajResult) -> Dict[str, torch.Tensor]:
        """Convert raw 11-DOF traj data to IsaacLab-Arena 12-DOF format.

        Operation per-trajectory:
            robot_joints = trajectory[..., :10]
            obj_state    = trajectory[..., 10:]    (negated)
            optional robot-joint offset applied if explicitly configured
            gripper columns appended (2 DOF)
            Grasp phase prepended (gripper closing from open → closed)
        """
        gripper_open = self.output_cfg.get("gripper_open", 0.04)
        gripper_closed = self.output_cfg.get("gripper_closed", 0.0)
        prepend_steps = self.output_cfg.get("prepend_grasp_steps", 4)
        lift_offset = self.output_cfg.get("lift_joint_offset", 0.0)

        def _split(t: torch.Tensor):
            arm = t[..., :10].clone()
            obj = t[..., 10:].clone()
            # Summit-Franka trajectories use base DOFs first, so no offset
            # should be applied unless explicitly configured for another layout.
            arm[..., 1] += lift_offset
            obj = -obj
            return arm, obj

        def _state(t: torch.Tensor, grip_val: float):
            arm, obj = _split(t)
            pad = list(arm.shape[:-1]) + [2]
            grip = torch.full(pad, grip_val, dtype=t.dtype)
            return torch.cat([arm, grip], dim=-1), obj

        def _traj(t: torch.Tensor):
            B, T, _ = t.shape
            arm, obj = _split(t)

            # Grasp phase: hold frame-0, close gripper linearly
            g_arm = arm[:, 0:1].repeat(1, prepend_steps, 1)
            g_obj = obj[:, 0:1].repeat(1, prepend_steps, 1)
            closing = torch.linspace(gripper_open, gripper_closed, steps=prepend_steps, dtype=t.dtype)
            g_grip = closing.view(1, -1, 1).repeat(B, 1, 2)
            g_robot = torch.cat([g_arm, g_grip], dim=-1)

            # Pull phase: move arm, gripper closed
            p_grip = torch.full((B, T, 2), gripper_closed, dtype=t.dtype)
            p_robot = torch.cat([arm, p_grip], dim=-1)

            return (
                torch.cat([g_robot, p_robot], dim=1),
                torch.cat([g_obj, obj], dim=1),
            )

        out = {}
        out["start_robot"], out["start_obj"] = _state(merged.start_states, gripper_open)
        out["goal_robot"], out["goal_obj"] = _state(merged.goal_states, gripper_closed)
        out["traj_robot"], out["traj_obj"] = _traj(merged.trajectories)
        out["traj_success"] = merged.success
        return out

    # ====================================================================
    # Helpers
    # ====================================================================

    def _build_scene_cfg(self, scene_dir: str, scene_name: str) -> Dict[str, Any]:
        scene_cfg_override = self.cfg.get("scene", {})
        scene_type = next(iter(scene_cfg_override))  # e.g. "infinigen"
        scene_defaults = scene_cfg_override[scene_type]
        usd_subpath = scene_defaults.get(
            "usd_subpath",
            "export/export_scene.blend/export_scene.usdc",
        )
        metadata_subpath = scene_defaults.get("metadata_subpath", "info/metadata.json")
        return {
            "path": os.path.join(scene_dir, scene_name, usd_subpath),
            "pose": scene_defaults.get("pose", [0, 0, 0, 1, 0, 0, 0]),
            "metadata_path": os.path.join(scene_dir, scene_name, metadata_subpath),
        }

    def _build_object_cfg(self, object_id: str, scene_cfg: Dict) -> Dict[str, Any]:
        obj_meta = self.cfg["objects"][object_id]
        obj_cfg = {
            "path": obj_meta["urdf_path"],
            "asset_type": obj_meta["asset_type"],
            "asset_id": object_id,
        }
        return load_object_from_metadata(scene_cfg["metadata_path"], obj_cfg)

    def _verify(self, converted: Dict[str, torch.Tensor]) -> None:
        """Print quick sanity checks on the converted output."""
        prepend = self.output_cfg.get("prepend_grasp_steps", 4)
        gripper_open = self.output_cfg.get("gripper_open", 0.04)
        gripper_closed = self.output_cfg.get("gripper_closed", 0.0)

        if converted["traj_robot"].shape[0] == 0:
            print("  Verify: no successful trajectories to inspect")
            return
        g = converted["traj_robot"][0, :, 10]  # left gripper of first traj
        print(f"  Verify: step 0 grip={g[0]:.4f} (expect {gripper_open})")
        print(f"  Verify: step {prepend - 1} grip={g[prepend - 1]:.4f} (expect ~{gripper_closed})")
        print(f"  Verify: step {prepend} grip={g[prepend]:.4f} (expect {gripper_closed})")
        print(f"  Verify: last grip={g[-1]:.4f} (expect {gripper_closed})")
