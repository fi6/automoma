"""Planning pipeline for motion planning with trajectory optimization."""

import os
import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from automoma.core.types import TaskType, StageType, IKResult, TrajResult, PlanResult
from automoma.core.config import PlanConfig, SceneConfig, ObjectConfig, RobotConfig
from automoma.planning.planner import CuroboPlanner
from automoma.utils.file_utils import (
    load_robot_cfg,
    process_robot_cfg,
    save_ik,
    save_traj,
    load_object_from_metadata,
    get_grasp_poses,
)
from automoma.utils.math_utils import get_open_ee_pose, stack_iks_angle


@dataclass
class PlanningResult:
    """Result from planning pipeline."""
    task_type: TaskType
    grasp_id: int
    start_ik_result: Optional[IKResult] = None
    goal_ik_result: Optional[IKResult] = None
    traj_result: Optional[TrajResult] = None
    filtered_traj_result: Optional[TrajResult] = None
    success: bool = False
    error_message: str = ""


class PlanningPipeline:
    """
    Complete planning pipeline for robot manipulation tasks.
    
    This class handles the end-to-end process of:
    1. Setting up the planning environment
    2. Loading grasp poses
    3. Computing IK solutions
    4. Planning trajectories
    5. Filtering and validating trajectories
    """
    
    def __init__(self, plan_cfg: Dict[str, Any]):
        """Initialize the planning pipeline."""
        self.plan_cfg = plan_cfg
        self.planner = None
        self.motion_gen = None
        self.robot_cfg = None
        
        self.output_dir = plan_cfg.get("output_dir", "data/output")
        self.clustering_params = {
            "ap_fallback_clusters": plan_cfg.get("cluster", {}).get("ap_fallback_clusters", 30),
            "ap_clusters_upperbound": plan_cfg.get("cluster", {}).get("ap_clusters_upperbound", 80),
            "ap_clusters_lowerbound": plan_cfg.get("cluster", {}).get("ap_clusters_lowerbound", 10),
        }
        self.ik_limits = plan_cfg.get("plan_ik", {}).get("limit", [50, 50])
        
    def setup(self, scene_cfg: Dict[str, Any], object_cfg: Dict[str, Any], robot_cfg_path: str) -> None:
        """Setup the planning environment."""
        self.robot_cfg = load_robot_cfg(robot_cfg_path)
        self.robot_cfg = process_robot_cfg(self.robot_cfg)
        
        if "metadata_path" in scene_cfg and os.path.exists(scene_cfg["metadata_path"]):
            object_cfg = load_object_from_metadata(scene_cfg["metadata_path"], object_cfg)
        
        self.object_cfg = object_cfg
        self.scene_cfg = scene_cfg
        
        planner_cfg = {
            "voxel_dims": self.plan_cfg.get("voxel_dims", [5.0, 5.0, 5.0]),
            "voxel_size": self.plan_cfg.get("voxel_size", 0.02),
            "expanded_dims": self.plan_cfg.get("expanded_dims", [1.0, 0.2, 0.2]),
            "collision_checker_type": self.plan_cfg.get("collision_checker_type", "VOXEL"),
        }
        
        from curobo.geom.sdf.world import CollisionCheckerType
        if isinstance(planner_cfg["collision_checker_type"], str):
            planner_cfg["collision_checker_type"] = getattr(
                CollisionCheckerType, planner_cfg["collision_checker_type"]
            )
        
        self.planner = CuroboPlanner(planner_cfg)
        self.planner.setup_env(scene_cfg, object_cfg)
        self.motion_gen = self.planner.init_motion_gen(self.robot_cfg)
        print("Planning pipeline setup complete")
    
    def plan_single_grasp(self, grasp_id: int, grasp_pose: List[float],
                          start_angles: List[float], goal_angles: List[float],
                          akr_robot_cfg_path: Optional[str] = None) -> PlanningResult:
        """Plan trajectories for a single grasp pose."""
        from curobo.types.math import Pose
        
        result = PlanningResult(task_type=TaskType.REACH_OPEN, grasp_id=grasp_id)
        
        try:
            start_ik_results = []
            for _ in range(10):
                for angle in start_angles:
                    ik_result = self._plan_ik_for_angle(grasp_pose, angle)
                    start_ik_results.append(ik_result)
                if sum([r.iks.shape[0] for r in start_ik_results]) >= self.ik_limits[0]:
                    break
            result.start_ik_result = IKResult.cat(start_ik_results)
            
            goal_ik_results = []
            for _ in range(10):
                for angle in goal_angles:
                    ik_result = self._plan_ik_for_angle(grasp_pose, angle)
                    goal_ik_results.append(ik_result)
                if sum([r.iks.shape[0] for r in goal_ik_results]) >= self.ik_limits[1]:
                    break
            result.goal_ik_result = IKResult.cat(goal_ik_results)
            
            if result.start_ik_result.iks.shape[0] == 0:
                result.error_message = "No valid start IK solutions found"
                return result
            if result.goal_ik_result.iks.shape[0] == 0:
                result.error_message = "No valid goal IK solutions found"
                return result
            
            plan_cfg = {
                "stage_type": StageType.MOVE_ARTICULATED,
                "batch_size": self.plan_cfg.get("plan_traj", {}).get("batch_size", 10),
                "expand_to_pairs": self.plan_cfg.get("plan_traj", {}).get("expand_to_pairs", True),
            }
            
            if akr_robot_cfg_path and os.path.exists(akr_robot_cfg_path):
                akr_robot_cfg = load_robot_cfg(akr_robot_cfg_path)
                akr_robot_cfg = process_robot_cfg(akr_robot_cfg)
                motion_gen_akr = self.planner.init_motion_gen(akr_robot_cfg, fixed_base=True)
            else:
                akr_robot_cfg = self.robot_cfg
                motion_gen_akr = self.motion_gen
            
            result.traj_result = self.planner.plan_traj(
                start_iks=result.start_ik_result.iks,
                goal_iks=result.goal_ik_result.iks,
                robot_cfg=akr_robot_cfg,
                plan_cfg=plan_cfg,
                motion_gen=motion_gen_akr,
            )
            
            filter_cfg = {
                "stage_type": StageType.MOVE_ARTICULATED,
                "position_tolerance": self.plan_cfg.get("filter", {}).get("position_tolerance", 0.01),
                "rotation_tolerance": self.plan_cfg.get("filter", {}).get("rotation_tolerance", 0.05),
            }
            
            result.filtered_traj_result = self.planner.filter_traj(
                result.traj_result, robot_cfg=akr_robot_cfg, filter_cfg=filter_cfg, motion_gen=motion_gen_akr
            )
            result.success = result.filtered_traj_result.num_samples > 0
            
        except Exception as e:
            result.error_message = str(e)
            result.success = False
        
        return result
    
    def _plan_ik_for_angle(self, grasp_pose: List[float], angle: float) -> IKResult:
        """Plan IK for a specific joint angle."""
        from curobo.types.math import Pose
        
        default_joint_cfg = {"joint_0": 0.0}
        joint_cfg = {"joint_0": angle}
        
        target_Pose = get_open_ee_pose(
            object_pose=Pose.from_list(self.planner.object_pose),
            grasp_pose=Pose.from_list(grasp_pose),
            object_urdf=self.planner.object_urdf,
            handle="link_0",
            joint_cfg=joint_cfg,
            default_joint_cfg=default_joint_cfg,
        )
        target_pose = torch.tensor(target_Pose.to_list())
        
        ik_result = self.planner.plan_ik(
            target_pose=target_pose,
            robot_cfg=self.robot_cfg,
            plan_cfg={"joint_cfg": joint_cfg, "enable_collision": True},
            motion_gen=self.motion_gen,
        )
        ik_result.iks = stack_iks_angle(ik_result.iks, -angle)
        
        print(f"  Angle {angle:.4f} rad: Found {ik_result.iks.shape[0]} IK solutions")
        ik_result = self.planner.ik_clustering(ik_result, **self.clustering_params)
        print(f"    After clustering: {ik_result.iks.shape[0]} IK solutions")
        
        return ik_result
    
    def run_full_pipeline(self, robot_name: str, scene_name: str, object_id: str,
                          grasp_poses: List[List[float]], start_angles: List[float],
                          goal_angles: List[float]) -> List[PlanningResult]:
        """Run the full planning pipeline for multiple grasp poses."""
        results = []
        
        for grasp_id, grasp_pose in enumerate(grasp_poses):
            print(f"\n{'='*80}\nProcessing Grasp {grasp_id + 1}/{len(grasp_poses)}\n{'='*80}")
            
            akr_robot_cfg_path = f"assets/object/{self.object_cfg['asset_type']}/{object_id}/summit_franka_{object_id}_0_grasp_{grasp_id:04d}.yml"
            
            result = self.plan_single_grasp(
                grasp_id=grasp_id, grasp_pose=grasp_pose,
                start_angles=start_angles, goal_angles=goal_angles,
                akr_robot_cfg_path=akr_robot_cfg_path,
            )
            
            base_dir = os.path.join(self.output_dir, "traj", robot_name, scene_name, object_id, f"grasp_{grasp_id:04d}")
            os.makedirs(base_dir, exist_ok=True)
            
            if result.start_ik_result is not None:
                save_ik(result.start_ik_result, os.path.join(base_dir, "start_iks.pt"))
            if result.goal_ik_result is not None:
                save_ik(result.goal_ik_result, os.path.join(base_dir, "goal_iks.pt"))
            if result.traj_result is not None:
                save_traj(result.traj_result, os.path.join(base_dir, "traj_data.pt"))
            if result.filtered_traj_result is not None:
                save_traj(result.filtered_traj_result, os.path.join(base_dir, "filtered_traj_data.pt"))
            
            results.append(result)
            print(f"Grasp {grasp_id}: {'SUCCESS' if result.success else 'FAILED'}")
            if result.error_message:
                print(f"  Error: {result.error_message}")
        
        return results
