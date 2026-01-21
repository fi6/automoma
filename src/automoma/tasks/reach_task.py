"""
Reach task for reaching to grasp positions on articulated objects.

This module implements the ReachTask class for reaching to a handle/grasp
position on an articulated object. Unlike OpenTask which opens the object,
ReachTask only moves the robot to reach the grasp position.

Pipeline Overview:
    Planning:
        1. Load grasp poses from object directory
        2. Generate collision-free initial IKs by sampling random base poses
        3. Solve goal IKs at the grasp position (handle at start angle)
        4. Plan trajectories from initial to goal positions
        5. Filter trajectories (exclude base rotations exceeding full circle)
    
    Recording:
        1. Load successful trajectories
        2. Replay in simulation, recording observations and actions
        3. Save to LeRobot format dataset (no object articulation)
    
    Evaluation:
        1. Load test initial states (random positions around object)
        2. Run policy inference to get actions  
        3. Execute actions and measure success (EE reached target position)
"""

import os
import logging
import math
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np
from pathlib import Path

from automoma.core.types import TaskType, StageType, IKResult, TrajResult
from automoma.core.config_loader import Config
from automoma.tasks.base_task import BaseTask, TaskResult, StageResult, MAX_IK_ITERATIONS
from automoma.utils.robot_utils import adjust_pose_for_robot
from automoma.utils.type_utils import to_list, to_tensor


logger = logging.getLogger(__name__)


class ReachTask(BaseTask):
    """
    Reach task for moving to a grasp position.
    
    This task involves moving the robot from a random initial position
    to a grasp position on the object handle.
    
    Key differences from OpenTask:
    1. Single MOVE stage (not MOVE_ARTICULATED)
    2. Start IKs are collision-free random positions around the object
    3. Goal IKs are at the grasp position (object at start angle)
    4. Trajectory filtering excludes base rotations exceeding one full circle
    5. No object articulation during recording/evaluation
    """
    
    STAGES = [StageType.MOVE]
    TASK_TYPE = TaskType.REACH
    
    def __init__(self, cfg: Config):
        """Initialize reach task."""
        super().__init__(cfg)
        self.name = "reach"
    
    def get_grasp_poses(self, object_cfg: Config, **kwargs) -> List[List[float]]:
        """
        Get grasp poses for the object handle.
        
        Args:
            object_cfg: Object configuration
            
        Returns:
            List of grasp poses
        """
        from automoma.utils.file_utils import get_grasp_poses
        
        # Get grasp directory
        grasp_dir = object_cfg.grasp_dir
        if not grasp_dir:
            grasp_dir = f"assets/object/{object_cfg.asset_type}/{object_cfg.asset_id}/grasp"
        
        grasp_dir = str(self.project_root / grasp_dir)
        
        # Get plan_cfg for this object
        object_id = object_cfg.asset_id
        plan_cfg = self.get_plan_cfg(object_id)
        num_grasps = plan_cfg.num_grasps
        scale = object_cfg.scale if object_cfg.scale else 1.0
        
        return get_grasp_poses(
            grasp_dir=grasp_dir,
            num_grasps=num_grasps,
            scaling_factor=scale,
        )
    
    def get_target_pose_for_stage(
        self,
        stage_index: int,
        grasp_pose: List[float],
        angle: float,
        object_cfg: Config,
    ) -> torch.Tensor:
        """
        Get target end-effector pose for reaching the grasp position.
        
        For reach task, the target is the grasp pose at the specified angle
        (usually start_angle = 0, meaning object is closed).
        
        Args:
            stage_index: Stage index (always 0 for single-stage reach)
            grasp_pose: Base grasp pose on handle
            angle: Current articulation angle
            object_cfg: Object configuration
            
        Returns:
            Target pose tensor [x, y, z, qw, qx, qy, qz]
        """
        from automoma.utils.math_utils import get_open_ee_pose
        from curobo.types.math import Pose
        
        # Get object pose from planner
        object_pose = Pose.from_list(self.planner.object_pose)
        grasp_Pose = Pose.from_list(grasp_pose)
        
        # Default and current joint configuration
        default_joint_cfg = {"joint_0": 0.0}
        joint_cfg = {"joint_0": angle}
        
        # Compute target pose at the grasp position
        target_Pose = get_open_ee_pose(
            object_pose=object_pose,
            grasp_pose=grasp_Pose,
            object_urdf=self.planner.object_urdf,
            handle="link_0",
            joint_cfg=joint_cfg,
            default_joint_cfg=default_joint_cfg,
        )
        
        return torch.tensor(target_Pose.to_list())
    
    def get_stage_type(self, stage_index: int) -> StageType:
        """Get stage type - always MOVE for reach task."""
        return StageType.MOVE
    
    def plan_ik_for_stage(
        self,
        stage_index: int,
        grasp_pose: List[float],
        angles: List[float],
        object_cfg: Config,
        is_start: bool = True,
    ) -> IKResult:
        """
        Plan IK solutions for the reach stage.
        
        For ReachTask:
        - Start IKs: Generated via collision-free random sampling around object
        - Goal IKs: Solved at the grasp position (object at start angle)
        
        Args:
            stage_index: Index of the stage
            grasp_pose: Grasp pose [x, y, z, qw, qx, qy, qz]
            angles: Joint angles to sample
            object_cfg: Object configuration
            is_start: Whether this is for start or goal IKs
            
        Returns:
            IKResult with IK solutions
        """
        object_id = object_cfg.asset_id
        plan_cfg = self.get_plan_cfg(object_id)
        ik_limit = plan_cfg.plan_ik.limit[0] if is_start else plan_cfg.plan_ik.limit[1]
        
        if is_start:
            # Generate collision-free initial IKs with random base poses
            return self._generate_collision_free_initial_iks(
                object_cfg=object_cfg,
                plan_cfg=plan_cfg,
                ik_limit=ik_limit,
            )
        else:
            # Plan goal IKs at grasp position
            return self._plan_goal_iks(
                grasp_pose=grasp_pose,
                angles=angles,
                object_cfg=object_cfg,
                plan_cfg=plan_cfg,
                ik_limit=ik_limit,
            )
    
    def _generate_collision_free_initial_iks(
        self,
        object_cfg: Config,
        plan_cfg: Config,
        ik_limit: int,
    ) -> IKResult:
        """
        Generate collision-free initial IK solutions with random base poses.
        
        This samples random base positions around the object and uses
        the default arm configuration. Only collision-free poses are kept.
        
        Args:
            object_cfg: Object configuration
            plan_cfg: Planning configuration
            ik_limit: Maximum number of IKs to generate
            
        Returns:
            IKResult with collision-free initial IK solutions
        """
        from curobo.types.robot import JointState
        
        # Get initial IK generation parameters
        initial_ik_cfg = getattr(plan_cfg, 'initial_ik', None)
        if initial_ik_cfg is None:
            initial_ik_cfg = Config({
                'num_samples': ik_limit,
                'radius_range': [0.5, 1.5],
                'angle_range': [5/8 * math.pi, 11/8 * math.pi],
                'theta_noise_std': 0.1,
            })
        
        # Ensure world is updated for collision checking before sampling
        start_angles = getattr(object_cfg, 'start_angles', [0.0])
        self.planner._update_world_collision(
            self.motion_gen,
            {"joint_0": start_angles[0]},
            enable_collision=True
        )
        
        num_samples = getattr(initial_ik_cfg, 'num_samples', ik_limit)
        radius_range = getattr(initial_ik_cfg, 'radius_range', [0.5, 1.5])
        angle_range = getattr(initial_ik_cfg, 'angle_range', [5/8 * math.pi, 11/8 * math.pi])
        theta_noise_std = getattr(initial_ik_cfg, 'theta_noise_std', 0.1)
        
        # Get robot retract config (default arm pose)
        retract_config = self.planner.tensor_args.to_device(
            torch.tensor(self.robot_cfg["kinematics"]["cspace"]["retract_config"])
        )
        default_arm_joints = retract_config[3:10]  # 7 arm joints for summit_franka
        
        # Get object position
        object_pose = self.planner.object_pose
        object_x, object_y = object_pose[0], object_pose[1]
        
        # Apply y-offset fix for summit_franka (from reference code)
        retract_config_adjusted = retract_config.clone()
        retract_config_adjusted[1] += 1.5  # Y offset fix
        
        collision_free_iks = []
        batch_size = 20
        max_attempts = num_samples * 10
        attempts = 0
        
        logger.info(f"Generating {num_samples} collision-free initial IKs...")
        
        with tqdm(total=num_samples, desc="Finding collision-free poses") as pbar:
            while len(collision_free_iks) < num_samples and attempts < max_attempts:
                current_batch_size = min(batch_size, num_samples - len(collision_free_iks))
                
                # Generate random base poses
                base_poses = self._generate_random_base_poses(
                    num_samples=current_batch_size,
                    object_x=object_x,
                    object_y=object_y,
                    radius_range=radius_range,
                    angle_range=angle_range,
                    theta_noise_std=theta_noise_std,
                )
                
                # Convert to full IK solutions
                batch_iks = self._base_poses_to_iks(
                    base_poses=base_poses,
                    default_arm_joints=default_arm_joints,
                    device=retract_config.device,
                    y_offset=1.5,  # Summit franka fix
                )
                
                # Check collision
                collision_free_mask = self.motion_gen.ik_solver.check_valid(batch_iks)
                valid_iks = batch_iks[collision_free_mask]
                
                for ik in valid_iks:
                    if len(collision_free_iks) < num_samples:
                        collision_free_iks.append(ik.unsqueeze(0))
                        pbar.update(1)
                
                attempts += current_batch_size
        
        if len(collision_free_iks) < num_samples:
            logger.warning(f"Only found {len(collision_free_iks)}/{num_samples} collision-free poses")
        else:
            logger.info(f"Generated {len(collision_free_iks)} collision-free initial poses")
        
        if not collision_free_iks:
            # Return empty result
            robot_dof = 10  # summit_franka: 3 base + 7 arm
            return IKResult(
                target_poses=torch.empty((0, 7)),
                iks=torch.empty((0, robot_dof))
            )
        
        initial_iks = torch.cat(collision_free_iks, dim=0)
        
        return IKResult(
            target_poses=torch.zeros((initial_iks.shape[0], 7)),  # No specific target for initial
            iks=initial_iks.cpu()
        )
    
    def _generate_random_base_poses(
        self,
        num_samples: int,
        object_x: float,
        object_y: float,
        radius_range: List[float],
        angle_range: List[float],
        theta_noise_std: float,
    ) -> List[List[float]]:
        """
        Generate random base poses around the object.
        
        Args:
            num_samples: Number of random base poses to generate
            object_x: Object X position
            object_y: Object Y position
            radius_range: [min, max] distance from object center
            angle_range: [min, max] angle range in radians
            theta_noise_std: Standard deviation for Gaussian noise on theta
            
        Returns:
            List of 3D base poses [base_x, base_y, theta]
        """
        base_poses = []
        
        for _ in range(num_samples):
            # Sample radius and angle
            radius = np.random.uniform(radius_range[0], radius_range[1])
            angle = np.random.uniform(angle_range[0], angle_range[1])
            
            # Calculate base position
            base_x = object_x + radius * np.cos(angle)
            base_y = object_y + radius * np.sin(angle)
            
            # Calculate orientation towards object with Gaussian noise
            base_theta = np.arctan2(object_y - base_y, object_x - base_x)
            theta = base_theta + np.random.normal(0, theta_noise_std)
            
            base_poses.append([base_x, base_y, theta])
        
        return base_poses
    
    def _base_poses_to_iks(
        self,
        base_poses: List[List[float]],
        default_arm_joints: torch.Tensor,
        device: torch.device,
        y_offset: float = 0.0,
    ) -> torch.Tensor:
        """
        Convert base poses to full IK solutions by concatenating with arm joints.
        
        Args:
            base_poses: List of [base_x, base_y, theta] poses
            default_arm_joints: Default arm joint configuration
            device: Device to create tensors on
            y_offset: Y offset to apply (summit_franka fix)
            
        Returns:
            Tensor of IK solutions [num_poses, dof]
        """
        ik_solutions = []
        
        for base_pose in base_poses:
            base_tensor = torch.tensor(base_pose, dtype=torch.float32, device=device)
            base_tensor[1] += y_offset  # Apply y offset
            ik = torch.cat([base_tensor, default_arm_joints], dim=0)
            ik_solutions.append(ik.unsqueeze(0))
        
        return torch.cat(ik_solutions, dim=0)
    
    def _plan_goal_iks(
        self,
        grasp_pose: List[float],
        angles: List[float],
        object_cfg: Config,
        plan_cfg: Config,
        ik_limit: int,
    ) -> IKResult:
        """
        Plan goal IK solutions at the grasp position.
        
        Args:
            grasp_pose: Grasp pose [x, y, z, qw, qx, qy, qz]
            angles: Joint angles to sample
            object_cfg: Object configuration
            plan_cfg: Planning configuration
            ik_limit: Maximum number of IKs
            
        Returns:
            IKResult with goal IK solutions
        """
        from automoma.utils.math_utils import stack_iks_angle
        
        all_ik_results = []
        
        logger.info(f"Planning goal IKs at angles: {angles}")
        
        for iteration in range(MAX_IK_ITERATIONS):
            for angle in angles:
                target_pose = self.get_target_pose_for_stage(
                    0, grasp_pose, angle, object_cfg
                )
                
                # Use default motion_gen for IK planning
                ik_result = self.planner.plan_ik(
                    target_pose=target_pose,
                    robot_cfg=self.robot_cfg,
                    plan_cfg={"joint_cfg": {"joint_0": angle}, "enable_collision": True},
                    motion_gen=self.motion_gen,
                )
                
                # Apply clustering
                ik_result = self.planner.ik_clustering(
                    ik_result,
                    ap_fallback_clusters=plan_cfg.cluster.ap_fallback_clusters,
                    ap_clusters_upperbound=plan_cfg.cluster.ap_clusters_upperbound,
                    ap_clusters_lowerbound=plan_cfg.cluster.ap_cluster_lowerbound,
                )
                
                all_ik_results.append(ik_result)
            
            total_iks = sum(r.iks.shape[0] for r in all_ik_results)
            if total_iks >= ik_limit:
                break
        
        result = IKResult.cat(all_ik_results)
        result = result.downsample(ik_limit)
        
        if result.iks.shape[0] == 0:
            robot_dof = 10
            logger.warning(f"No goal IK solutions found, returning empty result")
            return IKResult(
                target_poses=torch.empty((0, 7)),
                iks=torch.empty((0, robot_dof))
            )
        
        return result
    
    def _plan_trajectories(
        self,
        start_iks: IKResult,
        goal_iks: IKResult,
        stage_type: StageType,
        grasp_id: Optional[int] = None,
        object_id: Optional[str] = None,
    ) -> TrajResult:
        """
        Plan trajectories for the reach task.
        
        Uses the standard robot config (not AKR) since we're not manipulating
        the articulated object.
        
        Args:
            start_iks: Start IK solutions (collision-free initial positions)
            goal_iks: Goal IK solutions (grasp positions)
            stage_type: Type of stage (MOVE)
            grasp_id: Grasp ID (not used for reach task)
            object_id: Object ID
            
        Returns:
            TrajResult with planned trajectories
        """
        if object_id is None and self.cfg.env_cfg and self.cfg.env_cfg.object_cfg:
            object_id = list(self.cfg.env_cfg.object_cfg.keys())[0]
        
        plan_cfg = self.get_plan_cfg(object_id)
        traj_cfg = {
            "stage_type": stage_type,
            "batch_size": plan_cfg.plan_traj.batch_size,
            "expand_to_pairs": plan_cfg.plan_traj.expand_to_pairs,
        }
        
        # Use default motion_gen for trajectory planning (not AKR)
        return self.planner.plan_traj(
            start_iks=start_iks.iks,
            goal_iks=goal_iks.iks,
            robot_cfg=self.robot_cfg,
            plan_cfg=traj_cfg,
            motion_gen=self.motion_gen,
        )
    
    def _filter_trajectories(
        self,
        traj_result: TrajResult,
        stage_type: StageType,
        object_id: Optional[str] = None,
    ) -> TrajResult:
        """
        Filter trajectories for the reach task.
        
        Args:
            traj_result: Trajectory results to filter
            stage_type: Type of stage
            object_id: Object ID
            
        Returns:
            Filtered TrajResult
        """
        if object_id is None and self.cfg.env_cfg and self.cfg.env_cfg.object_cfg:
            object_id = list(self.cfg.env_cfg.object_cfg.keys())[0]
        
        plan_cfg = self.get_plan_cfg(object_id)
        
        plan_cfg.filter['stage_type'] = stage_type  # Pass stage type to filter
        
        # Apply standard filtering (includes base rotation filter in CuroboPlanner)
        filtered_result = self.planner.filter_traj(
            traj_result=traj_result,
            robot_cfg=self.robot_cfg,
            motion_gen=self.motion_gen,
            filter_cfg=plan_cfg.filter,
        )
        
        return filtered_result
    
    
    
    def run_planning_pipeline(
        self,
        scene_name: str,
        object_id: str,
        object_cfg: Config,
    ) -> TaskResult:
        """
        Run the complete planning pipeline for a scene/object.
        
        For reach task:
        - Stage 0 (MOVE): Random initial positions -> Grasp position
        
        Args:
            scene_name: Scene identifier
            object_id: Object identifier
            object_cfg: Object configuration
            
        Returns:
            TaskResult with stage results
        """
        from automoma.utils.file_utils import save_ik, save_traj
        
        grasp_poses = self.get_grasp_poses(object_cfg)
        
        task_result = TaskResult(task_name=self.name)
        
        # Check if resume is enabled
        resume = getattr(self.cfg.plan_cfg, 'resume', True)
        
        for grasp_idx, grasp_pose in enumerate(grasp_poses):
            # Check if this grasp has already been planned
            if resume and self._check_planning_complete(scene_name, object_id, grasp_idx):
                logger.info(f"Skipping grasp {grasp_idx + 1}/{len(grasp_poses)} (already planned)")
                continue
            
            logger.info(f"Processing grasp {grasp_idx + 1}/{len(grasp_poses)}")
            
            stage_type = self.STAGES[0]  # MOVE
            stage_result = StageResult(
                stage_type=stage_type,
                stage_index=0,
            )
            
            # Get angles (for reach, we only use start angles since we're reaching to closed position)
            start_angles, goal_angles = self._get_stage_angles(0, object_cfg)
            
            # Plan start IKs (collision-free random positions)
            stage_result.start_iks = self.plan_ik_for_stage(
                0, grasp_pose, start_angles, object_cfg, is_start=True
            )
            
            # Plan goal IKs (at grasp position)
            stage_result.goal_iks = self.plan_ik_for_stage(
                0, grasp_pose, start_angles, object_cfg, is_start=False
            )
            
            # Plan trajectories
            if stage_result.start_iks.iks.shape[0] > 0 and stage_result.goal_iks.iks.shape[0] > 0:
                stage_result.traj_result = self._plan_trajectories(
                    stage_result.start_iks,
                    stage_result.goal_iks,
                    stage_type,
                    grasp_id=grasp_idx,
                    object_id=object_id,
                )
                print(f"Trajectory Planning completed:")
                print(f"  Trajectories: {stage_result.traj_result.trajectories.shape}")
                print(f"  Successes: {stage_result.traj_result.success.sum().item()}/{stage_result.traj_result.success.shape[0]}")
                
                # Filter trajectories
                if stage_result.traj_result is not None and stage_result.traj_result.success.sum() > 0:
                    stage_result.traj_result = self._filter_trajectories(
                        stage_result.traj_result,
                        stage_type,
                        object_id=object_id,
                    )
                    print(f"Trajectory Filtering completed:")
                    print(f"  Trajectories: {stage_result.traj_result.trajectories.shape}")
                    print(f"  Successes: {stage_result.traj_result.success.sum().item()}/{stage_result.traj_result.success.shape[0]}")
                
                stage_result.success = stage_result.traj_result.success.sum() > 0 if stage_result.traj_result else False
            else:
                print(f"Skipping trajectory planning for grasp {grasp_idx}: "
                             f"start_iks={stage_result.start_iks.iks.shape[0]}, "
                             f"goal_iks={stage_result.goal_iks.iks.shape[0]}")
                stage_result.success = False
            
            # Save stage results
            stage_output_dir = os.path.join(
                self.output_dir, "traj",
                self.cfg.env_cfg.robot_cfg.robot_type,
                scene_name, object_id,
                f"grasp_{grasp_idx:04d}",
                "stage_0",
            )
            os.makedirs(stage_output_dir, exist_ok=True)
            stage_result.output_dir = stage_output_dir
            
            save_ik(stage_result.start_iks, os.path.join(stage_output_dir, "start_iks.pt"))
            save_ik(stage_result.goal_iks, os.path.join(stage_output_dir, "goal_iks.pt"))
            if stage_result.traj_result is not None:
                save_traj(stage_result.traj_result, os.path.join(stage_output_dir, "traj_data.pt"))
            
            task_result.stages.append(stage_result)
            
            if stage_result.traj_result is not None:
                task_result.total_trajectories += stage_result.traj_result.success.shape[0]
                task_result.successful_trajectories += stage_result.traj_result.success.sum().item()
        
        task_result.success = all(s.success for s in task_result.stages) if task_result.stages else False
        return task_result
    
    def _get_stage_angles(self, stage_index: int, object_cfg: Config) -> Tuple[List[float], List[float]]:
        """
        Get start and goal angles for the reach stage.
        
        For reach task, we only need start angles since we're reaching
        to the closed position (not opening the object).
        """
        start_angles = object_cfg.start_angles if object_cfg.start_angles else [0.0]
        # Goal angles same as start for reach (object doesn't move)
        return start_angles, start_angles
    
    def _check_task_complete(self, obs: Dict[str, Any]) -> bool:
        """
        Check if the reach task is complete.
        
        Task is complete when end-effector is close to target position.
        """
        if "ee_pose" not in obs or "target_pose" not in obs:
            return False
        
        ee_pose = obs["ee_pose"][:3]  # Position only
        target_pose = obs["target_pose"][:3]
        
        position_error = np.linalg.norm(np.array(ee_pose) - np.array(target_pose))
        
        return position_error < 0.05  # 5cm tolerance
    
    # ==================== Recording Pipeline ====================
    def run_recording_pipeline(self, traj_dir, dataset_wrapper):
        """Run recording pipeline for reach task."""
        return super().run_recording_pipeline(traj_dir, dataset_wrapper)
    
    def run_recording_pipeline_with_trajs(
        self,
        trajectories: torch.Tensor,
        dataset_wrapper,
    ) -> int:
        """
        Run the recording pipeline with pre-loaded trajectories.
        
        For reach task, no object articulation is needed.
        
        Args:
            trajectories: Tensor of trajectories to record (N, T, D)
            dataset_wrapper: Dataset wrapper for saving
            
        Returns:
            Number of episodes recorded
        """
        print(f"Recording {len(trajectories)} reach trajectories")
        
        episodes_recorded = 0
        
        for traj in tqdm(trajectories, desc="Recording trajectories"):
            if self._record_trajectory(traj, dataset_wrapper):
                episodes_recorded += 1
        
        return episodes_recorded
    
    def _record_trajectory(self, trajectory: torch.Tensor, dataset_wrapper) -> bool:
        """
        Record a single reach trajectory to the dataset.
        
        Unlike OpenTask, no object articulation is applied.
        
        Args:
            trajectory: Joint trajectory tensor (T, D) - robot DOF only
            dataset_wrapper: Dataset wrapper
            
        Returns:
            True if successful
        """
        if self.env is None:
            return False
        
        self.env.reset()
        
        # Record T-1 frames for T states
        for step_idx in range(len(trajectory) - 1):
            # Current state
            step_data = trajectory[step_idx]
            robot_state = step_data  # All values are robot state
            robot_state = adjust_pose_for_robot(robot_state, self.cfg.env_cfg.robot_cfg.robot_type)
            
            # Next state (for action computation)
            next_step_data = trajectory[step_idx + 1]
            next_robot_state = next_step_data
            next_robot_state = adjust_pose_for_robot(next_robot_state, self.cfg.env_cfg.robot_cfg.robot_type)
            
            # No object articulation for reach task
            
            self.env.set_state(robot_state, env_state=None)
            self.env.step()
            
            # Get data with next state for action computation
            obs_data = self.env.get_data(next_robot_state=next_robot_state)
            obs_data["task"] = self.name
            dataset_wrapper.add(obs_data)
        
        dataset_wrapper.save()
        print(f"Recorded trajectory with {len(trajectory)} steps")
        return True
    
    # ==================== Evaluation Pipeline ====================
    
    def run_evaluation_pipeline(
        self,
        policy_model,
        initial_state_path: str,
        num_episodes: int,
    ) -> Dict[str, Any]:
        """
        Run evaluation pipeline for reach task.
        frame
        Unlike OpenTask, no object articulation is applied.
        Success is measured by EE reaching the target position.
        
        Args:
            policy_model: Trained policy model
            initial_state_path: Path to a start_iks.pt file or a directory
                containing trajectory data for initial states
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics dictionary
        """
        from automoma.evaluation.metrics import MetricsCalculator
        
        # Get evaluation config
        eval_cfg = self.cfg.eval_cfg if hasattr(self.cfg, 'eval_cfg') and self.cfg.eval_cfg else getattr(self.cfg, 'evaluation', None)
        success_threshold = eval_cfg.success_threshold if eval_cfg else 0.05
        max_steps = eval_cfg.max_steps_per_episode if eval_cfg else 60
        
        # Get metrics config
        metrics_config = {}
        if eval_cfg and hasattr(eval_cfg, 'metrics_cfg') and eval_cfg.metrics_cfg:
            metrics_config = eval_cfg.metrics_cfg.to_dict() if hasattr(eval_cfg.metrics_cfg, 'to_dict') else eval_cfg.metrics_cfg

        initial_states = self.get_test_initial_states(initial_state_path)
        
        if not initial_states:
            logger.warning("No test initial states found")
            return {}
        
        metrics_calc = MetricsCalculator(
            success_threshold=success_threshold,
            **metrics_config
        )
        
        logger.info(f"Starting reach evaluation: episodes={num_episodes}, max_steps={max_steps}")
        
        for ep_idx in range(min(num_episodes, len(initial_states))):
            initial_state = initial_states[ep_idx % len(initial_states)]
            
            logger.info(f"\nEpisode {ep_idx + 1}/{num_episodes}")
            
            # Reset environment with initial state (robot only, no env_state)
            robot_state = initial_state
            robot_state = adjust_pose_for_robot(robot_state, self.cfg.env_cfg.robot_cfg.robot_type)
            
            self.env.reset(robot_state, env_state=None)
            
            episode_traj = []
            episode_success = False
            
            for step in range(max_steps):
                obs = self.env.get_data()
                
                # Get action from policy
                response = policy_model.infer_sync(obs)
                
                if not response.success:
                    logger.warning(f"  Inference failed at step {step}: {response.error_message}")
                    break
                
                action = response.action
                
                # Execute action (no object articulation)
                self.env.step(action)
                episode_traj.append(action)
                
                # Check termination
                obs_after = self.env.get_data()
                
                # Debug info
                if step % 10 == 0:
                    ee_pos = obs_after.get("ee_pose", [0, 0, 0])[:3]
                    logger.debug(f"  Step {step}: EE Position = {ee_pos}")

                if self._check_task_complete(obs_after):
                    episode_success = True
                    logger.info(f"  Task completed at step {step + 1}")
                    break
            
            # Add episode to metrics
            metrics_calc.add_episode(
                pred_trajectory=np.array(episode_traj),
                gt_trajectory=np.array(episode_traj),
                completed=episode_success,
            )
            
            logger.info(f"  Steps: {len(episode_traj)}, Success: {episode_success}")
        
        return metrics_calc.compute_metrics()
    
    def get_test_initial_states(self, test_data_dir: str) -> List[torch.Tensor]:
        """
        Load test initial states from saved IK data.
        
        Args:
            test_data_dir: Directory containing test data
            
        Returns:
            List of initial joint state tensors
        """
        from automoma.utils.file_utils import load_ik
        
        initial_states = []
        
        # Search for ik_data.pt or start_iks.pt files
        test_path = Path(test_data_dir)
        ik_files = list(test_path.glob("**/start_iks.pt")) + list(test_path.glob("**/ik_data.pt"))
        
        for ik_file in ik_files:
            try:
                # load_ik returns an IKResult-like object or dict
                res = load_ik(str(ik_file))
                # Depending on implementation, res might be IKResult or dict
                iks = None
                if isinstance(res, dict):
                    iks = res.get("start_iks", res.get("initial_iks", res.get("iks")))
                elif hasattr(res, "iks"):
                    iks = res.iks
                
                if iks is not None and iks.shape[0] > 0:
                    # Convert to list of individual tensors
                    for i in range(iks.shape[0]):
                        initial_states.append(iks[i])
            except Exception as e:
                logger.error(f"Error loading IK file {ik_file}: {e}")
        
        # Shuffle if many states
        if len(initial_states) > 0:
            import random
            random.shuffle(initial_states)
            
        return initial_states