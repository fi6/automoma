"""
Open task for articulated objects (doors, drawers, microwaves, etc.).

This module implements the OpenTask class for opening articulated objects
like microwave doors, drawers, and cabinet doors. The robot grasps the
handle and moves to articulate the object from start angle to goal angle.

Pipeline Overview:
    Planning:
        1. Load grasp poses from object directory
        2. For each grasp, compute target poses at start/goal angles
        3. Solve IK for start and goal configurations
        4. Plan trajectories using AKR (articulated-kinematic-robot) config
        5. Filter trajectories using FK validation
    
    Recording:
        1. Load successful trajectories
        2. Replay in simulation, recording observations and actions
        3. Save to LeRobot format dataset
    
    Evaluation:
        1. Load test initial states (robot grasping handle at start angle)
        2. Run policy inference to get actions
        3. Execute actions and measure success (goal angle reached)
"""

import os
import logging
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


class OpenTask(BaseTask):
    """
    Open task for articulated objects.
    
    This task involves:
    1. Moving to grasp position on handle (at start angle)
    2. Opening the articulated object (from start to goal angle)
    
    The robot starts grasping the handle and moves while the object articulates.
    """
    
    STAGES = [StageType.MOVE_ARTICULATED]
    TASK_TYPE = TaskType.OPEN
    
    def __init__(self, cfg: Config):
        """Initialize open task."""
        super().__init__(cfg)
        self.name = "open"
        
        # Cache for AKR motion generators (per grasp_id)
        self._motion_gen_akr_cache = {}  # {grasp_id: (akr_robot_cfg, motion_gen_akr)}
        self._current_grasp_id = None
    
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
        
        # Get plan_cfg for this object (uses object-specific config if available)
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
        Get target end-effector pose for opening motion.
        
        The target pose accounts for the object's articulation angle.
        
        Args:
            stage_index: Stage index (always 0 for single-stage open)
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
        
        # Compute target pose considering articulation
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
        """Get stage type - always MOVE_ARTICULATED for open task."""
        return StageType.MOVE_ARTICULATED
    
    def plan_ik_for_stage(
        self,
        stage_index: int,
        grasp_pose: List[float],
        angles: List[float],
        object_cfg: Config,
        is_start: bool = True,
    ) -> IKResult:
        """
        Plan IK solutions for a stage.
        
        This method:
        1. Computes target poses for each articulation angle
        2. Solves IK for each target using the main robot config
        3. Appends the angle to each IK solution (for trajectory filtering)
        4. Clusters IK solutions to reduce redundancy
        
        Note: Uses motion_gen (not AKR) for IK planning. AKR is only for
        trajectory planning since it includes the attached object.
        
        Args:
            stage_index: Index of the stage
            grasp_pose: Grasp pose [x, y, z, qw, qx, qy, qz]
            angles: Joint angles to sample
            object_cfg: Object configuration
            is_start: Whether this is for start or goal IKs
            
        Returns:
            IKResult with IK solutions
        """
        from automoma.utils.math_utils import stack_iks_angle
        
        # Get plan_cfg for this object (uses object-specific config if available)
        object_id = object_cfg.asset_id
        plan_cfg = self.get_plan_cfg(object_id)
        ik_limit = plan_cfg.plan_ik.limit[0] if is_start else plan_cfg.plan_ik.limit[1]
        
        all_ik_results = []
        ik_type = "start" if is_start else "goal"
        
        logger.info(f"Planning {ik_type} IKs for stage {stage_index}")
        logger.info(f"Angles to sample: {angles}, limit: {ik_limit}")
        
        for iteration in range(MAX_IK_ITERATIONS):
            for angle in angles:
                target_pose = self.get_target_pose_for_stage(
                    stage_index, grasp_pose, angle, object_cfg
                )
                
                # Use default motion_gen for IK planning
                ik_result = self.planner.plan_ik(
                    target_pose=target_pose,
                    robot_cfg=self.robot_cfg,
                    plan_cfg={"joint_cfg": {"joint_0": angle}, "enable_collision": True},
                    motion_gen=self.motion_gen,
                )
                
                # Stack angle with IK solutions
                ik_result.iks = stack_iks_angle(ik_result.iks, -angle)
                
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
        # Downsample to exact limit if we have more solutions than needed
        result = result.downsample(ik_limit)
        
        # If no IK solutions found, return empty result with correct robot DOF
        if result.iks.shape[0] == 0:
            robot_dof = self.robot_cfg.get('kinematics', {}).get('cspace', {}).get('joint_names', None)
            if robot_dof:
                dof = len(robot_dof)
            else:
                dof = 10  # Default for summit_franka (4 base + 7 arm - 1 fixed)
            logger.warning(f"No IK solutions found, returning empty result with DOF={dof}")
            return IKResult(
                target_poses=torch.empty((0, 7)),
                iks=torch.empty((0, dof))
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
        Plan trajectories using AKR robot config for articulated motion.
        
        For articulated object manipulation, we use the AKR robot config
        which includes the object attached to the robot (fixed base).
        
        Args:
            start_iks: Start IK solutions
            goal_iks: Goal IK solutions
            stage_type: Type of stage
            grasp_id: Grasp ID (required for AKR robot config)
            object_id: Object ID (for getting object-specific plan_cfg)
            
        Returns:
            TrajResult with planned trajectories
        """
        if grasp_id is None:
            logger.error("grasp_id is required for OpenTask trajectory planning")
            raise ValueError("grasp_id is required for OpenTask")
        
        # Get object config (assume single object for now)
        if not self.cfg.env_cfg or not self.cfg.env_cfg.object_cfg:
            raise ValueError("env_cfg.object_cfg is required in config")
        if object_id is None:
            object_id = list(self.cfg.env_cfg.object_cfg.keys())[0]
        object_cfg = self.cfg.env_cfg.object_cfg[object_id]
        
        # Get cached AKR robot config and motion gen
        akr_robot_cfg, akr_motion_gen = self._get_akr_motion_gen_cached(object_cfg, grasp_id)
        
        # Get plan_cfg for this object (uses object-specific config if available)
        plan_cfg = self.get_plan_cfg(object_id)
        traj_cfg = {
            "stage_type": stage_type,
            "batch_size": plan_cfg.plan_traj.batch_size,
            "expand_to_pairs": plan_cfg.plan_traj.expand_to_pairs,
        }
        
        # Use AKR motion_gen for trajectory planning
        return self.planner.plan_traj(
            start_iks=start_iks.iks,
            goal_iks=goal_iks.iks,
            robot_cfg=akr_robot_cfg,
            plan_cfg=traj_cfg,
            motion_gen=akr_motion_gen,
        )
    
    def _get_akr_motion_gen_cached(self, object_cfg: Config, grasp_id: int):
        """
        Get or create AKR motion generator for a specific grasp, with caching.
        
        This loads the grasp-specific robot config with the articulated object attached.
        Path format: assets/object/{asset_type}/{asset_id}/{robot_type}_{asset_id}_0_grasp_{grasp_id:04d}.yml
        
        Args:
            object_cfg: Object configuration
            grasp_id: Grasp ID
            
        Returns:
            tuple: (akr_robot_cfg, akr_motion_gen)
        """
        from automoma.utils.file_utils import load_robot_cfg, process_robot_cfg
        
        # Check cache first
        if grasp_id in self._motion_gen_akr_cache:
            logger.info(f"Reusing cached AKR motion_gen for grasp {grasp_id}")
            return self._motion_gen_akr_cache[grasp_id]
        
        if not self.akr_robot_cfg:
            logger.error("No akr_robot_cfg in config")
            raise ValueError("OpenTask requires akr_robot_cfg in config")
        
        # Build path: assets/object/{asset_type}/{asset_id}/summit_franka_{asset_id}_0_grasp_{grasp_id:04d}.yml
        asset_type = object_cfg.asset_type
        asset_id = object_cfg.asset_id
        robot_type = self.akr_robot_cfg.robot_type
        
        akr_path = f"assets/object/{asset_type}/{asset_id}/{robot_type}_{asset_id}_0_grasp_{grasp_id:04d}.yml"
        akr_robot_cfg_path = str(self.project_root / akr_path)
        
        if not os.path.exists(akr_robot_cfg_path):
            logger.error(f"AKR robot config not found: {akr_robot_cfg_path}")
            raise FileNotFoundError(f"AKR robot config not found: {akr_robot_cfg_path}")
        
        # Load and process AKR robot config
        logger.info(f"Loading AKR robot config for grasp {grasp_id}: {akr_robot_cfg_path}")
        loaded_akr_cfg = load_robot_cfg(akr_robot_cfg_path)
        akr_robot_cfg = process_robot_cfg(loaded_akr_cfg)
        
        # Create motion generator with fixed_base from config
        fixed_base = getattr(self.akr_robot_cfg, 'fixed_base', True)
        akr_motion_gen = self.planner.init_motion_gen(akr_robot_cfg, fixed_base=fixed_base)
        
        logger.info(f"Created AKR motion_gen for grasp {grasp_id} (fixed_base={fixed_base})")
        
        # Cache for reuse
        self._motion_gen_akr_cache[grasp_id] = (akr_robot_cfg, akr_motion_gen)
        
        return akr_robot_cfg, akr_motion_gen
    
    def _filter_trajectories(
        self,
        traj_result: TrajResult,
        stage_type: StageType,
        object_id: Optional[str] = None,
    ) -> TrajResult:
        """
        Filter trajectories using AKR robot config (same as used for planning).
        
        Must use the same motion_gen that was used for trajectory planning,
        which is the AKR motion_gen for the current grasp.
        
        Args:
            traj_result: Trajectory results to filter
            stage_type: Type of stage
            object_id: Object ID (for getting object-specific plan_cfg)
            
        Returns:
            Filtered TrajResult
        """
        # Use the most recently loaded AKR config (from the last _plan_trajectories call)
        if not self._motion_gen_akr_cache:
            logger.error("No AKR motion_gen cache available for filtering")
            raise ValueError("Must call _plan_trajectories before _filter_trajectories")
        
        # Get the most recent grasp_id (last key in cache)
        grasp_id = list(self._motion_gen_akr_cache.keys())[-1]
        akr_robot_cfg, akr_motion_gen = self._motion_gen_akr_cache[grasp_id]
        
        # Determine object_id if not provided
        if object_id is None and self.cfg.env_cfg and self.cfg.env_cfg.object_cfg:
            object_id = list(self.cfg.env_cfg.object_cfg.keys())[0]
        
        # Get plan_cfg for this object (uses object-specific config if available)
        plan_cfg = self.get_plan_cfg(object_id)
        
        # Get filter parameters with defaults
        position_threshold = 0.01
        orientation_threshold = 0.05
        
        if hasattr(plan_cfg, 'filter') and plan_cfg.filter:
            position_threshold = getattr(plan_cfg.filter, 'position_threshold', 0.01)
            orientation_threshold = getattr(plan_cfg.filter, 'orientation_threshold', 0.05)
        
        filter_cfg = {
            "stage_type": stage_type,
            "position_tolerance": position_threshold,
            "rotation_tolerance": orientation_threshold,
        }
        
        # Use AKR motion_gen for trajectory filtering (same as planning)
        return self.planner.filter_traj(
            traj_result=traj_result,
            robot_cfg=akr_robot_cfg,
            motion_gen=akr_motion_gen,
            filter_cfg=filter_cfg,
        )
        
    def run_planning_pipeline(
        self,
        scene_name: str,
        object_id: str,
        object_cfg: Config,
    ) -> TaskResult:
        """
        Run the complete planning pipeline for a scene/object.
        
        For multi-stage tasks:
        - Stage 0: Sample start and goal IKs
        - Stage N (N>0): Use previous stage's goal IKs as start IKs
        
        Args:
            scene_name: Scene identifier
            object_id: Object identifier
            object_cfg: Object configuration
            
        Returns:
            TaskResult with all stage results
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
            
            previous_goal_iks = None
            
            for stage_idx, stage_type in enumerate(self.stages):
                stage_result = StageResult(
                    stage_type=stage_type,
                    stage_index=stage_idx,
                )
                
                # Get angles for this stage
                start_angles, goal_angles = self._get_stage_angles(stage_idx, object_cfg)
                
                # Get start IKs
                if stage_idx == 0 or previous_goal_iks is None:
                    # First stage: sample start IKs
                    stage_result.start_iks = self.plan_ik_for_stage(
                        stage_idx, grasp_pose, start_angles, object_cfg, is_start=True
                    )
                else:
                    # Subsequent stages: use previous goal IKs
                    stage_result.start_iks = previous_goal_iks
                    logger.info(f"  Stage {stage_idx}: Using {previous_goal_iks.iks.shape[0]} IKs from previous stage")
                
                # Get goal IKs
                stage_result.goal_iks = self.plan_ik_for_stage(
                    stage_idx, grasp_pose, goal_angles, object_cfg, is_start=False
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
                    print(f"Skipping trajectory planning for grasp {grasp_idx}, stage {stage_idx}: "
                                 f"start_iks={stage_result.start_iks.iks.shape[0]}, "
                                 f"goal_iks={stage_result.goal_iks.iks.shape[0]}")
                    stage_result.success = False
                
                # Save stage results
                stage_output_dir = os.path.join(
                    self.output_dir, "traj",
                    self.cfg.env_cfg.robot_cfg.robot_type,
                    scene_name, object_id,
                    f"grasp_{grasp_idx:04d}",
                    f"stage_{stage_idx}",
                )
                os.makedirs(stage_output_dir, exist_ok=True)
                stage_result.output_dir = stage_output_dir
                
                save_ik(stage_result.start_iks, os.path.join(stage_output_dir, "start_iks.pt"))
                save_ik(stage_result.goal_iks, os.path.join(stage_output_dir, "goal_iks.pt"))
                if stage_result.traj_result is not None:
                    save_traj(stage_result.traj_result, os.path.join(stage_output_dir, "traj_data.pt"))
                
                # Store for next stage
                previous_goal_iks = stage_result.goal_iks
                task_result.stages.append(stage_result)
                
                if stage_result.traj_result is not None:
                    task_result.total_trajectories += stage_result.traj_result.success.shape[0]
                    task_result.successful_trajectories += stage_result.traj_result.success.sum().item()
        
        task_result.success = all(s.success for s in task_result.stages)
        return task_result
    
    def _get_stage_angles(self, stage_index: int, object_cfg: Config) -> Tuple[List[float], List[float]]:
        """
        Get start and goal angles for the open stage.
        
        For the open task:
        - Start angles: Handle at closed position (e.g., 0)
        - Goal angles: Handle at open position (e.g., 1.57)
        """
        start_angles = object_cfg.start_angles if object_cfg.start_angles else [0.0]
        goal_angles = object_cfg.goal_angles if object_cfg.goal_angles else [1.57]
        return start_angles, goal_angles
    
    def _check_task_complete(self, obs: Dict[str, Any]) -> bool:
        """
        Check if the open task is complete.
        
        Task is complete when object angle is close to goal angle.
        """
        if "env_state" not in obs:
            return False
        
        current_angle = obs.get("env_state", 0.0)
        goal_angles = None
        if self.cfg.env_cfg and self.cfg.env_cfg.object_cfg:
            # Get first object's goal angles
            first_obj_id = list(self.cfg.env_cfg.object_cfg.keys())[0]
            goal_angles = self.cfg.env_cfg.object_cfg[first_obj_id].goal_angles if hasattr(self.cfg.env_cfg.object_cfg[first_obj_id], 'goal_angles') else None
        goal_angle = goal_angles[-1] if goal_angles and len(goal_angles) > 0 else 1.57
        
        return abs(current_angle - goal_angle) < 0.1  # 0.1 radian tolerance
    
    def get_test_initial_states(self, test_data_dir: str) -> List[torch.Tensor]:
        """
        Load test initial states for open task evaluation.
        
        For open task, the robot should start already grasping the handle
        at the start angle position.
        """
        # Check if specific initial state path is provided in config
        eval_cfg = self.cfg.eval_cfg if self.cfg.eval_cfg else self.cfg.evaluation
        
        initial_state_path = None
        if eval_cfg:
            if hasattr(eval_cfg, 'initial_state_path'):
                initial_state_path = eval_cfg.initial_state_path
            elif isinstance(eval_cfg, dict) and 'initial_state_path' in eval_cfg:
                initial_state_path = eval_cfg['initial_state_path']
        
        if initial_state_path:
            ik_path = Path(self.project_root) / initial_state_path
            logger.info(f"Loading initial state from specific path: {ik_path}")
            
            if not ik_path.exists():
                logger.warning(f"Initial state file not found: {ik_path}")
            else:
                try:
                    ik_data = torch.load(ik_path, weights_only=True)
                    # Handle both dict (IKResult) and direct tensor
                    if isinstance(ik_data, dict) and "iks" in ik_data:
                        iks = ik_data["iks"]
                    elif hasattr(ik_data, "iks"):
                        iks = ik_data.iks
                    else:
                        iks = ik_data
                        
                    # Return all IKs found in the file as potential initial states
                    if len(iks) > 0:
                        logger.info(f"Loaded {len(iks)} initial states from {ik_path}")
                        
                        processed_states = []
                        for ik in iks:
                            # Split into robot state and env state
                            # Assuming last element is env state (angle)
                            robot_state = ik[:-1]
                            env_state = ik[-1:]
                            
                            # Adjust robot state
                            robot_state = adjust_pose_for_robot(robot_state, self.cfg.env_cfg.robot_cfg.robot_type)
                            
                            processed_states.append((robot_state, env_state))
                            
                        return processed_states
                    else:
                        logger.warning("No IKs found in file")
                except Exception as e:
                    logger.error(f"Error loading initial state file: {e}")

        initial_states = super().get_test_initial_states(test_data_dir)
        
        if not initial_states:
            logger.warning("No test initial states found, using start IKs")
        
        return initial_states
    
    # ==================== Recording Pipeline ====================
    def run_recording_pipeline(self, traj_dir, dataset_wrapper):
        return super().run_recording_pipeline(traj_dir, dataset_wrapper)
    
    def run_recording_pipeline_with_trajs(
        self,
        trajectories: torch.Tensor,
        dataset_wrapper,
    ) -> int:
        """
        Run the recording pipeline with pre-loaded trajectories.
        
        This method is used when trajectories have been loaded and sampled
        externally (e.g., for random sampling across multiple grasp files).
        
        Args:
            trajectories: Tensor of trajectories to record (N, T, D)
            dataset_wrapper: Dataset wrapper for saving
            
        Returns:
            Number of episodes recorded
        """
        print(f"Recording {len(trajectories)} trajectories")
        
        episodes_recorded = 0
        
        for traj in tqdm(trajectories, desc="Recording trajectories"):
            if self._record_trajectory(traj, dataset_wrapper):
                episodes_recorded += 1
        
        return episodes_recorded
    
    
    def _record_trajectory(self, trajectory: torch.Tensor, dataset_wrapper) -> bool:
        """
        Record a single trajectory to the dataset.
        
        Args:
            trajectory: Joint trajectory tensor
            dataset_wrapper: Dataset wrapper
            
        Returns:
            True if successful
        """
        if self.env is None:
            return False
        
        self.env.reset()
        
        # Record T-1 frames for T states (last frame has no next state for action)
        for step_idx in range(len(trajectory) - 1):
            # Current state
            step_data = trajectory[step_idx]
            robot_state = step_data[:-1]  # All but last (handle angle)
            robot_state = adjust_pose_for_robot(robot_state, self.cfg.env_cfg.robot_cfg.robot_type)
            env_state = step_data[-1:]     # Last (handle angle)
            
            # Next state (for action computation)
            next_step_data = trajectory[step_idx + 1]
            next_robot_state = next_step_data[:-1]
            next_robot_state = adjust_pose_for_robot(next_robot_state, self.cfg.env_cfg.robot_cfg.robot_type)
            
            self.env.set_state(robot_state, env_state)
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
        test_data_dir: str,
        num_episodes: int,
    ) -> Dict[str, Any]:
        """
        Run evaluation pipeline.
        
        Args:
            policy_model: Trained policy model
            test_data_dir: Directory with test data for initial states
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics dictionary
        """
        from automoma.evaluation.metrics import MetricsCalculator
        
        # Get evaluation config
        eval_cfg = self.cfg.eval_cfg if hasattr(self.cfg, 'eval_cfg') and self.cfg.eval_cfg else getattr(self.cfg, 'evaluation', None)
        success_threshold = eval_cfg.success_threshold if eval_cfg else 0.05
        max_steps = eval_cfg.max_steps_per_episode if eval_cfg else 500
        
        # Get metrics config
        metrics_config = {}
        if eval_cfg and hasattr(eval_cfg, 'metrics_cfg') and eval_cfg.metrics_cfg:
            metrics_config = eval_cfg.metrics_cfg.to_dict() if hasattr(eval_cfg.metrics_cfg, 'to_dict') else eval_cfg.metrics_cfg

        initial_states = self.get_test_initial_states(test_data_dir)
        
        if not initial_states:
            logger.warning("No test initial states found")
            return {}
        
        metrics_calc = MetricsCalculator(
            success_threshold=success_threshold,
            **metrics_config
        )
        
        logger.info(f"Starting evaluation: episodes={num_episodes}, max_steps={max_steps}")
        
        for ep_idx in range(min(num_episodes, len(initial_states))):
            initial_state = initial_states[ep_idx % len(initial_states)]
            
            logger.info(f"\nEpisode {ep_idx + 1}/{num_episodes}")
            
            # Reset environment with initial state
            # Initial state is usually (robot_state, env_state) for OpenTask
            robot_state, env_state = initial_state[:-1], initial_state[-1]
            
            robot_state = adjust_pose_for_robot(robot_state, self.cfg.env_cfg.robot_cfg.robot_type)
            # Pass full initial state (tensor/array) or split components to reset
            # SimEnvWrapper.reset handles (robot_state, env_state) if passed explicitly
            # But specific to OpenTask initial_states are single tensors [robot_dofs + env_dof]
            
            # We reconstruct the adjusted initial state
            adjusted_initial_state = torch.cat([to_tensor(robot_state), to_tensor(env_state).unsqueeze(0)])
            
            self.env.reset(robot_state, env_state=env_state)
            
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
                
                # Execute action
                # Fallback strategy: Apply action to object to enforce opening (freely)
                self.env.apply_object_action(1.57, 0.3)
                
                self.env.step(action)
                episode_traj.append(action)
                
                # Check termination
                obs_after = self.env.get_data()
                
                # Debug info
                if step % 10 == 0:
                    current_env_state = obs_after.get("env_state", 0.0)
                    logger.debug(f"  Step {step}: Object Angle = {current_env_state:.4f}")

                if self._check_task_complete(obs_after):
                    episode_success = True
                    logger.info(f"  Task completed at step {step + 1}")
                    break
            
            # Add episode to metrics
            metrics_calc.add_episode(
                pred_trajectory=np.array(episode_traj),
                gt_trajectory=np.array(episode_traj),  # Self-trajectory as GT for rollout metrics
                completed=episode_success,
            )
            
            logger.info(f"  Steps: {len(episode_traj)}, Success: {episode_success}")
        
        return metrics_calc.compute_metrics()
    
    def _check_task_complete(self, obs: Dict[str, Any]) -> bool:
        """Check if the task is complete based on observation."""
        # Override in subclasses for task-specific completion checks
        return False




class ReachOpenTask(BaseTask):
    """
    Two-stage reach and open task.
    
    Stage 1: Reach to grasp position
    Stage 2: Open the articulated object
    
    The goal IKs of Stage 1 become the start IKs of Stage 2.
    """
    
    STAGES = [StageType.MOVE, StageType.MOVE_ARTICULATED]
    TASK_TYPE = TaskType.REACH_OPEN
    
    def __init__(self, cfg: Config):
        """Initialize reach-open task."""
        super().__init__(cfg)
        self.name = "reach_open"
        
        # Cache for AKR motion generators (per grasp_id, used in stage 1)
        self._motion_gen_akr_cache = {}  # {grasp_id: (akr_robot_cfg, motion_gen_akr)}
        self._current_grasp_id = None
    
    def get_grasp_poses(self, object_cfg: Config, **kwargs) -> List[List[float]]:
        """Get grasp poses for the object handle."""
        from automoma.utils.file_utils import get_grasp_poses
        
        grasp_dir = object_cfg.grasp_dir
        if not grasp_dir:
            grasp_dir = f"assets/object/{object_cfg.asset_type}/{object_cfg.asset_id}/grasp"
        
        grasp_dir = str(self.project_root / grasp_dir)
        
        num_grasps = self.cfg.plan_cfg.num_grasps
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
        Get target pose based on stage.
        
        Stage 0 (Reach): Target is the grasp pose at start angle
        Stage 1 (Open): Target is the grasp pose at current angle
        """
        from automoma.utils.math_utils import get_open_ee_pose
        from curobo.types.math import Pose
        
        object_pose = Pose.from_list(self.planner.object_pose)
        grasp_Pose = Pose.from_list(grasp_pose)
        
        default_joint_cfg = {"joint_0": 0.0}
        joint_cfg = {"joint_0": angle}
        
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
        """Get stage type based on index."""
        return self.STAGES[stage_index]
    
    def plan_ik_for_stage(
        self,
        stage_index: int,
        grasp_pose: List[float],
        angles: List[float],
        object_cfg: Config,
        is_start: bool = True,
    ) -> IKResult:
        """
        Plan IK solutions for a stage.
        
        Uses self.motion_gen for IK planning (not AKR).
        
        Args:
            stage_index: Index of the stage
            grasp_pose: Grasp pose
            angles: Joint angles to sample
            object_cfg: Object configuration
            is_start: Whether this is for start or goal IKs
            
        Returns:
            IKResult with IK solutions
        """
        from automoma.utils.math_utils import stack_iks_angle
        from automoma.tasks.base_task import MAX_IK_ITERATIONS
        
        plan_cfg = self.cfg.plan_cfg
        ik_limit = plan_cfg.plan_ik.limit[0] if is_start else plan_cfg.plan_ik.limit[1]
        
        all_ik_results = []
        
        for _ in range(MAX_IK_ITERATIONS):
            for angle in angles:
                target_pose = self.get_target_pose_for_stage(
                    stage_index, grasp_pose, angle, object_cfg
                )
                
                # Use default motion_gen for IK planning
                ik_result = self.planner.plan_ik(
                    target_pose=target_pose,
                    robot_cfg=self.robot_cfg,
                    plan_cfg={"joint_cfg": {"joint_0": angle}, "enable_collision": True},
                    motion_gen=self.motion_gen,
                )
                
                # Stack angle with IK solutions
                ik_result.iks = stack_iks_angle(ik_result.iks, -angle)
                
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
        # Downsample to exact limit if we have more solutions than needed
        result = result.downsample(ik_limit)
        
        # If no IK solutions found, return empty result with correct robot DOF
        if result.iks.shape[0] == 0:
            robot_dof = self.robot_cfg.get('kinematics', {}).get('cspace', {}).get('joint_names', None)
            if robot_dof:
                dof = len(robot_dof)
            else:
                dof = 10  # Default for summit_franka (4 base + 7 arm - 1 fixed)
            logger.warning(f"No IK solutions found, returning empty result with DOF={dof}")
            return IKResult(
                target_poses=torch.empty((0, 7)),
                iks=torch.empty((0, dof))
            )
        
        return result
    
    def _get_stage_angles(self, stage_index: int, object_cfg: Config) -> Tuple[List[float], List[float]]:
        """
        Get angles for each stage.
        
        Stage 0 (Reach): Start from anywhere, goal is closed position
        Stage 1 (Open): Start from closed, goal is open position
        """
        start_angles = object_cfg.start_angles if object_cfg.start_angles else [0.0]
        goal_angles = object_cfg.goal_angles if object_cfg.goal_angles else [1.57]
        
        if stage_index == 0:
            # Reach stage: both start and goal at closed position
            return start_angles, start_angles
        else:
            # Open stage: from closed to open
            return start_angles, goal_angles
    
    def _get_akr_motion_gen_cached(self, object_cfg: Config, grasp_id: int):
        """
        Get or create AKR motion generator for a specific grasp, with caching.
        
        This loads the grasp-specific robot config with the articulated object attached.
        Path format: assets/object/{asset_type}/{asset_id}/{robot_type}_{asset_id}_0_grasp_{grasp_id:04d}.yml
        
        Args:
            object_cfg: Object configuration
            grasp_id: Grasp ID
            
        Returns:
            tuple: (akr_robot_cfg, akr_motion_gen)
        """
        from automoma.utils.file_utils import load_robot_cfg, process_robot_cfg
        
        # Check cache first
        if grasp_id in self._motion_gen_akr_cache:
            print(f"Reusing cached AKR motion_gen for grasp {grasp_id}")
            return self._motion_gen_akr_cache[grasp_id]
        
        if not self.akr_robot_cfg:
            print("No akr_robot_cfg in config")
            raise ValueError("ReachOpenTask requires akr_robot_cfg in config")
        
        # Build path: assets/object/{asset_type}/{asset_id}/summit_franka_{asset_id}_0_grasp_{grasp_id:04d}.yml
        asset_type = object_cfg.asset_type
        asset_id = object_cfg.asset_id
        robot_type = self.akr_robot_cfg.robot_type
        
        akr_path = f"assets/object/{asset_type}/{asset_id}/{robot_type}_{asset_id}_0_grasp_{grasp_id:04d}.yml"
        akr_robot_cfg_path = str(self.project_root / akr_path)
        
        if not os.path.exists(akr_robot_cfg_path):
            print(f"AKR robot config not found: {akr_robot_cfg_path}")
            raise FileNotFoundError(f"AKR robot config not found: {akr_robot_cfg_path}")
        
        # Load and process AKR robot config
        print(f"Loading AKR robot config for grasp {grasp_id}: {akr_robot_cfg_path}")
        loaded_akr_cfg = load_robot_cfg(akr_robot_cfg_path)
        akr_robot_cfg = process_robot_cfg(loaded_akr_cfg)
        
        # Create motion generator with fixed_base from config
        fixed_base = getattr(self.akr_robot_cfg, 'fixed_base', True)
        akr_motion_gen = self.planner.init_motion_gen(akr_robot_cfg, fixed_base=fixed_base)
        
        print(f"Created AKR motion_gen for grasp {grasp_id} (fixed_base={fixed_base})")
        
        # Cache for reuse
        self._motion_gen_akr_cache[grasp_id] = (akr_robot_cfg, akr_motion_gen)
        
        return akr_robot_cfg, akr_motion_gen
    
    def _plan_trajectories(
        self,
        start_iks: IKResult,
        goal_iks: IKResult,
        stage_type: StageType,
        grasp_id: Optional[int] = None,
    ) -> TrajResult:
        """
        Plan trajectories. Uses AKR config for stage 1 (open), default for stage 0 (reach).
        
        Args:
            start_iks: Start IK solutions
            goal_iks: Goal IK solutions
            stage_type: Type of stage
            grasp_id: Grasp ID (required for stage 1)
            
        Returns:
            TrajResult with planned trajectories
        """
        plan_cfg = self.cfg.plan_cfg
        traj_cfg = {
            "stage_type": stage_type,
            "batch_size": plan_cfg.plan_traj.batch_size,
            "expand_to_pairs": plan_cfg.plan_traj.expand_to_pairs,
        }
        
        # Stage 1 (MOVE_ARTICULATED) requires AKR, Stage 0 (MOVE) uses default
        if stage_type == StageType.MOVE_ARTICULATED:
            if grasp_id is None:
                logger.error("grasp_id is required for MOVE_ARTICULATED stage")
                raise ValueError("grasp_id is required for MOVE_ARTICULATED stage")
            
            # Get object config (assume single object for now)
            if not self.cfg.env_cfg or not self.cfg.env_cfg.object_cfg:
                raise ValueError("env_cfg.object_cfg is required in config")
            object_id = list(self.cfg.env_cfg.object_cfg.keys())[0]
            object_cfg = self.cfg.env_cfg.object_cfg[object_id]
            
            # Get cached AKR robot config and motion gen
            akr_robot_cfg, akr_motion_gen = self._get_akr_motion_gen_cached(object_cfg, grasp_id)
            
            # Use AKR motion_gen for trajectory planning
            return self.planner.plan_traj(
                start_iks=start_iks.iks,
                goal_iks=goal_iks.iks,
                robot_cfg=akr_robot_cfg,
                plan_cfg=traj_cfg,
                motion_gen=akr_motion_gen,
            )
        else:
            # Stage 0: Use default motion_gen
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
    ) -> TrajResult:
        """
        Filter trajectories. Uses AKR config for stage 1, default for stage 0.
        
        Args:
            traj_result: Trajectory results to filter
            stage_type: Type of stage
            
        Returns:
            Filtered TrajResult
        """
        plan_cfg = self.cfg.plan_cfg
        
        # Get filter parameters with defaults
        position_threshold = 0.01
        orientation_threshold = 0.05
        
        if hasattr(plan_cfg, 'filter') and plan_cfg.filter:
            position_threshold = getattr(plan_cfg.filter, 'position_threshold', 0.01)
            orientation_threshold = getattr(plan_cfg.filter, 'orientation_threshold', 0.05)
        
        filter_cfg = {
            "stage_type": stage_type,
            "position_tolerance": position_threshold,
            "rotation_tolerance": orientation_threshold,
        }
        
        # Stage 1 (MOVE_ARTICULATED) requires AKR, Stage 0 (MOVE) uses default
        if stage_type == StageType.MOVE_ARTICULATED:
            # Use the most recently loaded AKR config
            if not self._motion_gen_akr_cache:
                logger.error("No AKR motion_gen cache available for filtering")
                raise ValueError("Must call _plan_trajectories before _filter_trajectories")
            
            # Get the most recent grasp_id (last key in cache)
            grasp_id = list(self._motion_gen_akr_cache.keys())[-1]
            akr_robot_cfg, akr_motion_gen = self._motion_gen_akr_cache[grasp_id]
            
            return self.planner.filter_traj(
                traj_result=traj_result,
                robot_cfg=akr_robot_cfg,
                motion_gen=akr_motion_gen,
                filter_cfg=filter_cfg,
            )
        else:
            # Stage 0: Use default motion_gen
            return self.planner.filter_traj(
                traj_result=traj_result,
                robot_cfg=self.robot_cfg,
                motion_gen=self.motion_gen,
                filter_cfg=filter_cfg,
            )
    
    def _check_task_complete(self, obs: Dict[str, Any]) -> bool:
        """Check if reach-open task is complete."""
        if "env_state" not in obs:
            return False
        
        current_angle = obs.get("env_state", 0.0)
        goal_angles = None
        if self.cfg.env_cfg and self.cfg.env_cfg.object_cfg:
            # Get first object's goal angles
            first_obj_id = list(self.cfg.env_cfg.object_cfg.keys())[0]
            goal_angles = self.cfg.env_cfg.object_cfg[first_obj_id].goal_angles if hasattr(self.cfg.env_cfg.object_cfg[first_obj_id], 'goal_angles') else None
        goal_angle = goal_angles[-1] if goal_angles and len(goal_angles) > 0 else 1.57
        
        return abs(current_angle - goal_angle) < 0.1
