"""Base task class with planning, recording, and evaluation pipelines."""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Type
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch

from automoma.core.types import TaskType, StageType, IKResult, TrajResult
from automoma.core.config_loader import Config


logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Result of a single stage execution."""
    stage_type: StageType
    stage_index: int
    start_iks: Optional[IKResult] = None
    goal_iks: Optional[IKResult] = None
    traj_result: Optional[TrajResult] = None
    success: bool = False
    error_message: str = ""
    output_dir: str = ""


@dataclass  
class TaskResult:
    """Result of complete task execution."""
    task_name: str
    stages: List[StageResult] = field(default_factory=list)
    success: bool = False
    total_trajectories: int = 0
    successful_trajectories: int = 0


class BaseTask(ABC):
    """
    Base class for manipulation tasks.
    
    Each task defines:
    - Stages to execute (e.g., reach, grasp, open)
    - Planning pipeline: How to generate motion plans
    - Recording pipeline: How to record demonstrations
    - Evaluation pipeline: How to evaluate trained policies
    
    Multi-stage tasks automatically chain IK results:
    - Stage N+1 start IKs = Stage N goal IKs
    """
    
    # Class-level stage definition (override in subclasses)
    STAGES: List[StageType] = []
    TASK_TYPE: TaskType = TaskType.PICK_PLACE
    
    def __init__(self, cfg: Config):
        """
        Initialize task from configuration.
        
        Args:
            cfg: Configuration object with attribute access
        """
        self.cfg = cfg
        self.name = cfg.info_cfg.task if cfg.info_cfg else "unknown"
        self.stages = self.STAGES
        
        # Components (initialized during setup)
        self.planner = None
        self.env = None
        self.motion_gen = None
        
        # Results storage
        self.stage_results: List[StageResult] = []
        
    @property
    def output_dir(self) -> str:
        """Get output directory from config."""
        return self.cfg.plan_cfg.output_dir if self.cfg.plan_cfg else "data/output"
    
    @property
    def project_root(self) -> Path:
        """Get project root from config."""
        return Path(self.cfg._project_root) if self.cfg._project_root else Path.cwd()
    
    # ==================== Setup Methods ====================
    
    def setup_planner(self, scene_cfg: Config, object_cfg: Config, robot_cfg: Config) -> None:
        """
        Setup the motion planner.
        
        Args:
            scene_cfg: Scene configuration
            object_cfg: Object configuration  
            robot_cfg: Robot configuration
        """
        from automoma.planning.planner import CuroboPlanner
        from automoma.utils.file_utils import load_robot_cfg, process_robot_cfg
        
        plan_cfg = self.cfg.plan_cfg
        
        planner_cfg = {
            "voxel_dims": plan_cfg.voxel_dims,
            "voxel_size": plan_cfg.voxel_size,
            "expanded_dims": plan_cfg.expand_dims if plan_cfg.expand_dims else [1.0, 0.2, 0.2],
            "collision_checker_type": plan_cfg.collision_checker_type,
        }
        
        self.planner = CuroboPlanner(planner_cfg)
        self.planner.setup_env(scene_cfg.to_dict(), object_cfg.to_dict())
        
        # Load and process robot config
        robot_cfg_path = str(self.project_root / robot_cfg.path)
        loaded_robot_cfg = load_robot_cfg(robot_cfg_path)
        self.robot_cfg = process_robot_cfg(loaded_robot_cfg)
        
        self.motion_gen = self.planner.init_motion_gen(self.robot_cfg)
        logger.info("Planner setup complete")
    
    def setup_env(self, env_wrapper) -> None:
        """
        Setup the environment wrapper.
        
        Args:
            env_wrapper: SimEnvWrapper instance
        """
        self.env = env_wrapper
        logger.info("Environment setup complete")
    
    # ==================== Planning Pipeline ====================
    
    @abstractmethod
    def get_grasp_poses(self, object_cfg: Config, **kwargs) -> List[List[float]]:
        """
        Get grasp poses for the task.
        
        Args:
            object_cfg: Object configuration
            
        Returns:
            List of grasp poses [x, y, z, qw, qx, qy, qz]
        """
        pass
    
    @abstractmethod
    def get_target_pose_for_stage(
        self,
        stage_index: int,
        grasp_pose: List[float],
        angle: float,
        object_cfg: Config,
    ) -> torch.Tensor:
        """
        Get target end-effector pose for a stage.
        
        Args:
            stage_index: Index of the stage
            grasp_pose: Base grasp pose
            angle: Joint angle (for articulated objects)
            object_cfg: Object configuration
            
        Returns:
            Target pose tensor [x, y, z, qw, qx, qy, qz]
        """
        pass
    
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
        
        plan_cfg = self.cfg.plan_cfg
        ik_limit = plan_cfg.plan_ik.limit[0] if is_start else plan_cfg.plan_ik.limit[1]
        
        all_ik_results = []
        
        for _ in range(10):  # Max iterations
            for angle in angles:
                target_pose = self.get_target_pose_for_stage(
                    stage_index, grasp_pose, angle, object_cfg
                )
                
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
        
        return IKResult.cat(all_ik_results)
    
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
        
        for grasp_idx, grasp_pose in enumerate(grasp_poses):
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
                    )
                    stage_result.success = stage_result.traj_result.success.sum() > 0
                
                # Save stage results
                stage_output_dir = os.path.join(
                    self.output_dir, "traj",
                    self.cfg.robot_cfg.robot_type,
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
        """Get start and goal angles for a stage."""
        # Default: use object config angles
        start_angles = object_cfg.start_angles if object_cfg.start_angles else [0.0]
        goal_angles = object_cfg.goal_angles if object_cfg.goal_angles else [1.57]
        return start_angles, goal_angles
    
    def _plan_trajectories(
        self,
        start_iks: IKResult,
        goal_iks: IKResult,
        stage_type: StageType,
    ) -> TrajResult:
        """Plan trajectories between start and goal IKs."""
        plan_cfg = self.cfg.plan_cfg
        
        traj_cfg = {
            "stage_type": stage_type,
            "batch_size": plan_cfg.plan_traj.batch_size,
            "expand_to_pairs": plan_cfg.plan_traj.expand_to_pairs,
        }
        
        return self.planner.plan_traj(
            start_iks=start_iks.iks,
            goal_iks=goal_iks.iks,
            robot_cfg=self.robot_cfg,
            plan_cfg=traj_cfg,
            motion_gen=self.motion_gen,
        )
    
    # ==================== Recording Pipeline ====================
    
    def run_recording_pipeline(
        self,
        traj_dir: str,
        dataset_wrapper,
    ) -> int:
        """
        Run the recording pipeline to create a dataset from trajectories.
        
        Args:
            traj_dir: Directory containing trajectory files
            dataset_wrapper: Dataset wrapper for saving
            
        Returns:
            Number of episodes recorded
        """
        from automoma.utils.file_utils import load_traj
        
        traj_path = Path(traj_dir)
        traj_files = list(traj_path.glob("**/traj_data.pt"))
        
        episodes_recorded = 0
        
        for traj_file in traj_files:
            try:
                traj_data = torch.load(traj_file, weights_only=True)
                trajectories = traj_data["trajectories"]
                success = traj_data["success"]
                
                successful_trajs = trajectories[success.bool()]
                
                for traj in successful_trajs:
                    if self._record_trajectory(traj, dataset_wrapper):
                        episodes_recorded += 1
                        
            except Exception as e:
                logger.warning(f"Error processing {traj_file}: {e}")
        
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
        
        for step_idx in range(len(trajectory)):
            step_data = trajectory[step_idx]
            robot_state = step_data[:-1]  # All but last (handle angle)
            env_state = step_data[-1:]     # Last (handle angle)
            
            self.env.set_state(robot_state, env_state)
            self.env.step()
            
            obs_data = self.env.get_data()
            obs_data["task"] = self.name
            dataset_wrapper.add(obs_data)
        
        dataset_wrapper.save()
        return True
    
    # ==================== Evaluation Pipeline ====================
    
    def get_test_initial_states(self, test_data_dir: str) -> List[torch.Tensor]:
        """
        Load test initial states from trajectory data.
        
        For evaluation, we need to initialize the robot to specific states
        (e.g., already grasping handle for open task).
        
        Args:
            test_data_dir: Directory containing test trajectory data
            
        Returns:
            List of initial state tensors
        """
        initial_states = []
        test_path = Path(test_data_dir)
        
        # Load start IKs from trajectory data
        ik_files = list(test_path.glob("**/start_iks.pt"))
        
        for ik_file in ik_files:
            try:
                ik_data = torch.load(ik_file, weights_only=True)
                iks = ik_data["iks"] if isinstance(ik_data, dict) else ik_data.iks
                
                # Take first IK as initial state
                if len(iks) > 0:
                    initial_states.append(iks[0])
                    
            except Exception as e:
                logger.warning(f"Error loading {ik_file}: {e}")
        
        return initial_states
    
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
        
        initial_states = self.get_test_initial_states(test_data_dir)
        
        if not initial_states:
            logger.warning("No test initial states found")
            return {}
        
        metrics_calc = MetricsCalculator(
            success_threshold=self.cfg.evaluation.success_threshold
        )
        
        for ep_idx in range(min(num_episodes, len(initial_states))):
            initial_state = initial_states[ep_idx % len(initial_states)]
            
            # Set initial state
            self.env.set_state(initial_state)
            self.env.step()
            
            episode_traj = []
            
            for step in range(self.cfg.evaluation.max_steps_per_episode):
                obs = self.env.get_data()
                
                # Get action from policy
                action = policy_model.infer_sync(obs)
                
                if not action.success:
                    break
                
                # Execute action
                self.env.set_state(action.action)
                self.env.step()
                
                episode_traj.append(action.action)
                
                # Check termination
                if self._check_task_complete(obs):
                    break
            
            # Add episode to metrics
            if episode_traj:
                metrics_calc.add_episode(
                    pred_trajectory=np.array(episode_traj),
                    gt_trajectory=np.array(episode_traj),  # Self-trajectory
                    completed=True,
                )
        
        return metrics_calc.compute_metrics()
    
    def _check_task_complete(self, obs: Dict[str, Any]) -> bool:
        """Check if the task is complete based on observation."""
        # Override in subclasses for task-specific completion checks
        return False
    
    # ==================== Abstract Methods for Subclasses ====================
    
    @abstractmethod
    def get_stage_type(self, stage_index: int) -> StageType:
        """Get the stage type for a given index."""
        pass
