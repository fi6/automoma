"""
Base task class with planning, recording, and evaluation pipeline definitions.

This module provides the abstract base class for all manipulation tasks.
Each task subclass must implement its own pipeline methods since different
tasks have different stages, IK sampling strategies, and execution logic.

Architecture:
    BaseTask (abstract)
    ├── run_planning_pipeline()   [abstract]
    ├── run_recording_pipeline()  [abstract] 
    ├── run_evaluation_pipeline() [abstract]
    └── Helper methods for common operations

Example Tasks:
    - OpenTask: Single stage (MOVE_ARTICULATED)
    - PickPlaceTask: Multiple stages (REACH + GRASP + LIFT + MOVE + PLACE)
"""

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


# =============================================================================
# Constants
# =============================================================================

MAX_IK_ITERATIONS = 10  # Maximum iterations for IK sampling loop


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class StageResult:
    """
    Result of a single stage execution.
    
    Attributes:
        stage_type: Type of stage (e.g., MOVE, MOVE_ARTICULATED)
        stage_index: Index of stage in the task
        start_iks: IK solutions at stage start
        goal_iks: IK solutions at stage goal
        traj_result: Planned trajectories
        success: Whether stage completed successfully
        error_message: Error details if failed
        output_dir: Directory where results are saved
    """
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
    """
    Result of complete task execution.
    
    Attributes:
        task_name: Name of the task
        stages: List of individual stage results
        success: Whether entire task completed successfully
        total_trajectories: Total number of trajectories planned
        successful_trajectories: Number of successful trajectories
    """
    task_name: str
    stages: List[StageResult] = field(default_factory=list)
    success: bool = False
    total_trajectories: int = 0
    successful_trajectories: int = 0


# =============================================================================
# Base Task Class
# =============================================================================

class BaseTask(ABC):
    """
    Abstract base class for manipulation tasks.
    
    Each task defines its own:
    - STAGES: List of stage types to execute
    - TASK_TYPE: Type identifier for the task
    - Pipeline methods: How to plan, record, and evaluate
    
    Key Design Principles:
    1. Each subclass implements its own pipeline methods since different
       tasks have fundamentally different logic
    2. Multi-stage tasks chain IK results: Stage N+1 start IKs = Stage N goal IKs
    3. The base class provides common utilities and setup methods
    
    Subclass Requirements:
        Must implement:
        - run_planning_pipeline()
        - run_recording_pipeline()
        - run_evaluation_pipeline()
        - get_grasp_poses()
        - get_target_pose_for_stage()
        - plan_ik_for_stage()
        - get_stage_type()
    """
    
    # Class-level stage definition (override in subclasses)
    STAGES: List[StageType] = []
    TASK_TYPE: TaskType = TaskType.PICK_PLACE
    
    def __init__(self, cfg: Config):
        """
        Initialize task from configuration.
        
        Args:
            cfg: Configuration object with attribute access (e.g., cfg.plan_cfg.num_grasps)
        """
        self.cfg = cfg
        self.name = cfg.info_cfg.task if cfg.info_cfg else "unknown"
        self.stages = self.STAGES
        
        # Components (initialized during setup)
        self.planner = None
        self.env = None
        
        # Motion generators
        self.motion_gen = None  # Primary motion generator
        
        # Robot configurations
        self.robot_cfg = None      # Main robot config for IK planning
        self.akr_robot_cfg = None  # Template for articulated robot config (traj planning)
        
        # Results storage
        self.stage_results: List[StageResult] = []
        
        logger.info(f"Initialized task: {self.name}")
        logger.info(f"Stages: {[s.name for s in self.stages]}")
        
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def output_dir(self) -> str:
        """Get output directory from config."""
        return self.cfg.plan_cfg.output_dir if self.cfg.plan_cfg else "data/output"
    
    @property
    def project_root(self) -> Path:
        """Get project root from config."""
        return Path(self.cfg._project_root) if self.cfg._project_root else Path.cwd()
    
    def get_plan_cfg(self, object_id: Optional[str] = None):
        """
        Get the appropriate plan_cfg for an object.
        
        Args:
            object_id: Object ID. If None, returns the default plan_cfg.
            
        Returns:
            Config object with plan_cfg for the specified object.
            If object has a custom plan_cfg, returns the merged config.
            Otherwise, returns the default plan_cfg.
        """
        if object_id is None:
            return self.cfg.plan_cfg
        
        from automoma.utils.config_utils import get_object_plan_cfg
        return get_object_plan_cfg(self.cfg, object_id)
    
    # =========================================================================
    # Setup Methods
    # =========================================================================
    
    def setup_planner(
        self, 
        scene_cfg: Config, 
        object_cfg: Config, 
        robot_cfg: Config = None
    ) -> None:
        """
        Setup the motion planner with scene, object, and robot.
        
        This method:
        1. Creates the CuroboPlanner instance
        2. Loads the scene and object into the collision world
        3. Loads and processes robot configuration
        4. Initializes the motion generator
        
        Args:
            scene_cfg: Scene configuration (path, pose, metadata)
            object_cfg: Object configuration (path, pose, articulation params)
            robot_cfg: Robot configuration (optional, uses cfg.env_cfg.robot_cfg if None)
        """
        from automoma.planning.planner import CuroboPlanner
        from automoma.utils.file_utils import load_robot_cfg, process_robot_cfg
        
        plan_cfg = self.cfg.plan_cfg
        
        # Create planner config from plan_cfg
        planner_cfg = {
            "voxel_dims": plan_cfg.voxel_dims,
            "voxel_size": plan_cfg.voxel_size,
            "expanded_dims": plan_cfg.expand_dims,
            "collision_checker_type": plan_cfg.collision_checker_type,
        }
        
        logger.info(f"Setting up planner with collision type: {plan_cfg.collision_checker_type}")
        
        # Initialize planner and setup environment
        self.planner = CuroboPlanner(planner_cfg)
        self.planner.setup_env(scene_cfg.to_dict(), object_cfg.to_dict())
        
        # Load robot config
        if robot_cfg is None:
            # Use robot_cfg from env_cfg
            if not self.cfg.env_cfg or not self.cfg.env_cfg.robot_cfg:
                raise ValueError("env_cfg.robot_cfg is required in config")
            robot_cfg = self.cfg.env_cfg.robot_cfg
        robot_cfg_path = str(self.project_root / robot_cfg.path)
        loaded_robot_cfg = load_robot_cfg(robot_cfg_path)
        self.robot_cfg = process_robot_cfg(loaded_robot_cfg)
        
        logger.info(f"Loaded robot config: {robot_cfg_path}")
        
        # Store AKR robot config template (loaded dynamically per grasp if needed)
        if self.cfg.env_cfg and self.cfg.env_cfg.akr_robot_cfg:
            self.akr_robot_cfg = self.cfg.env_cfg.akr_robot_cfg
            logger.info(f"AKR robot config template loaded (robot_type={self.akr_robot_cfg.robot_type})")
        else:
            self.akr_robot_cfg = None
        
        # Initialize default motion generator
        fixed_base = getattr(robot_cfg, 'fixed_base', False)
        self.motion_gen = self.planner.init_motion_gen(self.robot_cfg, fixed_base=fixed_base)
        
        logger.info(f"Planner setup complete (fixed_base={fixed_base})")
    
    def setup_env(self, env_wrapper) -> None:
        """
        Setup the environment wrapper for recording/evaluation.
        
        Args:
            env_wrapper: SimEnvWrapper instance
        """
        self.env = env_wrapper
        logger.info("Environment setup complete")
    
    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def get_grasp_poses(self, object_cfg: Config, **kwargs) -> List[List[float]]:
        """
        Get grasp poses for the task.
        
        Different tasks may load grasps from different sources:
        - OpenTask: Load from object grasp directory
        - PickTask: Sample from point cloud or use learned grasps
        
        Args:
            object_cfg: Object configuration
            **kwargs: Additional task-specific arguments
            
        Returns:
            List of grasp poses, each in format [x, y, z, qw, qx, qy, qz]
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
        
        This computes the goal pose considering:
        - The base grasp pose
        - The current articulation angle (for articulated objects)
        - Stage-specific adjustments (e.g., lift height for pick)
        
        Args:
            stage_index: Index of the stage
            grasp_pose: Base grasp pose [x, y, z, qw, qx, qy, qz]
            angle: Joint angle (for articulated objects)
            object_cfg: Object configuration
            
        Returns:
            Target pose tensor [x, y, z, qw, qx, qy, qz]
        """
        pass
    
    @abstractmethod
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
        
        This should:
        1. Compute target poses for each angle
        2. Solve IK for each target
        3. Cluster IK solutions to reduce redundancy
        4. Return aggregated results
        
        Args:
            stage_index: Index of the stage
            grasp_pose: Grasp pose
            angles: Joint angles to sample
            object_cfg: Object configuration
            is_start: Whether this is for start or goal IKs
            
        Returns:
            IKResult with IK solutions and target poses
        """
        pass
    
    @abstractmethod
    def get_stage_type(self, stage_index: int) -> StageType:
        """
        Get the stage type for a given index.
        
        Args:
            stage_index: Index of the stage
            
        Returns:
            StageType enum value
        """
        pass
    
    @abstractmethod
    def run_planning_pipeline(
        self,
        scene_name: str,
        object_id: str,
        object_cfg: Config,
    ) -> TaskResult:
        """
        Run the complete planning pipeline for a scene/object.
        
        This is the main entry point for motion planning. Each task
        implements its own logic because:
        - Different tasks have different numbers of stages
        - Start IK sampling differs (random, from file, from previous stage)
        - Trajectory planning may use different robot configs per stage
        
        Common pattern for multi-stage tasks:
        1. Stage 0: Sample both start and goal IKs
        2. Stage N (N>0): Use previous stage's goal IKs as start IKs
        
        Args:
            scene_name: Scene identifier (e.g., "scene_0_seed_0")
            object_id: Object identifier (e.g., "7221")
            object_cfg: Object configuration
            
        Returns:
            TaskResult with all stage results
        """
        pass
    
    @abstractmethod
    def run_recording_pipeline(
        self,
        traj_dir: str,
        dataset_wrapper,
    ) -> int:
        """
        Run the recording pipeline to create a dataset from trajectories.
        
        This replays planned trajectories in simulation and records:
        - Camera observations (RGB, depth, point cloud)
        - Robot state (joint positions, gripper state)
        - Actions (state[t+1] - state[t])
        
        Args:
            traj_dir: Directory containing trajectory files
            dataset_wrapper: Dataset wrapper for saving episodes
            
        Returns:
            Number of episodes recorded
        """
        pass
    
    @abstractmethod
    def run_evaluation_pipeline(
        self,
        policy_model,
        initial_state_path: str,
        num_episodes: int,
    ) -> Dict[str, Any]:
        """
        Run evaluation pipeline with a trained policy.
        
        This:
        1. Loads test initial states from trajectory data
        2. Sets robot to initial state (e.g., grasping handle for open task)
        3. Runs policy inference to get actions
        4. Executes actions in simulation
        5. Computes success metrics
        
        Args:
            policy_model: Trained policy model with infer_sync() method
            initial_state_path: Path to a start_iks.pt file or a directory
                containing trajectory data for initial states
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics dictionary
        """
        pass
    
    # =========================================================================
    # Helper Methods (can be used by subclasses)
    # =========================================================================
        
    
    def get_test_initial_states(self, initial_state_path: str) -> List[torch.Tensor]:
        """
        Load test initial states from trajectory data.
        
        For evaluation, we need to initialize the robot to specific states.
        This default implementation loads start IKs from trajectory data.
        Subclasses can override for different initialization strategies.
        
        Args:
            initial_state_path: Path to a start_iks.pt file or a directory
                containing trajectory data.
            
        Returns:
            List of initial state tensors
        """
        initial_states = []
        test_path = Path(initial_state_path)

        # Accept either a single file or a directory
        if test_path.is_file():
            ik_files = [test_path]
        else:
            ik_files = list(test_path.glob("**/start_iks.pt"))

        logger.info(f"Found {len(ik_files)} IK files in {initial_state_path}")
        
        for ik_file in ik_files:
            ik_data = torch.load(ik_file, weights_only=True)
            if isinstance(ik_data, dict) and "iks" in ik_data:
                iks = ik_data["iks"]
            elif hasattr(ik_data, "iks"):
                iks = ik_data.iks
            else:
                iks = ik_data

            # Take first IK as initial state
            initial_states.extend(iks)
        
        logger.info(f"Loaded {len(initial_states)} initial states")
        return initial_states
    
    def _check_task_complete(self, obs: Dict[str, Any]) -> bool:
        """
        Check if the task is complete based on observation.
        
        Override in subclasses for task-specific completion checks.
        
        Args:
            obs: Current observation dictionary
            
        Returns:
            True if task is complete
        """
        return False
    
    def _check_planning_complete(
        self, 
        scene_name: str, 
        object_id: str, 
        grasp_idx: int
    ) -> bool:
        """
        Check if planning is already complete for a specific grasp.
        
        Used for resume functionality - skips already planned grasps.
        
        Args:
            scene_name: Scene identifier
            object_id: Object identifier
            grasp_idx: Grasp index
            
        Returns:
            True if all stage files exist for this grasp
        """
        for stage_idx in range(len(self.stages)):
            # Get robot_type from env_cfg
            if not self.cfg.env_cfg or not self.cfg.env_cfg.robot_cfg or not self.cfg.env_cfg.robot_cfg.robot_type:
                raise ValueError("env_cfg.robot_cfg.robot_type is required in config")
            
            stage_output_dir = os.path.join(
                self.output_dir, "traj",
                self.cfg.env_cfg.robot_cfg.robot_type,
                scene_name, object_id,
                f"grasp_{grasp_idx:04d}",
                f"stage_{stage_idx}",
            )
            
            # Check if required files exist
            required_files = [
                os.path.join(stage_output_dir, "start_iks.pt"),
                os.path.join(stage_output_dir, "goal_iks.pt"),
                os.path.join(stage_output_dir, "traj_data.pt"),
            ]
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    return False
        
        return True
