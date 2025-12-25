"""
Environment wrapper - main API for simulation interactions.

This module provides the SimEnvWrapper class which serves as the primary
interface for recording and evaluation pipelines to interact with the
Isaac Sim environment.
"""

from automoma.simulation.simulator import IsaacSimManager
from automoma.simulation.scene_builder import SceneBuilder, InfinigenBuilder
from automoma.simulation.sensors import SensorRig
from automoma.planning.planner import BasePlanner
from automoma.core.config_loader import Config

from automoma.utils.math_utils import pose_multiply

from curobo.geom.types import Pose
from curobo.types.state import JointState

import torch
import numpy as np
from typing import Any, Dict, List, Literal, Optional, Union
import logging


logger = logging.getLogger(__name__)


class SimEnvWrapper:
    """
    Simulation environment wrapper.
    
    This is the main API for interacting with the Isaac Sim environment.
    Used by:
    - Recording pipeline: to replay trajectories and collect observations
    - Evaluation pipeline: to run policy inference and evaluate performance
    
    Usage:
        cfg = load_config("multi_object_open")
        env = SimEnvWrapper(cfg)
        env.setup_env()
        
        # Set robot state
        env.set_state(robot_state, env_state)
        env.step()
        
        # Get observations
        data = env.get_data()
    """
    
    def __init__(self, cfg: Union[Config, Dict[str, Any]]):
        """
        Initialize environment wrapper.
        
        Args:
            cfg: Configuration object or dictionary
        """
        self.cfg = cfg
        
        # Convert dict to Config if needed
        if isinstance(cfg, dict):
            from automoma.core.config_loader import Config as ConfigClass
            self.cfg = ConfigClass(cfg)
        
        # Initialize components
        self.sim = IsaacSimManager(self.cfg.sim)
        self.scene = InfinigenBuilder(self.sim)
        self.sensors = SensorRig(self.sim)
        
        # Basic planner for FK/IK utilities
        self.planner = BasePlanner()
        
        # State
        self.robot = None
        self.robot_joint_names = []
        self.robot_idx_list = []
        self.is_setup = False
        
        # History buffers
        self.history_robot_state = []
        self.history_env_state = []
        self.current_step = 0
        
    def setup_env(self):
        """
        Setup the simulation environment.
        
        Loads scene, object, robot, and sensors based on configuration.
        """
        logger.info("Setting up simulation environment...")
        
        # Get configs (support both dict and Config objects)
        object_cfg = self._to_dict(self.cfg.object_cfg)
        scene_cfg = self._to_dict(self.cfg.scene_cfg)
        robot_cfg = self._to_dict(self.cfg.robot_cfg)
        sensors_cfg = self._to_dict(self.cfg.sensors_cfg) if self.cfg.sensors_cfg else {}
        
        # Setup scene, object, and robot
        self.scene.init_root_pose(object_cfg.get('pose'), scene_cfg.get('pose'), type="object_center")
        self.scene.load_scene(scene_cfg, prim_path="/World/Scene")
        self.scene.load_object(object_cfg, prim_path="/World/Object")
        self.scene.load_robot(robot_cfg, prim_path="/World/Robot")
        
        self.robot = self.scene.robot
        self.robot_joint_names = robot_cfg.get("kinematics", {}).get("cspace", {}).get("joint_names", [])
        self.robot_idx_list = [self.robot.get_dof_index(name) for name in self.robot_joint_names]
        
        # Setup sensors
        if sensors_cfg:
            self.sensors.setup_sensors(sensors_cfg)
        
        # Initialize planner for FK utilities
        self.planner.init_motion_gen(robot_cfg)
        
        self.is_setup = True
        logger.info("Environment setup complete")
    
    def _to_dict(self, cfg) -> Dict[str, Any]:
        """Convert Config to dict if needed."""
        if cfg is None:
            return {}
        if hasattr(cfg, 'to_dict'):
            return cfg.to_dict()
        if isinstance(cfg, dict):
            return cfg
        return {}
        
    def reset(self, initial_state: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Reset the environment.
        
        Args:
            initial_state: Optional initial robot state
            
        Returns:
            Initial observation
        """
        self.history_robot_state = []
        self.history_env_state = []
        self.current_step = 0
        
        if initial_state is not None:
            self.set_state(initial_state)
        
        self.step()
        return self.get_data()
    
    def step(self, action: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Step the simulation.
        
        Args:
            action: Optional action to apply before stepping
            
        Returns:
            Observation after step
        """
        if action is not None:
            self.set_state(action)
        
        # Step simulation
        self.sim.world.step(render=True)
        self.current_step += 1
        
        return self.get_data()
    
    def set_state(self, robot_state: torch.Tensor = None, env_state = None):
        """
        Set robot and environment state.
        
        Args:
            robot_state: Robot joint positions
            env_state: Environment state (e.g., object joint angle)
        """
        # Set robot state
        if robot_state is not None:
            if isinstance(robot_state, torch.Tensor):
                robot_state_list = robot_state.tolist()
            else:
                robot_state_list = list(robot_state)
            
            self.robot.set_joint_positions(robot_state_list, self.robot_idx_list)
            self.robot._articulation_view.set_max_efforts(
                values=np.array([5000] * len(self.robot_idx_list)), 
                joint_indices=self.robot_idx_list,
            )
            self.history_robot_state.append(robot_state)

        # Set environment state
        if env_state is not None:
            if isinstance(env_state, torch.Tensor):
                env_state = env_state.item()
            env_state = abs(float(env_state))  # Avoid negative values
            
            obj = self.sim.world.scene.get_object("target_object")
            if not obj._articulation_view.initialized:
                obj.get_articulation_controller()
            
            object_cfg = self._to_dict(self.cfg.object_cfg)
            joint_id = object_cfg.get("joint_id", 0)
            obj.set_joint_positions(env_state, [joint_id])
            self.history_env_state.append(env_state)
    
    def set_gripper(self, value: float) -> None:
        """
        Set gripper state.
        
        Args:
            value: Gripper value (0=closed, 0.04=open typically)
        """
        if self.robot is not None:
            # Assuming last 2 DOFs are gripper
            gripper_indices = self.robot_idx_list[-2:]
            self.robot.set_joint_positions([value, value], gripper_indices)
    
    def get_gripper_state(self) -> float:
        """Get current gripper state."""
        if self.robot is not None:
            positions = self.robot.get_joint_positions()
            gripper_pos = positions[-2:]  # Last 2 are gripper
            return sum(gripper_pos) / 2
        return 0.0
    
    def get_env_state(self) -> float:
        """Get current environment state (e.g., object joint angle)."""
        try:
            obj = self.sim.world.scene.get_object("target_object")
            if obj is not None:
                positions = obj.get_joint_positions()
                if positions:
                    return float(positions[0])
        except Exception:
            pass
        return 0.0
    
    def _get_end_effector_pose(self) -> np.ndarray:
        # Get end effector pose
        if not self.history_robot_state:
            joint_data = self.robot.get_joint_positions()
        else:
            joint_data = self.history_robot_state[-1]
        
        js = JointState.from_position(self.planner.tensor_args.to_device(joint_data))
        fk_result = self.planner.motion_gen.ik_solver.fk(js.position)
        eef_pose_7d = np.array(fk_result.ee_pose.to_list())
        return eef_pose_7d

    def _get_action_data(self, type: Literal["absolute", "relative"] = "relative"):
        """
        Encode action from the state.
        """
        if not self.history_robot_state:
            return None
            
        joint_data = self.history_robot_state[-1]
        if isinstance(joint_data, torch.Tensor):
            joint_data = joint_data.detach().cpu().numpy()

        if type == "absolute":
            return joint_data
        
        if len(self.history_robot_state) < 2:
            # If only one state, relative action is zero for base
            action = joint_data.copy()
            base_dof = self.cfg.robot_cfg.get("mobile_base_dof", 3)
            if len(action) >= base_dof:
                action[:base_dof] = 0.0
            return action
        
        prev_joint_data = self.history_robot_state[-2]
        if isinstance(prev_joint_data, torch.Tensor):
            prev_joint_data = prev_joint_data.detach().cpu().numpy()
            
        action = joint_data.copy()        
        # Assuming first 3 are mobile base if they exist
        base_dof = self.cfg.robot_cfg.get("mobile_base_dof", 3)
        if len(joint_data) >= base_dof:
            action[:base_dof] = joint_data[:base_dof] - prev_joint_data[:base_dof]
        
        return action

    def get_data(self):
        # Collect observations
        obs_data = self.sensors.get_obs()
        
        # Get current robot state for joint observations
        joint_data = self.robot.get_joint_positions()
        self.history_robot_state.append(joint_data)
        
        # Get end effector pose
        eef_pose_data = self._get_end_effector_pose()
        
        # Get action data
        action_data = self._get_action_data(type="relative")
        
        # Get environment state
        env_state = self.get_env_state()
        
        return {
            "obs_data": obs_data,
            "joint_data": joint_data,
            "eef_pose_data": eef_pose_data,
            "action_data": action_data,
            "env_state": env_state,
            "gripper_state": self.get_gripper_state(),
            "step": self.current_step,
        }
    
    def get_robot_state(self) -> np.ndarray:
        """Get current robot joint positions."""
        if self.robot is not None:
            return np.array(self.robot.get_joint_positions())
        return np.array([])
    
    def close(self) -> None:
        """Close the environment and cleanup resources."""
        if self.sim is not None:
            self.sim.close()
        logger.info("Environment closed")
