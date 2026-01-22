"""
Environment wrapper - main API for simulation interactions.

This module provides the SimEnvWrapper class which serves as the primary
interface for recording and evaluation pipelines to interact with the
Isaac Sim environment.

IMPORTANT: SimulationApp must be initialized before using this module.
Use automoma.simulation.sim_app_manager.get_simulation_app() first.

Usage:
    # First, initialize SimulationApp
    from automoma.simulation import get_simulation_app
    sim_app = get_simulation_app(headless=False)
    
    # Then create and use SimEnvWrapper
    cfg = load_config("multi_object_open")
    env = SimEnvWrapper(cfg)
    env.setup_env()
    
    # Set robot state and step
    env.set_state(robot_state, env_state)
    env.step()
    
    # Get observations
    data = env.get_data()
"""

from typing import Any, Dict, List, Literal, Optional, Union, Tuple
import logging

# Safe imports (don't require Isaac Sim)
from automoma.core.config_loader import Config
from automoma.utils.math_utils import pose_multiply
from automoma.utils.type_utils import to_list, to_numpy, to_float, ensure_non_negative

from curobo.geom.types import Pose
from curobo.types.state import JointState

import torch
import numpy as np


logger = logging.getLogger(__name__)


class SimEnvWrapper:
    """
    Simulation environment wrapper.
    
    This is the main API for interacting with the Isaac Sim environment.
    Used by:
    - Recording pipeline: to replay trajectories and collect observations
    - Evaluation pipeline: to run policy inference and evaluate performance
    
    Usage:
        # First, initialize SimulationApp
        from automoma.simulation import get_simulation_app
        sim_app = get_simulation_app(headless=False)
        
        # Then create and use SimEnvWrapper
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
            
        Raises:
            RuntimeError: If SimulationApp is not initialized
        """
        # Check that SimulationApp is initialized
        from automoma.utils.sim_utils import require_simulation_app
        require_simulation_app()
        
        self.cfg = cfg
        
        # Convert dict to Config if needed
        if isinstance(cfg, dict):
            from automoma.core.config_loader import Config as ConfigClass
            self.cfg = ConfigClass(cfg)
        
        # Import simulation components (safe now that SimulationApp is initialized)
        from automoma.simulation.simulator import IsaacSimManager
        from automoma.simulation.scene_builder import InfinigenBuilder
        from automoma.simulation.sensors import SensorRig
        from automoma.planning.planner import BasePlanner
        
        # Initialize components
        self.sim = IsaacSimManager(self.cfg)
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
        
    def setup_env(self, object_cfg=None, scene_cfg=None):
        """
        Setup the simulation environment.
        
        Loads scene, object, robot, and sensors based on configuration.
        
        Args:
            object_cfg: Optional object configuration to override default
            scene_cfg: Optional scene configuration to override default
        """
        logger.info("Setting up simulation environment...")
        
        # Get configs (support both dict and Config objects)
        if object_cfg is None:
            # Use object_cfg from env_cfg
            if not self.cfg or not self.cfg.object_cfg:
                raise ValueError("env_cfg.object_cfg is required in config")
            object_cfg = self._to_dict(self.cfg.object_cfg)
            # Handle multi-object config: pick first one if 'pose' not present
            if 'pose' not in object_cfg and object_cfg:
                first_key = list(object_cfg.keys())[0]
                print(f"Multi-object config detected, using first object: {first_key}")
                object_cfg = object_cfg[first_key]
        else:
            object_cfg = self._to_dict(object_cfg)
                
        if scene_cfg is None:
            # Use scene_cfg from env_cfg
            if not self.cfg or not self.cfg.scene_cfg:
                raise ValueError("env_cfg.scene_cfg is required in config")
            scene_cfg = self._to_dict(self.cfg.scene_cfg)
            # Handle multi-scene config
            if 'pose' not in scene_cfg and scene_cfg:
                first_key = list(scene_cfg.keys())[0]
                scene_cfg = scene_cfg[first_key]
        else:
            scene_cfg = self._to_dict(scene_cfg)
            
        # TODO: fix robot_cfg robot_config can be either a dict with 'path' key or already loaded robot config
        from automoma.utils.file_utils import load_robot_cfg, process_robot_cfg
        if not self.cfg or not self.cfg.robot_cfg:
            raise ValueError("env_cfg.robot_cfg is required in config")
        self.cfg.robot_cfg.robot = process_robot_cfg(load_robot_cfg(self.cfg.robot_cfg["path"]))
        robot_cfg = self._to_dict(self.cfg.robot_cfg)
        # print(f"Robot config used for setup: {robot_cfg}")
        # Use camera_cfg from env_cfg
        sensors_cfg = None
        if self.cfg:
            if hasattr(self.cfg, 'sensors_cfg') and self.cfg.sensors_cfg:
                sensors_cfg = self._to_dict(self.cfg.sensors_cfg)
            elif hasattr(self.cfg, 'camera_cfg') and self.cfg.camera_cfg:
                sensors_cfg = self._to_dict(self.cfg.camera_cfg)
        
        # Setup scene, object, and robot
        obj_pose = object_cfg.get('pose')
        scn_pose = scene_cfg.get('pose')
        print(f"Object pose: {obj_pose}, Scene pose: {scn_pose}")
        
        if obj_pose is not None and scn_pose is not None:
            self.scene.init_root_pose(obj_pose, scn_pose, type="object_center")
        else:
            logger.warning(f"Missing pose in config: object_pose={obj_pose}, scene_pose={scn_pose}")
            
        self.scene.load_scene(scene_cfg, prim_path="/World/Scene")
        self.scene.load_object(object_cfg, prim_path="/World/Object")
        self.scene.load_robot(robot_cfg["robot"], prim_path="/World/Robot")
        
        # Set deactivate 
        deactivate_prim_paths = [f"StaticCategoryFactory_{object_cfg['asset_type']}_{object_cfg['asset_id']}", "exterior", "ceiling", "Ceiling"]
        self.sim.set_deactivate_prims(deactivate_prim_paths)
        # Set collision free
        collision_free_prim_paths = []
        if self.cfg and self.cfg.sim_cfg and hasattr(self.cfg.sim_cfg, 'collision_free_prim_paths'):
            collision_free_prim_paths = self.cfg.sim_cfg.collision_free_prim_paths
        # TODO: enable when eval
        self.sim.set_isaacsim_collision_free(prim_paths=collision_free_prim_paths)
        
        # Set lighting
        lighting_mode = 2  # default
        if self.cfg and self.cfg.sim_cfg and hasattr(self.cfg.sim_cfg, 'lighting_mode'):
            lighting_mode = self.cfg.sim_cfg.lighting_mode
        self.sim.set_lighting(lighting_mode)
        
        # Setup physics
        self.sim.init_world_physics()
        
        
        self.robot = self.scene.robot
        self.robot_joint_names = robot_cfg["robot"].get("kinematics", {}).get("cspace", {}).get("joint_names", [])
        self.robot_idx_list = [self.robot.get_dof_index(name) for name in self.robot_joint_names]
        
        # Setup sensors
        self.sensors.setup_sensors(sensors_cfg)
        
        # Initialize planner for FK utilities
        self.planner.init_motion_gen(robot_cfg["robot"])
        
        self.is_setup = True
        logger.info("Environment setup complete")
        
        # Warmup sensors (Isaac Sim cameras need a few steps to start producing data)
        logger.info("Warming up sensors...")
        for _ in range(10):
            self.sim.step()
    
    def _to_dict(self, cfg) -> Dict[str, Any]:
        """Convert Config to dict if needed."""
        if cfg is None:
            return {}
        if hasattr(cfg, 'to_dict'):
            return cfg.to_dict()
        if isinstance(cfg, dict):
            return cfg
        return {}
        
    def reset(self, initial_state: Optional[Union[torch.Tensor, Tuple]] = None, env_state=None) -> Dict[str, Any]:
        """
        Reset the environment.
        
        Args:
            initial_state: Optional initial robot state (or tuple of robot_state, env_state)
            env_state: Optional environment state (if not provided in initial_state tuple)
            
        Returns:
            Initial observation (action will be None as there's no next state)
        """
        self.history_robot_state = []
        self.history_env_state = []
        self.current_step = 0
        
        if initial_state is not None:
            if isinstance(initial_state, tuple):
                self.set_state(*initial_state)
            else:
                self.set_state(initial_state, env_state)
        
        self.step()
        return self.get_data(next_robot_state=None)
    
    def step(self, action: Optional[torch.Tensor] = None):
        """
        Step the simulation.
        
        Args:
            action: Optional action to apply before stepping.
                   Assumed to be [delta_base, absolute_arm] format if base_dof > 0.
            
        Returns:
            Observation after step
        """
        
        robot_state = None
        
        # Apply action if provided
        if action is not None:
            # Handle action application (delta base + absolute arm)
            current_state = self.get_robot_state()
            action_np = to_numpy(action)
            
            base_dof = self.cfg.robot_cfg.get("mobile_base_dof", 3)
            
            if len(action_np) == len(current_state) and base_dof > 0:
                # Construct new state
                new_state = action_np.copy()
                # Integrate base: new_base = current_base + delta_base
                new_state[:base_dof] = current_state[:base_dof] + action_np[:base_dof]
                # Arm is already absolute in action
            else:
                # Assume full absolute positions
                new_state = action_np
            robot_state = new_state
        
        # Step simulation
        for _ in range(5):
            self.sim.step(step=1,render=True)
            self.set_state(robot_state=robot_state)
            self.sensors.update()
        
        self.current_step += 1
        
    def set_state(self, robot_state: torch.Tensor = None, env_state = None):
        """
        Set robot and environment state.
        
        Args:
            robot_state: Robot joint positions (tensor, array, or list)
            env_state: Environment state (e.g., object joint angle)
        """
        # Set robot state using type_utils for clean conversion
        if robot_state is not None:
            robot_state_list = to_list(robot_state)
            self.robot.set_joint_positions(robot_state_list, self.robot_idx_list)
            self.robot._articulation_view.set_max_efforts(
                values=np.array([5000] * len(self.robot_idx_list)), 
                joint_indices=self.robot_idx_list,
            )
            self.history_robot_state.append(robot_state)

        # Set environment state (e.g., object joint angle)
        if env_state is not None:
            # Convert to non-negative float for articulated object joints
            # Note: For door/drawer opening, angles are typically non-negative
            env_state_float = ensure_non_negative(to_float(env_state))
            
            obj = self.sim.world.scene.get_object("target_object")
            if not obj._articulation_view.initialized:
                obj.get_articulation_controller()
            
            # Use object_cfg from env_cfg
            if not self.cfg or not self.cfg.object_cfg:
                raise ValueError("env_cfg.object_cfg is required in config")
            object_cfg = self._to_dict(self.cfg.object_cfg)
            # Handle multi-object config
            if 'joint_id' not in object_cfg and object_cfg:
                first_key = list(object_cfg.keys())[0]
                object_cfg = object_cfg[first_key]
            joint_id = object_cfg.get("joint_id", 0)
            obj.set_joint_positions(env_state_float, [joint_id])
            self.history_env_state.append(env_state_float)
    
    def apply_object_action(self, joint_position: float, joint_velocity: float = 0.0) -> None:
        """
        Apply action to the target object (fallback strategy).
        
        Args:
            joint_position: Target joint position
            joint_velocity: Target joint velocity
        """
        from omni.isaac.core.utils.types import ArticulationAction
        import numpy as np
        
        obj = self.sim.world.scene.get_object("target_object")
        if obj is None:
            return
            
        if not obj._articulation_view.initialized:
            obj.get_articulation_controller()

        # Use object_cfg from env_cfg
        if not self.cfg or not self.cfg.object_cfg:
            return
        object_cfg = self._to_dict(self.cfg.object_cfg)
        if 'joint_id' not in object_cfg and object_cfg:
            first_key = list(object_cfg.keys())[0]
            object_cfg = object_cfg[first_key]
        
        # Get joint id
        joint_id = object_cfg.get("joint_id", 0)
        # Some implementation uses joint_id-1 but set_state uses joint_id. 
        # We follow set_state pattern here for consistency within this class.
        
        action = ArticulationAction(
            joint_positions=np.array([joint_position]), 
            joint_velocities=np.array([joint_velocity]), 
            joint_indices=np.array([joint_id])
        )
        obj.get_articulation_controller().apply_action(action)

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
        """Get end effector pose using forward kinematics."""
        # Get current joint positions
        joint_data = self.robot.get_joint_positions()
        
        js = JointState.from_position(self.planner.tensor_args.to_device(joint_data))
        fk_result = self.planner.motion_gen.ik_solver.fk(js.position)
        eef_pose_7d = np.array(fk_result.ee_pose.to_list())
        return eef_pose_7d

    def _compute_action(self, current_state, next_state):
        """
        Compute action as next_state - current_state.
        
        For mobile base (first N DOFs), use delta.
        For arm joints, use absolute next position.
        
        Args:
            current_state: Current robot state
            next_state: Next robot state
            
        Returns:
            Action array
        """
        # Convert to numpy using type_utils
        current_np = to_numpy(current_state)
        next_np = to_numpy(next_state)
        
        # Copy next state as base action
        action = next_np.copy()
        
        # For mobile base DOFs, compute delta
        base_dof = self.cfg.robot_cfg.get("mobile_base_dof", 3)
        if len(current_np) >= base_dof and len(next_np) >= base_dof:
            action[:base_dof] = next_np[:base_dof] - current_np[:base_dof]
        
        return action

    def get_data(self, next_robot_state=None):
        """
        Get observation data.
        
        Args:
            next_robot_state: Optional next robot state for computing action.
                            If provided, action = next_state - current_state.
                            If None, action is None (last frame has no valid action).
        
        Returns:
            Dictionary with observation data
        """
        # Collect observations
        obs_data = self.sensors.get_obs()
        
        # Get current robot state for joint observations
        joint_data = self.robot.get_joint_positions()
        
        # Get end effector pose
        eef_pose_data = self._get_end_effector_pose()
        
        # Compute action: state[t+1] - state[t]
        action_data = None
        if next_robot_state is not None:
            action_data = self._compute_action(joint_data, next_robot_state)
        
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
