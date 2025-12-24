from automoma.simulation.simulator import IsaacSimManager
from automoma.simulation.scene_builder import SceneBuilder, InfinigenBuilder
from automoma.simulation.sensors import SensorRig
from automoma.planning.planner import BasePlanner

from automoma.utils.math_utils import pose_multiply

from curobo.geom.types import Pose
from curobo.types.state import JointState

import torch
import numpy as np
from typing import Any, Dict, List, Literal


class SimEnvWrapper:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sim = IsaacSimManager(cfg.sim)      # 片场
        self.scene = InfinigenBuilder(self.sim)  # 道具
        self.sensors = SensorRig(self.sim)       # 摄像
        
        # Basic planner for utils
        self.planner = BasePlanner()
        self.planner.init_motion_gen(self.cfg.robot_cfg)  # 运动生成器
        
        # History buffers
        self.history_robot_state = []
        self.history_env_state = []
        
    def setup_env(self):
        
        # Setup scene, object, and robot
        self.scene.init_root_pose(self.cfg.object_cfg['pose'], self.cfg.scene_cfg['pose'], type="object_center")
        self.scene.load_scene(self.cfg.scene_cfg, prim_path="/World/Scene")
        self.scene.load_object(self.cfg.object_cfg, prim_path="/World/Object")
        self.scene.load_robot(self.cfg.robot_cfg, prim_path="/World/Robot")
        
        self.robot = self.scene.robot
        self.robot_joint_names = self.cfg.robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.robot_idx_list = [self.robot.get_dof_index(name) for name in self.robot_joint_names]
        self.sensors.setup_sensors(self.cfg.sensors_cfg)
        
    def reset(self):
        raise NotImplementedError("The reset method must be implemented by subclasses.")
    
    def step(self):
        raise NotImplementedError("The step method must be implemented by subclasses.")
    
    def set_state(self, robot_state: torch.Tensor = None, env_state = None):
        # set robot state
        if robot_state is not None:
            self.robot.set_joint_positions(robot_state.tolist(), self.robot_idx_list)
            self.robot._articulation_view.set_max_efforts(
                    values=np.array([5000] * len(self.robot_idx_list)), joint_indices=self.robot_idx_list,
                )

        # set env state
        if env_state is not None:
            if type(env_state) is torch.Tensor:
                env_state = env_state.item()
            env_state = abs(float(env_state)) # set abs to avoid negative values
            obj = self.sim.world.scene.get_object("target_object")
            if not obj._articulation_view.initialized:
                obj.get_articulation_controller()
            obj.set_joint_positions(env_state, [self.cfg.object_cfg["joint_id"]])    
    def _get_end_effector_pose(self):
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
        
        return {
            "obs_data": obs_data,
            "joint_data": joint_data,
            "eef_pose_data": eef_pose_data,
            "action_data": action_data
        }
        