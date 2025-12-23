from automoma.simulation.simulator import IsaacSimManager
from automoma.simulation.scene_builder import SceneBuilder, InfinigenBuilder
from automoma.simulation.sensors import SensorRig
from automoma.planning.planner import BasePlanner

from automoma.utils.math_utils import pose_multiply

import torch
import numpy as np
from typing import Any, Dict, List


class SimEnvWrapper:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sim = IsaacSimManager(cfg.sim)      # 片场
        self.scene = InfinigenBuilder(self.sim)  # 道具
        self.sensors = SensorRig(self.sim)       # 摄像
        
        # Basic planner for utils
        self.planner = BasePlanner()
        self.planner.init_motion_gen(self.cfg.robot_cfg)  # 运动生成器
        
    def setup_env(self):
        
        # Setup scene, object, and robot
        self.scene.init_root_pose(self.cfg.object_cfg['pose'], type="object_center")
        self.cfg.object_cfg['pose'] = pose_multiply(self.cfg.scene_cfg['pose'], self.cfg.object_cfg['pose']) 
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
    
    def get_obs_data(self):
        raise NotImplementedError("The get_obs method must be implemented by subclasses.")
    
    
    def _get_end_effector_pose(self, joint_data):
        # Get end effector pose
        eef_pose_data = self.planner.get_world_pose(
            self.robot.get_link_pose(self.cfg.robot_cfg["end_effector_link_name"])
        )
        return eef_pose_data
    def get_data(self):
        # Collect observations
        obs_data = self.get_obs_data()
        
        # Get current robot state for joint observations
        joint_data = self.robot.get_joint_positions()
        
        # Get end effector pose
        eef_pose_data = self._get_end_effector_pose(joint_data)