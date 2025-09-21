"""
ReplayPipeline for visualizing motion planning results in Isaac Sim.
This pipeline loads saved IK and trajectory results for visualization.
"""

import torch
import os
from typing import Optional

# Import automoma modules first to avoid USD conflicts
from automoma.models.task import TaskDescription
from pathlib import Path
from typing import Optional



    
class ReplayPipeline:
    def __init__(self, task: TaskDescription, simulation_app=None, output_base_dir: str = "output"):
        self.task = task
        self.output_base_dir = output_base_dir
        self.simulation_app = simulation_app  # Lazy initialization
        self.replayer = None
        self._init_isaac_sim()
        
    def _init_isaac_sim(self):
        """Initialize Isaac Sim on first use (lazy initialization)."""
        # if self.simulation_app is not None:
        #     return  # Already initialized
            
        # print("Initializing Isaac Sim...")
        
        # # Import Isaac Sim modules only when needed
        # import isaacsim
        # from omni.isaac.kit import SimulationApp
        
        # self.simulation_app = SimulationApp({
        #     "headless": False,
        #     "width": 1920,
        #     "height": 1080
        # })
        
        # print("Isaac Sim initialized.")
        
        # Prepare configuration for replayer
        scene_cfg = {
            "path": self.task.scene.scene_usd_path,
            "pose": self.task.scene.pose,
        }
        object_cfg = {
            "path": self.task.object.urdf_path,
            "asset_type": self.task.object.asset_type,
            "asset_id": self.task.object.asset_id,
            "pose": self.task.object.pose,
            "joint_id": 0,
        }
        
        robot_cfg = self.task.robot.robot_cfg
        
        from automoma.utils.replayer import Replayer
        
        # Initialize replayer with proper configuration
        self.replayer = Replayer(
            simulation_app=self.simulation_app,
            robot_cfg=robot_cfg,
            scene_cfg=scene_cfg,
            object_cfg=object_cfg
        )
        
    def _get_output_directory(self, grasp_id: int) -> str:
        """Generate organized output directory path."""
        robot_name = "summit_franka"
        scene_path = Path(self.task.scene.scene_usd_path)
        scene_name = scene_path.parent.parent.name
        
        output_dir = os.path.join(
            self.output_base_dir,
            robot_name,
            scene_name,
            self.task.object.asset_id,
            f"grasp_{grasp_id:04d}"
        )
        
        return output_dir
        
    def _get_object_pose(self):
        """Get object pose from scene metadata."""
        if hasattr(self.task.scene, 'get_object_pose'):
            object_pose_np = self.task.scene.get_object_pose(self.task.object)
            return object_pose_np.tolist()
        else:
            # Default pose if not available
            return [0, 0, 0, 1, 0, 0, 0]
    
    def _get_robot_name(self):
        """Extract robot name from configuration."""
        # Try to get robot name from config, default to summit_franka
        return "summit_franka"  # This could be made configurable
    
    def replay_ik(self, grasp_id: int = 0):
        """Replay IK solutions for visualization."""
        print(f"=== Replaying IK for grasp {grasp_id} ===")
        
        # Load IK results from file
        output_dir = self._get_output_directory(grasp_id)
        ik_path = os.path.join(output_dir, "ik_data.pt")
        
        if not os.path.exists(ik_path):
            print(f"IK data not found at {ik_path}. Please run IK planning first.")
            return
            
        ik_data = torch.load(ik_path, weights_only=False)
        start_iks = ik_data["start_iks"]
        goal_iks = ik_data["goal_iks"]
        
        # Initialize Isaac Sim if not already done
        if self.replayer is None:
            self._init_isaac_sim()
        
        robot_name = self._get_robot_name()
        print(f"Replaying {start_iks.shape[0]} start IKs and {goal_iks.shape[0]} goal IKs")
        
        # Replay IK solutions
        self.replayer.replay_ik(start_iks, goal_iks, robot_name)
        
    def replay_traj(self, grasp_id: int = 0):
        """Replay trajectory solutions for visualization."""
        print(f"=== Replaying trajectories for grasp {grasp_id} ===")
        
        # Load trajectory results from file
        output_dir = self._get_output_directory(grasp_id)
        traj_path = os.path.join(output_dir, "traj_data.pt")
        
        if not os.path.exists(traj_path):
            print(f"Trajectory data not found at {traj_path}. Please run trajectory planning first.")
            return
            
        traj_data = torch.load(traj_path, weights_only=False)
        start_states = traj_data["start_state"]
        goal_states = traj_data["goal_state"] 
        trajectories = traj_data["traj"]
        success = traj_data["success"]
        
        # Initialize Isaac Sim if not already done
        if self.replayer is None:
            self._init_isaac_sim()
        
        robot_name = self._get_robot_name()
        successful_count = success.sum().item()
        print(f"Replaying {successful_count}/{len(success)} successful trajectories")
        
        # Replay trajectories
        self.replayer.replay_traj(
            start_states=start_states,
            goal_states=goal_states, 
            trajs=trajectories,
            successes=success,
            robot_name=robot_name
        )
        
    def replay_traj_akr(self, grasp_id: int = 0):
        """Replay AKR trajectory solutions for visualization."""
        print(f"=== Replaying AKR trajectories for grasp {grasp_id} ===")
        
        # Load trajectory results (try filtered first, then regular)
        output_dir = self._get_output_directory(grasp_id)
        
        # Try filtered first
        filtered_path = os.path.join(output_dir, "filtered_traj_data.pt")
        traj_path = os.path.join(output_dir, "traj_data.pt")
        
        if os.path.exists(filtered_path):
            traj_data = torch.load(filtered_path, weights_only=False)
            print("Using filtered trajectory data")
        elif os.path.exists(traj_path):
            traj_data = torch.load(traj_path, weights_only=False)
            print("Using regular trajectory data")
        else:
            print(f"No trajectory data found. Please run trajectory planning first.")
            return
            
        start_states = traj_data["start_state"]
        goal_states = traj_data["goal_state"]
        trajectories = traj_data["traj"] 
        success = traj_data["success"]
        
        # Initialize Isaac Sim if not already done
        if self.replayer is None:
            self._init_isaac_sim()
        
        robot_name = self._get_robot_name()
        successful_count = success.sum().item()
        print(f"Replaying {successful_count}/{len(success)} successful AKR trajectories")
        
        # Replay AKR trajectories
        self.replayer.replay_traj_akr(
            start_states=start_states,
            goal_states=goal_states,
            trajs=trajectories,
            successes=success, 
            robot_name=robot_name
        )
        
    def replay_filtered_traj(self, grasp_id: int = 0):
        """Replay filtered trajectory solutions for visualization."""
        print(f"=== Replaying filtered trajectories for grasp {grasp_id} ===")
        
        # Load filtered trajectory results from file
        output_dir = self._get_output_directory(grasp_id)
        filtered_path = os.path.join(output_dir, "filtered_traj_data.pt")
        
        if not os.path.exists(filtered_path):
            print(f"Filtered trajectory data not found at {filtered_path}. Please run trajectory filtering first.")
            return
            
        traj_data = torch.load(filtered_path, weights_only=False)
        start_states = traj_data["start_state"]
        goal_states = traj_data["goal_state"]
        trajectories = traj_data["traj"]
        success = traj_data["success"]
        
        # Initialize Isaac Sim if not already done
        if self.replayer is None:
            self._init_isaac_sim()
        
        robot_name = self._get_robot_name()
        successful_count = success.sum().item()
        print(f"Replaying {successful_count}/{len(success)} filtered trajectories")
        
        # Replay filtered trajectories
        self.replayer.replay_traj(
            start_states=start_states,
            goal_states=goal_states,
            trajs=trajectories,
            successes=success,
            robot_name=robot_name
        )
    
    def close(self):
        """Close Isaac Sim application."""
        if self.simulation_app is not None:
            self.simulation_app.close()
            self.simulation_app = None
            self.replayer = None