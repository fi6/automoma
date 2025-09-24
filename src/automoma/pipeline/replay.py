"""
ReplayPipeline for visualizing motion planning results in Isaac Sim.
This pipeline loads saved IK and trajectory results for visualization and data collection.
"""
import os
import torch
from typing import Optional, List
from pathlib import Path

# Import automoma modules first to avoid USD conflicts
from automoma.models.task import TaskDescription
from automoma.utils.data_structures import TrajectoryEvaluationResult
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
        scene_name = scene_path.parent.parent.parent.name
        
        output_dir = os.path.join(
            self.output_base_dir,
            robot_name,
            scene_name,
            self.task.object.asset_id,
            f"grasp_{grasp_id:04d}"
        )
        
        os.makedirs(output_dir, exist_ok=True)
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
            print(f"Filtered trajectory data not found at {filtered_path}")
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
        self.replayer.replay_traj_akr(
            start_states=start_states,
            goal_states=goal_states,
            trajs=trajectories,
            successes=success,
            robot_name=robot_name
        )
    
    # ========================================
    # New Data Collection Methods
    # ========================================
    
    def replay_traj_record(self, grasp_id: int = 0, num_episodes: int = 10) -> None:
        """
        Record trajectory data with camera observations for data collection.
        
        Args:
            grasp_id: Grasp ID to record trajectories for
            num_episodes: Maximum number of episodes to record
        """
        print(f"=== Recording trajectory data for grasp {grasp_id} ===")
        
        # Load trajectory results from file (try filtered first, then regular)
        output_dir = self._get_output_directory(grasp_id)
        
        filtered_path = os.path.join(output_dir, "filtered_traj_data.pt")
        traj_path = os.path.join(output_dir, "traj_data.pt")
        
        if os.path.exists(filtered_path):
            data_path = filtered_path
            print("Using filtered trajectory data")
        elif os.path.exists(traj_path):
            data_path = traj_path
            print("Using regular trajectory data")
        else:
            print(f"No trajectory data found in {output_dir}")
            return []
            
        traj_data = torch.load(data_path, weights_only=False)
        start_states = traj_data["start_state"]
        goal_states = traj_data["goal_state"]
        trajectories = traj_data["traj"]
        success = traj_data["success"]
        
        # Initialize Isaac Sim if not already done
        if self.replayer is None:
            self._init_isaac_sim()
        
        robot_name = self._get_robot_name()
        successful_count = success.sum().item()
        print(f"Recording from {successful_count}/{len(success)} successful trajectories")
        
        # Extract scene information
        scene_path = Path(self.task.scene.scene_usd_path)
        scene_name = scene_path.parent.parent.parent.name
        
        # Record trajectory data
        self.replayer.replay_traj_record(
            start_states=start_states,
            goal_states=goal_states,
            trajs=trajectories,
            successes=success,
            robot_name=robot_name,
            output_dir=output_dir,
            scene_id=scene_name,
            object_id=self.task.object.asset_id,
            angle_id="0",  # Could be made configurable
            pose_id=str(grasp_id),   # Could be made configurable
            num_episodes=num_episodes,
        )
    
    def replay_traj_evaluate(self, policy_model, grasp_id: int = 0, 
                           num_episodes: int = 5) -> List[TrajectoryEvaluationResult]:
        """
        Evaluate a policy model on trajectory data.
        
        Args:
            policy_model: Policy model for inference
            grasp_id: Grasp ID to evaluate trajectories for
            num_episodes: Maximum number of episodes to evaluate
            
        Returns:
            List of TrajectoryEvaluationResult objects containing evaluation results
        """
        print(f"=== Evaluating policy model for grasp {grasp_id} ===")
        
        # Load trajectory results from file
        output_dir = self._get_output_directory(grasp_id)
        
        filtered_path = os.path.join(output_dir, "filtered_traj_data.pt")
        traj_path = os.path.join(output_dir, "traj_data.pt")
        
        if os.path.exists(filtered_path):
            data_path = filtered_path
            print("Using filtered trajectory data for evaluation")
        elif os.path.exists(traj_path):
            data_path = traj_path
            print("Using regular trajectory data for evaluation")
        else:
            print(f"No trajectory data found in {output_dir}")
            return []
            
        traj_data = torch.load(data_path, weights_only=False)
        start_states = traj_data["start_state"]
        goal_states = traj_data["goal_state"]
        trajectories = traj_data["traj"]
        success = traj_data["success"]
        
        # Initialize Isaac Sim if not already done
        if self.replayer is None:
            self._init_isaac_sim()
        
        robot_name = self._get_robot_name()
        successful_count = success.sum().item()
        print(f"Evaluating on {successful_count}/{len(success)} successful trajectories")
        
        # Extract scene information
        scene_path = Path(self.task.scene.scene_usd_path)
        scene_name = scene_path.parent.parent.parent.name
        
        # Evaluate policy
        evaluation_results = self.replayer.replay_traj_evaluate(
            policy_model=policy_model,
            start_states=start_states,
            goal_states=goal_states,
            trajs=trajectories,
            successes=success,
            robot_name=robot_name,
            scene_id=scene_name,
            object_id=self.task.object.asset_id,
            angle_id="0",  # Could be made configurable
            pose_id="0",   # Could be made configurable
            grasp_id=grasp_id
        )
        
        print(f"Completed evaluation on {len(evaluation_results)} trajectories")
        return evaluation_results
    
    def save_evaluation_results(self, evaluation_results: List[TrajectoryEvaluationResult], 
                              output_dir: str, grasp_id: int = 0) -> None:
        """
        Save evaluation results to file.
        
        Args:
            evaluation_results: List of evaluation results to save
            output_dir: Directory to save results
            grasp_id: Grasp ID for filename
        """
        if not evaluation_results:
            print("No evaluation results to save")
            return
        
        # Prepare data for saving
        eval_data = {
            "eef_poses": [],
            "open_angles": [],
            "success": [],
            "num_steps": [],
            "trajectory_indices": [],
            "metrics": []
        }
        
        for result in evaluation_results:
            eval_data["eef_poses"].append(result.eef_poses)
            eval_data["open_angles"].append(result.open_angles)
            eval_data["success"].append(result.success)
            eval_data["num_steps"].append(result.num_steps)
            eval_data["trajectory_indices"].append(result.trajectory_idx)
            eval_data["metrics"].append(result.compute_metrics())
        
        # Convert lists to tensors where appropriate
        if eval_data["eef_poses"]:
            eval_data["eef_poses"] = torch.cat(eval_data["eef_poses"], dim=0)
        if eval_data["open_angles"]:  
            eval_data["open_angles"] = torch.cat(eval_data["open_angles"], dim=0)
        
        eval_data["success"] = torch.tensor(eval_data["success"])
        eval_data["num_steps"] = torch.tensor(eval_data["num_steps"])
        eval_data["trajectory_indices"] = torch.tensor(eval_data["trajectory_indices"])
        
        # Save to file
        eval_dir = os.path.join(output_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        eval_path = os.path.join(eval_dir, f"eval_results_grasp_{grasp_id:04d}.pt")
        torch.save(eval_data, eval_path)
        
        print(f"Saved evaluation results to {eval_path}")
        
        # Also save metrics summary
        metrics_summary = {
            "avg_position_error": sum(m["position_error"] for m in eval_data["metrics"]) / len(eval_data["metrics"]),
            "avg_rotation_error": sum(m["rotation_error"] for m in eval_data["metrics"]) / len(eval_data["metrics"]),
            "success_rate": sum(m["success"] for m in eval_data["metrics"]) / len(eval_data["metrics"]),
            "avg_num_steps": sum(m["num_steps"] for m in eval_data["metrics"]) / len(eval_data["metrics"])
        }
        
        summary_path = os.path.join(eval_dir, f"metrics_summary_grasp_{grasp_id:04d}.txt")
        with open(summary_path, "w") as f:
            f.write("Evaluation Metrics Summary\n")
            f.write("=" * 30 + "\n")
            for key, value in metrics_summary.items():
                f.write(f"{key}: {value:.4f}\n")
        
        print(f"Saved metrics summary to {summary_path}")
    
    def close(self):
        """Close Isaac Sim application."""
        if self.simulation_app is not None:
            self.simulation_app.close()
            self.simulation_app = None
            self.replayer = None