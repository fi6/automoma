from automoma.models.task import TaskDescription, TaskType
from automoma.utils.file import process_robot_cfg
from curobo.util_file import load_yaml
from cuakr.planner.planner import AKRPlanner, IKResult, TrajResult
import torch
import os
from pathlib import Path

class TrajectoryPipeline:
    def __init__(self, task: TaskDescription, output_base_dir: str = "output"):
        self.task = task
        self.output_base_dir = output_base_dir
        self.akr_robot_cfg = None
        self.ik_result = None
        self.traj_result = None
        self.filtered_traj_result = None
        self._init_planner()
        
    def _init_planner(self):
        scene_cfg = {
            "path": self.task.scene.scene_usd_path,
            "pose": self.task.scene.pose,
        }
        object_cfg = {
            "path": self.task.object.urdf_path,
            "asset_type": self.task.object.asset_type,
            "asset_id": self.task.object.asset_id,
        }
        robot_cfg = self.task.robot.robot_cfg
        
        object_cfg = AKRPlanner.load_object_from_metadata(self.task.scene.metadata_path, object_cfg)
        self.planner = AKRPlanner(scene_cfg, object_cfg, robot_cfg)
                
    
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
    def load_akr_robot(self, path: str):
        self.akr_robot_cfg = process_robot_cfg(load_yaml(path)["robot_cfg"])
    def plan_ik(self):
        """Plan IK solutions for articulated object manipulation."""
        print(f"Planning IK for {len(self.task.goal['angle'])} goal angles...")
        
        ik_result = self.planner.plan_ik(
            grasp_pose=self.task.grasp_pose,
            start_angle=self.task.start["angle"],
            goal_angle=self.task.goal["angle"][0],
            robot_cfg=self.task.robot.robot_cfg,
            handle_link=getattr(self.task.object, 'handle_link', 'link_0')
        )
        
        # Handle multiple goal angles
        if len(self.task.goal["angle"]) > 1:
            all_start_iks = [ik_result.start_ik]
            all_goal_iks = [ik_result.goal_ik]
            
            for goal_angle in self.task.goal["angle"][1:]:
                additional_ik = self.planner.plan_ik(
                    grasp_pose=self.task.grasp_pose,
                    start_angle=self.task.start["angle"],
                    goal_angle=goal_angle,
                    robot_cfg=self.task.robot.robot_cfg,
                    handle_link=getattr(self.task.object, 'handle_link', 'link_0')
                )
                all_start_iks.append(additional_ik.start_ik)
                all_goal_iks.append(additional_ik.goal_ik)
            
            # Randomly choose start 60 if more than 60
            if len(all_start_iks) > 60:
                idx = torch.randperm(len(all_start_iks))[:60]
                all_start_iks = [all_start_iks[i] for i in idx]
            start_iks = torch.cat(all_start_iks, dim=0)
            goal_iks = torch.cat(all_goal_iks, dim=0)
            ik_result = IKResult(start_ik=start_iks, goal_ik=goal_iks)
        
        self.ik_result = ik_result
        print(f"IK planning completed: {ik_result.start_ik.shape[0], ik_result.goal_ik.shape[0]} solutions")
        return ik_result

    def plan_traj(self, batch_size: int = 10):
        """Plan AKR trajectories."""
        print("Planning trajectories...")
        traj_result = self.planner.plan_traj(self.ik_result, self.akr_robot_cfg, batch_size=batch_size)
        self.traj_result = traj_result
        print(f"Trajectory planning completed: {traj_result.success.sum().item()}/{len(traj_result.success)} successful")
        return traj_result
    
    def filter_traj(self):
        """Apply filtering to trajectories."""
        print("Filtering trajectories...")
        self.filtered_traj_result = self.planner.traj_filter(self.traj_result, self.akr_robot_cfg)
        print(f"Filtering completed: {self.filtered_traj_result.success.sum().item()}/{len(self.filtered_traj_result.success)} passed")
        return self.filtered_traj_result
    
    def save_results(self, grasp_id: int):
        """Save results to organized directory."""
        output_dir = self._get_output_directory(grasp_id)
        
        if self.ik_result is not None:
            ik_path = os.path.join(output_dir, "ik_data.pt")
            self.planner.save_ik(self.ik_result, ik_path)
            print(f"Saved IK results to: {ik_path}")
        
        if self.traj_result is not None:
            traj_path = os.path.join(output_dir, "traj_data.pt")
            self.planner.save_traj(self.traj_result, traj_path)
            print(f"Saved trajectory results to: {traj_path}")
        
        if self.filtered_traj_result is not None:
            filtered_path = os.path.join(output_dir, "filtered_traj_data.pt")
            self.planner.save_traj(self.filtered_traj_result, filtered_path)
            print(f"Saved filtered trajectory results to: {filtered_path}")
        
        return output_dir
