from automoma.models.task import TaskDescription, TaskType
from automoma.utils.file import process_robot_cfg
from curobo.util_file import load_yaml
from cuakr.planner.planner import AKRPlanner, IKResult, TrajResult
import torch
import os
from pathlib import Path

class TrajectoryPipeline:
    def __init__(self, task: TaskDescription, output_base_dir: str = "output", record_clustering_stats: bool = False):
        self.task = task
        self.output_base_dir = output_base_dir
        self.record_clustering_stats = record_clustering_stats
        self.akr_robot_cfg = None
        self.ik_result = None
        self.traj_result = None
        self.filtered_traj_result = None
        self.selected_traj_result = None
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
        robot_name = self.task.robot.robot_name
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
            handle_link=getattr(self.task.object, 'handle_link', 'link_0'),
            record_clustering_stats=self.record_clustering_stats,
            selection_strategy="similar",
            target_count=200,
            num_retries=20,
            num_seeds=20000
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
                    handle_link=getattr(self.task.object, 'handle_link', 'link_0'),
                    record_clustering_stats=False,
                    selection_strategy="similar",
                    target_count=100,
                    num_retries=20,
                    num_seeds=20000
                )
                all_start_iks.append(additional_ik.start_ik)
                all_goal_iks.append(additional_ik.goal_ik)
            
            start_iks = torch.cat(all_start_iks, dim=0)
            goal_iks = torch.cat(all_goal_iks, dim=0)
            # Keep clustering stats from first angle if available
            new_ik_result = IKResult(start_ik=start_iks, goal_ik=goal_iks)
            if hasattr(ik_result, 'clustering_stats') and ik_result.clustering_stats is not None:
                new_ik_result.clustering_stats = ik_result.clustering_stats
            ik_result = new_ik_result
        
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
        self.filtered_traj_result = self.planner.traj_filter(
            self.traj_result,
            self.akr_robot_cfg,
            target_count=None
        )
        print(
            f"Filtering completed: {self.filtered_traj_result.success.sum().item()}/{len(self.filtered_traj_result.success)} passed"
        )

        selected_indices = self.planner._select_unique_similar_traj_indices(
            self.filtered_traj_result.trajectories,
            target_count=1000
        )
        if selected_indices.shape[0] == 0:
            self.selected_traj_result = self.planner.traj_fallback()
        else:
            self.selected_traj_result = TrajResult(
                start_states=self.filtered_traj_result.start_states[selected_indices],
                goal_states=self.filtered_traj_result.goal_states[selected_indices],
                trajectories=self.filtered_traj_result.trajectories[selected_indices],
                success=self.filtered_traj_result.success[selected_indices]
            )
        print(
            f"Selected trajectories: {self.selected_traj_result.success.sum().item()}/{len(self.selected_traj_result.success)} kept"
        )
        return self.selected_traj_result
    
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

        if self.selected_traj_result is not None:
            selected_path = os.path.join(output_dir, "selected_traj_data.pt")
            self.planner.save_traj(self.selected_traj_result, selected_path)
            print(f"Saved selected trajectory results to: {selected_path}")
        
        return output_dir
    
    def check_results_exist(self, grasp_id: int) -> bool:
        """Check if results already exist for the given grasp ID."""
        output_dir = self._get_output_directory(grasp_id)
        ik_path = os.path.join(output_dir, "ik_data.pt")
        traj_path = os.path.join(output_dir, "traj_data.pt")
        filtered_path = os.path.join(output_dir, "filtered_traj_data.pt")
        selected_path = os.path.join(output_dir, "selected_traj_data.pt")
        
        return all(os.path.exists(p) for p in [ik_path, traj_path, filtered_path, selected_path])
