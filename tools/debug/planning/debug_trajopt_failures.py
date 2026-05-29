import sys
import torch
import os
import argparse
from omegaconf import OmegaConf

from automoma.planning.planner import CuroboPlanner
from automoma.planning.pipeline import stack_iks_angle

def main():
    cfg = OmegaConf.load("configs/plan.yaml")
    planner = CuroboPlanner(cfg)

    # Setup Environment
    object_cfg = {
        "pose": cfg["scene"]["infinigen"]["pose"],
        "path": cfg.objects["7221"].urdf_path,
        "dimensions": [0.3711111843585968, 0.576832115650177, 0.32584288716316223],
        "asset_type": "Microwave",
        "asset_id": "7221",
    }
    scene_cfg = {
        "pose": cfg.scene.infinigen.pose,
        "path": os.path.join(cfg.scene_dir, "scene_0_seed_0", cfg.scene.infinigen.usd_subpath),
        "metadata_path": os.path.join(cfg.scene_dir, "scene_0_seed_0", cfg.scene.infinigen.metadata_subpath),
    }

    planner.setup_env(scene_cfg, object_cfg)

    # Load IK
    from automoma.core.types import IKResult
    start_ik_dict = torch.load("data/trajs/summit_franka/microwave_7221/scene_0_seed_0/grasp_0000/ik_data.pt")
    goal_ik_dict = torch.load("data/trajs/summit_franka/microwave_7221/scene_0_seed_0/grasp_0000/ik_goal_data.pt")
    
    if isinstance(start_ik_dict, dict):
        start_iks = start_ik_dict.get("iks", start_ik_dict.get("start_ik", list(start_ik_dict.values())[0]))
        goal_iks = goal_ik_dict.get("iks", goal_ik_dict.get("goal_ik", list(goal_ik_dict.values())[0]))
        start_ik = IKResult(iks=start_iks, target_poses=None)
        goal_ik = IKResult(iks=goal_iks, target_poses=None)
    else:
        # It's a tensor
        start_ik = IKResult(iks=start_ik_dict, target_poses=None)
        goal_ik = IKResult(iks=goal_ik_dict, target_poses=None)

    # Cluster to small set
    start_mask = planner.cluster_ik(start_ik, {"kmeans_clusters": 5, "ap_fallback_clusters": 5})
    goal_mask = planner.cluster_ik(goal_ik, {"kmeans_clusters": 5, "ap_fallback_clusters": 5})

    start_clustered = start_ik.iks[start_mask]
    goal_clustered = goal_ik.iks[goal_mask]

    # Stack angles
    start_with_angle = stack_iks_angle(start_clustered, 0.0)
    goal_with_angle = stack_iks_angle(goal_clustered, -1.57)

    akr_robot_cfg_path = os.path.abspath("assets/object/Microwave/7221/summit_franka_7221_0_grasp_0000.yml")
    import yaml
    with open(akr_robot_cfg_path, 'r') as f:
        akr_robot_cfg = yaml.safe_load(f)
    urdf = akr_robot_cfg["robot_cfg"]["kinematics"]["urdf_path"]
    akr_robot_cfg["robot_cfg"]["kinematics"]["urdf_path"] = os.path.abspath(urdf)
    
    motion_gen = planner.init_motion_gen(akr_robot_cfg, enable_collision=False)
    
    traj_cfg_plan = {
        "expand_to_pairs": True,
        "batch_size": 25,
        "joint_cfg": {"joint_0": 1.57},
        "enable_collision": False,
    }

    # Pass the pre-initialized motion_gen to match the debug exactly
    traj_result = planner.plan_traj(start_with_angle, goal_with_angle, akr_robot_cfg, plan_cfg=traj_cfg_plan, motion_gen=motion_gen)

    print("Trajectories planned:", traj_result.trajectories.shape)
    print("Successes:", traj_result.success.sum().item())

    if traj_result.success.sum().item() == 0:
        from curobo.types.robot import JointState
        from curobo.types.math import Goal
        
        js_s = JointState.from_position(planner.tensor_args.to_device(start_with_angle[0:1]))
        js_g = JointState.from_position(planner.tensor_args.to_device(goal_with_angle[0:1]))
        ee = motion_gen.ik_solver.fk(js_g.position).ee_pose
        goal = Goal(goal_pose=ee, goal_state=js_g, current_state=js_s)
        
        result = motion_gen.trajopt_solver.solve_batch(goal)
        print("Status of single solve_batch:", result.status)

if __name__ == "__main__":
    main()
