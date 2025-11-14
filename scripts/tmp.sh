python scripts/pipeline_plan.py --scene_dir output/collect/infinigen_scene_100 \
     --plan_dir output/collect/traj --robot_name summit_franka_fixed_base

python scripts/pipeline_collect.py --scene_dir output/collect/infinigen_scene_100 \
     --plan_dir output/collect/traj --robot_name summit_franka_fixed_base --num_episodes 12800