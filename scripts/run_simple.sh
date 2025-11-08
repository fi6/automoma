python scripts/pipeline_plan.py --scene_dir output/collect/infinigen_scene_100 \
     --plan_dir output/collect/traj --robot_name summit_franka

python scripts/pipeline_plan.py --scene_dir output/collect/infinigen_scene_100 \
     --plan_dir output/collect/traj --robot_name summit_franka --stats_only

python scripts/pipeline_collect.py --scene_dir output/collect/infinigen_scene_100 \
     --plan_dir output/collect/traj --robot_name summit_franka --num_episodes 1000


# Table 1.1
python scripts/pipeline_plan.py --scene_dir output/collect_table1/infinigen_scene \
     --plan_dir output/collect_table1/traj --robot_name summit_franka


python scripts/pipeline_plan.py --scene_dir output/collect_table1/infinigen_scene \
     --plan_dir output/collect_table1/traj --robot_name summit_franka --stats_only


python scripts/pipeline_collect.py --scene_dir output/collect_table1/infinigen_scene \
     --plan_dir output/collect_table1/traj --robot_name summit_franka --num_episodes 6400


python scripts/pipeline_collect.py --scene_dir output/collect_table1/infinigen_scene \
     --plan_dir output/collect_table1/traj --robot_name summit_franka --stats_only

# Table 1.2
python scripts/pipeline_plan.py --scene_dir output/collect_table1/infinigen_scene \
     --plan_dir output/collect_table1/traj --robot_name summit_franka_fixed_base

python scripts/pipeline_collect.py --scene_dir output/collect_table1/infinigen_scene \
     --plan_dir output/collect_table1/traj --robot_name summit_franka_fixed_base --num_episodes 6400

python scripts/pick_data_automoma.py --mode collect --output_dir output/collect_table1/traj
python scripts/pick_data_automoma.py --mode pick --output_dir output/collect_table1/traj --link
