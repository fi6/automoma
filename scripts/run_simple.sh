## 01.24
OUTPUT_DIR=output/collect_0123
OUTPUT_DIR=output/automoma-docker-1/collect_0123
OUTPUT_DIR=output/automoma-docker-2/collect_0123
OUTPUT_DIR=output/automoma-docker-3/collect_0123
OUTPUT_DIR=output/automoma-docker-4/collect_0123
OUTPUT_DIR=output/automoma-docker-5/collect_0123

python scripts/pick_data_automoma.py --mode collect --output_dir ${OUTPUT_DIR}/traj
python scripts/pick_data_automoma.py --mode pick --output_dir ${OUTPUT_DIR}/traj --link


python scripts/pipeline_plan.py --scene_dir output/collect/infinigen_scene_100 \
     --plan_dir output/collect/traj --robot_name summit_franka

for object_id in 7221 11622 103634 46197 101773; do
    python scripts/pipeline_plan.py --scene_dir assets/scene/infinigen/kitchen_1130 \
         --plan_dir output/collect_1205/traj --robot_name summit_franka --object_id ${object_id}
done

python scripts/pipeline_plan.py --scene_dir assets/scene/infinigen/kitchen_1130 \
     --plan_dir output/collect_1205/traj --robot_name summit_franka --object_id 7221

python scripts/pipeline_plan.py --scene_dir output/collect/infinigen_scene_100 \
     --plan_dir output/collect/traj --robot_name summit_franka_fixed_base

python scripts/pipeline_plan.py --scene_dir output/collect/infinigen_scene_100 \
     --plan_dir output/collect/traj --robot_name summit_franka --stats_only

python scripts/pipeline_collect.py --scene_dir output/collect/infinigen_scene_100 \
     --plan_dir output/collect/traj --robot_name summit_franka --num_episodes 6400


python scripts/pipeline_plan.py \
  --scene_dir output/collect/infinigen_scene_100 \
  --plan_dir output/collect/traj \
  --robot_name summit_franka \
  --ik-only \
  --record-clustering-stats

python scripts/pipeline_plan.py \
  --scene_dir output/collect/infinigen_scene_100 \
  --plan_dir output/collect/traj \
  --robot_name summit_franka \
  --record-clustering-stats \
  --stats_only

# Analyze results
python scripts/analyze_clustering_stats.py

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


python scripts/debug/batch_usd_preprocess.py
