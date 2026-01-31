## 01.30
python scripts/compute_urdf_metadata.py \
     --urdf assets/object/Refrigerator/10000/10000_0_scaling.urdf \
     --pose 0.5361366205337185 -4.457830299261277 1.01643887334675 1.0 0.0 0.0 0.0 \
     --metadata assets/scene/mshab/kitchen_0130/scene_0_seed_0/info/metadata.json \
     --object-key "StaticCategoryFactory(Refrigerator_10000_0_scaling_mobility)"

python scripts/compute_urdf_metadata.py \
     --urdf assets/object/Refrigerator/10000/10000_0_scaling.urdf \
     --pose -2.21290059087744 1.0421154010510683 1.0348518115086407 1.0 0.0 0.0 0.0 \
     --metadata assets/scene/mshab/kitchen_0130/scene_1_seed_1/info/metadata.json \
     --object-key "StaticCategoryFactory(Refrigerator_10000_0_scaling_mobility)"

python scripts/compute_urdf_metadata.py \
     --urdf assets/object/Refrigerator/10000/10000_0_scaling.urdf \
     --pose -2.21290059087744 1.0421154010510683 1.0348518115086407 1.0 0.0 0.0 0.0 \
     --metadata assets/scene/mshab/kitchen_0130/scene_2_seed_2/info/metadata.json \
     --object-key "StaticCategoryFactory(Refrigerator_10000_0_scaling_mobility)"

# (-2.172428681511452, 0.24012306028506913, 1.0348)
python scripts/compute_urdf_metadata.py \
     --urdf assets/object/Refrigerator/10000/10000_0_scaling.urdf \
     --pose -2.172428681511452 0.24012306028506913 1.0348 1.0 0.0 0.0 0.0 \
     --metadata assets/scene/mshab/kitchen_0130/scene_3_seed_3/info/metadata.json \
     --object-key "StaticCategoryFactory(Refrigerator_10000_0_scaling_mobility)"

python third_party/cuakr/src/cuakr/planner/planner_reach.py

python examples/example_replay_reach.py \
     --scene-dir assets/scene/mshab/kitchen_0130/scene_0_seed_0  \
     --plan-dir output/collect_0130/traj \
     --robot-name fetch \
     --object-id 10000  \
     --grasp-id 0 \
     --mode replay_ik \
     --stage reach

python examples/example_replay_reach.py \
     --scene-dir assets/scene/mshab/kitchen_0130/scene_0_seed_0  \
     --plan-dir output/collect_0130/traj \
     --robot-name fetch \
     --object-id 10000  \
     --grasp-id 0 \
     --mode replay_traj \
     --stage reach

rm -rf /home/xinhai/projects/automoma/output/collect_0130

## 01.24
OUTPUT_DIR=output/collect_0123
OUTPUT_DIR=output/automoma-docker-1/collect_0123
OUTPUT_DIR=output/automoma-docker-2/collect_0123
OUTPUT_DIR=output/automoma-docker-3/collect_0123
OUTPUT_DIR=output/automoma-docker-4/collect_0123
OUTPUT_DIR=output/automoma-docker-5/collect_0123

python scripts/pick_data_automoma.py --mode collect --output_dir ${OUTPUT_DIR}/traj
python scripts/pick_data_automoma.py --mode pick --output_dir ${OUTPUT_DIR}/traj --link

python scripts/pipeline_plan.py \
     --scene_dir assets/scene/infinigen/kitchen_0913 \
     --plan_dir output/collect_0123/traj \
     --robot_name summit_franka \
     --object_id 7221

python scripts/pipeline_collect.py \
     --scene_dir assets/scene/infinigen/kitchen_0913 \
     --plan_dir output/collect_0123/traj \
     --robot_name summit_franka \
     --num_episodes 1000 \
     --object_id 7221

python scripts/pipeline_plan.py --scene_dir output/collect/infinigen_scene_100 \
     --plan_dir output/collect/traj --robot_name summit_franka

for object_id in 7221 11622 103634 46197 101773; do
    python scripts/pipeline_plan.py --scene_dir assets/scene/infinigen/kitchen_1130 \
         --plan_dir output/collect_0123/traj --robot_name summit_franka --object_id ${object_id}
done

python scripts/pipeline_plan.py --scene_dir assets/scene/infinigen/kitchen_1130 \
     --plan_dir output/collect_0123/traj --robot_name summit_franka --object_id 7221
     
python scripts/pipeline_plan.py --scene_dir output/collect/infinigen_scene_100 \
     --plan_dir output/collect/traj --robot_name summit_franka_fixed_base

python scripts/pipeline_plan.py --scene_dir output/collect/infinigen_scene_100 \
     --plan_dir output/collect/traj --robot_name summit_franka --stats_only

python scripts/pipeline_collect.py --scene_dir assets/scene/infinigen/kitchen_1130 \
     --plan_dir output/collect_0123/traj --robot_name summit_franka --num_episodes 1000


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
