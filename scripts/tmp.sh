# for object_id in 7221 11622 103634 46197 101773; do
#     python scripts/pipeline_plan.py --scene_dir assets/scene/infinigen/kitchen_1130 \
#          --plan_dir output/collect_1205/traj --robot_name summit_franka --object_id ${object_id} --stats_only
# done

# for object_id in 46197 101773 103634 11622 7221; do
#     python scripts/pipeline_plan.py --scene_dir assets/scene/infinigen/kitchen_1130 \
#          --plan_dir output/collect_1205/traj --robot_name summit_franka --object_id ${object_id}
# done

# for object_id in 7221; do
#     python scripts/pipeline_plan.py --scene_dir assets/scene/infinigen/kitchen_1130 \
#          --plan_dir output/collect_1205/traj --robot_name summit_franka --object_id ${object_id}
# done

# python scripts/pipeline_plan.py --scene_dir assets/scene/infinigen/kitchen_1130 \
#      --plan_dir output/collect_1205/traj --robot_name summit_franka --stats_only --object_id 7221

for object_id in 46197 101773 103634 11622; do
    python scripts/pipeline_collect.py --scene_dir assets/scene/infinigen/kitchen_1130 \
         --plan_dir output/collect_1205/traj --robot_name summit_franka --object_id ${object_id} --num_episodes 10
done