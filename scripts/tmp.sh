<<<<<<< HEAD
python scripts/pipeline_plan.py --scene_dir output/collect/infinigen_scene_100 \
     --plan_dir output/collect_table1_new/traj --robot_name summit_franka_fixed_base

python scripts/pipeline_collect.py --scene_dir output/collect/infinigen_scene_100 \
     --plan_dir output/collect_table1_new/traj --robot_name summit_franka_fixed_base --num_episodes 6400
=======
# python scripts/pipeline_plan.py --scene_dir output/collect_table1_new_new/infinigen_scene_100 \
#      --plan_dir output/collect_table1_new_new/traj --robot_name summit_franka

python scripts/pipeline_collect.py --scene_dir output/collect_table1_new_new/infinigen_scene_100 \
     --plan_dir output/collect_table1_new_new/traj --robot_name summit_franka --num_episodes 12800

# ROBOT_CONFIG="automoma_manip_summit_franka"
# TASK_CONFIG="task_1object_1scene_20pose_new-6400"
# rsync -avzP "/home/yida/projects/automoma/baseline/RoboTwin/policy/DP3/data/${ROBOT_CONFIG}-${TASK_CONFIG}.zarr" \
#       "xinhai@10.2.152.14:/home/xinhai/automoma/baseline/RoboTwin/policy/DP3/data/${ROBOT_CONFIG}-${TASK_CONFIG}.zarr" 

ROBOT_CONFIG="automoma_manip_summit_franka"
TASK_CONFIG="task_1object_1scene_20pose_new_new-12800"
rsync -avzP "/home/yida/projects/automoma/baseline/RoboTwin/policy/DP3/data/${ROBOT_CONFIG}-${TASK_CONFIG}.zarr" \
      "xinhai@10.2.152.14:/home/xinhai/automoma/baseline/RoboTwin/policy/DP3/data/${ROBOT_CONFIG}-${TASK_CONFIG}.zarr" 

rsync -avzP "/home/yida/projects/cuakr-docker/output/infinigen_traj_convert/scene_0_seed_0" \
      "xinhai@10.2.152.14:/home/xinhai/automoma/output/infinigen_traj_convert"
>>>>>>> 836ae64c9a19dcb881fd5b9cc2013a40b5a2ecc5
