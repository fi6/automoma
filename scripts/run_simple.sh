python scripts/pipeline_collect.py --scene_dir output/collect/infinigen_scene_100 \
     --plan_dir output/collect/traj --robot_name summit_franka --num_episodes 1000


python scripts/pick_data_automoma.py --mode collect
python /home/xinhai/automoma/scripts/pick_data_automoma.py --mode pick --link