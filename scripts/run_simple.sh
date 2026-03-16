# =============================================================================
# 1. PLAN — generate trajectories
# =============================================================================
# Default: Microwave 7221, scene_0_seed_0
python scripts/plan.py

# Override scene / object from CLI
python scripts/plan.py scene_name=scene_0_seed_0 object_id=7221 planner.visualize_collision=true

# 输出同时在terminal和logs/plan_test.log下
python scripts/plan.py scene_name=scene_0_seed_0 object_id=7221 planner.visualize_collision=true | tee logs/plan_test.log

# Dishwasher, different scene
python scripts/plan.py object_id=11622 scene_name=scene_0_seed_0

# Debug
python scripts/plan.py scene_name=scene_0_seed_0 object_id=7221 planner.visualize_collision=true

# Convert all grasps in a scene
python scripts/debug/convert_traj_backup.py --input_dir data/trajs/summit_franka/microwave_7221/scene_0_seed_0

# Or convert a specific grasp
python scripts/debug/convert_traj_backup.py --input_dir data/trajs/summit_franka/microwave_7221/scene_0_seed_0/grasp_0001

# Debug IK solutions for a specific grasp
bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 \
  --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/grasp_0001/ik_data.pt \
  --num_episodes 1000 \
  --set_state

# Debug a specific per-grasp trajectory
bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 \
  --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/grasp_0001/traj_data.pt

# Debug the final merged 12D training trajectory
# (exam the code, this is the gt trajectory)
bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 \
  --debug_file .idea/data/trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data_train.pt \
  --num_episodes 1000 \
  --set_state

bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 \
  --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data_train.pt \
  --num_episodes 1000 \
  --set_state



# Custom config file
# python scripts/plan.py --config configs/my_plan.yaml


# =============================================================================
# 2. RECORD — replay trajectories in IsaacLab-Arena
# =============================================================================
python scripts/prepare_object.py --object_type Microwave --object_id 7221
bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 30
bash scripts/run_pipeline.sh record microwave_7221 scene_1_seed_1 30 --interpolated 2
bash scripts/run_pipeline.sh convert microwave_7221 scene_1_seed_1 30



python scripts/prepare_object.py --object_type Dishwasher --object_id 11622
bash scripts/run_pipeline.sh record dishwasher_11622 scene_0_seed_0 30 --set_state --disable_collision

/World/envs/env_0/dishwasher_11622/link_0/collisions/original_9/World/mesh
/World/envs/env_0/dishwasher_11622/link_0/collisions/new_2/World/mesh

/World/envs/env_0/Robot/panda_hand/collisions/hand_gripper/World/mesh
/World/envs/env_0/Robot/panda_hand/collisions/mesh_1/box
/World/envs/env_0/Robot/panda_leftfinger/collisions/finger/World/mesh
/World/envs/env_0/Robot/panda_rightfinger/collisions/finger/World/mesh

bash scripts/run_pipeline.sh record dishwasher_11622 scene_0_seed_0 30 --set_state --disable_collision --headless
bash scripts/run_pipeline.sh record dishwasher_11622 scene_0_seed_0 30 --set_state --disable_collision --interpolated 1000 --device cpu

bash scripts/run_pipeline.sh convert dishwasher_11622 scene_0_seed_0 30

lerobot-dataset-viz \
    --repo-id automoma/summit_franka_open-dishwasher_11622-scene_0_seed_0-30 \
    --root data/lerobot/automoma/summit_franka_open-dishwasher_11622-scene_0_seed_0-30 \
    --episode-index 0 \
    --video-backend pyav