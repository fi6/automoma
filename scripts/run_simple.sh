# =============================================================================
# 1. PLAN — generate trajectories
# =============================================================================
# Default: Microwave 7221, scene_0_seed_0
python scripts/plan.py

# Override scene / object from CLI
python scripts/plan.py scene_name=scene_0_seed_0 object_id=7221 planner.visualize_collision=true

# 输出同时在terminal和logs/plan_test.log下
python scripts/plan.py scene_name=scene_0_seed_0 object_id=7221 | tee logs/plan_test.log

# Dishwasher, different scene
python scripts/plan.py object_id=11622 scene_name=scene_0_seed_0

# Debug
python scripts/plan.py scene_name=scene_0_seed_0 object_id=7221 planner.visualize_collision=true

# Convert all grasps in a scene
python scripts/debug/convert_traj_backup.py --input_dir data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train

# Or convert a specific grasp
python scripts/debug/convert_traj_backup.py --input_dir data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train/grasp_0001

# Debug IK solutions for a specific grasp
bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 \
  --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train/grasp_0001/ik_data.pt \
  --num_episodes 1000 \
  --set_state

# Debug a specific per-grasp trajectory
bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 \
  --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train/grasp_0001/traj_data.pt

# Debug the final merged 12D training trajectory
# (exam the code, this is the gt trajectory)
bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 \
  --debug_file .idea/data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train/traj_data_train.pt \
  --num_episodes 1000 \
  --set_state

bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 \
  --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train/traj_data_train.pt \
  --num_episodes 1000 \
  --set_state

# =============================================================================
# DEBUG — visualize grasp poses and wrist camera views
# =============================================================================
# Run from the repo root after: conda activate lerobot-arena
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Interactive viewer for Microwave 7221 grasps 0-19.
# Controls: N/Right/Down=next, B/Left/Up=prev, R=refresh, Q/Esc=quit.
python scripts/debug/grasp_convert/viz_grasp_pose.py \
  --camera \
  --object-id 7221 \
  --grasp-ids 0-19 \
  --ik-seeds 64 \
  summit_franka_grasp_viz \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0 \
  --object_center

# Headless batch export of ego_topdown + ego_wrist + fix_local camera sheets.
python scripts/debug/grasp_convert/viz_grasp_pose.py \
  --headless \
  --camera \
  --save-all \
  --object-id 7221 \
  --grasp-ids 0-19 \
  --ik-seeds 64 \
  --hold-frames 1 \
  --save-camera-dir outputs/viz_grasp_pose_7221_0_19 \
  summit_franka_grasp_viz \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0 \
  --object_center

# =============================================================================
# DEBUG — GUI recording (with display)
# =============================================================================
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export AUTOMOMA_SCENE_ROOT=/home/xinhai/projects/lerobot-arena/assets/scene/infinigen/kitchen_1130
export AUTOMOMA_OBJECT_ROOT=/home/xinhai/projects/lerobot-arena/assets/object
export AUTOMOMA_ROBOT_ROOT=/home/xinhai/projects/lerobot-arena/assets/robot

cd third_party/IsaacLab-Arena
python isaaclab_arena/scripts/record_automoma_demos.py \
  --enable_cameras \
  --mobile_base_relative \
  --traj_file /home/xinhai/projects/lerobot-arena/data/trajs/summit_franka/microwave_7221/scene_1_seed_1/train/traj_data_train.pt \
  --dataset_file /home/xinhai/projects/lerobot-arena/data/automoma/test_gui.hdf5 \
  --num_episodes 1 \
  summit_franka_open_door \
  --object_name microwave_7221 \
  --scene_name scene_1_seed_1 \
  --object_center

# Convert to LeRobot
python isaaclab_arena_gr00t/data_utils/convert_hdf5_to_lerobot_v30.py \
  --yaml_file isaaclab_arena_gr00t/config/summit_franka_manip_config.yaml \
  --data_root /home/xinhai/projects/lerobot-arena/data/automoma \
  --hdf5_name test_gui.hdf5 \
  --repo_id automoma/test_gui \
  --output_dir /home/xinhai/projects/lerobot-arena/data/lerobot/automoma/test_gui


# Custom config file
# python scripts/plan.py --config configs/my_plan.yaml


# =============================================================================
# 2. RECORD — replay trajectories in IsaacLab-Arena
# =============================================================================
python scripts/prepare_object.py --object_type Microwave --object_id 7221
bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 100 --set_state
bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 10 --headless
bash scripts/run_pipeline.sh record microwave_7221 scene_1_seed_1 10 --interpolated 2
bash scripts/run_pipeline.sh convert microwave_7221 scene_0_seed_0 10



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

python scripts/dataset/automoma_dataset_viz.py data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5

# For automoma-30k-convert
python scripts/dataset/automoma_dataset_viz.py /home/xinhai/projects/lerobot-arena/data/automoma_30scenes/automoma-30k-convert/scene_0_seed_0/episode000000.hdf5

rclone copy "123pan:/automoma_30scenes" /media/xinhai/GIANT/Research/AutoMoMa/dataset/automoma_30scenes -P