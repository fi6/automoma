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