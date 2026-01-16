###############################################################
# Example usage: [EXP] Multi-object open task
###############################################################

# Step 1: Generate plans
python scripts/pipeline/1_generate_plans.py --exp multi_object_open --scene scene_0_seed_0 --object 7221

# Step 2: Render dataset
python scripts/pipeline/2_render_dataset.py --exp single_object_open_test --scene scene_0_seed_0 --object 7221 --headless --max-episodes 10

# Step 3: Train policies
# (1) ACT
exp_name="multi_object_open_7221_scene_0_seed_0"
dataset_root=data/multi_object_open/lerobot/$exp_name
rm -rf outputs/train/act_$exp_name
lerobot-train \
  --policy.type=act \
  --batch_size=128 \
  --steps=100000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=10000 \
  --job_name=act_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.optimizer_lr=1e-4 \
  --policy.push_to_hub=false \
  --policy.device=cuda:6 \
  --wandb.enable=true \
  --output_dir=outputs/train/act_$exp_name \
  --dataset.preload=true

# (2) DP3
exp_name="multi_object_open_7221_scene_0_seed_0"
rm -rf outputs/train/dp3_$exp_name
lerobot-train \
  --policy.type=dp3 \
  --batch_size=128 \
  --steps=10000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=5000 \
  --job_name=dp3_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=data/$exp_name \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=true \
  --output_dir=outputs/train/dp3_$exp_name \
  --dataset.preload=true


# (3) Diffusion Policy
exp_name="multi_object_open_7221_scene_0_seed_0"
dataset_root=data/multi_object_open/lerobot/$exp_name
rm -rf outputs/train/dp_$exp_name
lerobot-train \
  --policy.type=diffusion \
  --batch_size=128 \
  --steps=10000 \
  --log_freq=50 \
  --eval_freq=100 \
  --save_freq=1000 \
  --job_name=dp_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=true \
  --output_dir=outputs/train/dp_$exp_name \
  --dataset.preload=true



# Step 4: Evaluate policies
# (1) ACT
python scripts/pipeline/4_evaluate.py \
    --run-dir outputs/train/act_multi_object_open_7221_scene_0_seed_0 \
    --dataset_root data/multi_object_open/lerobot/multi_object_open_7221_scene_0_seed_0

# (2) DP3
python scripts/pipeline/4_evaluate.py \
    --run-dir outputs/train/dp3_multi_object_open_7221_scene_0_seed_0 \
    --dataset_root data/multi_object_open/lerobot/multi_object_open_7221_scene_0_seed_0

# (3) Diffusion Policy
python scripts/pipeline/4_evaluate.py \
    --run-dir outputs/train/dp_multi_object_open_7221_scene_0_seed_0 \
    --dataset_root data/multi_object_open/lerobot/multi_object_open_7221_scene_0_seed_0



###############################################################
# Example usage: [EXP] Single-object reach task
###############################################################

# Step 1: Generate plans
python scripts/pipeline/1_generate_plans.py --exp single_object_reach --scene scene_0_seed_0 --object 7221

# Step 2: Render dataset
python scripts/pipeline/2_render_dataset.py --exp single_object_reach --scene scene_0_seed_0 --object 7221 --max-episodes 10 --headless

# Step 3: Train policies
# (1) ACT
CUDA_VISIBLE_DEVICES=0
exp_name="single_object_reach_7221_scene_0_seed_0_1000"
dataset_root=data/single_object_reach/lerobot/$exp_name
rm -rf outputs/train/act_$exp_name
lerobot-train \
  --policy.type=act \
  --batch_size=128 \
  --steps=100000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=10000 \
  --job_name=act_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.optimizer_lr=1e-4 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=true \
  --output_dir=outputs/train/act_$exp_name \
  --dataset.preload=true

# (2) DP3
CUDA_VISIBLE_DEVICES=1
exp_name="single_object_reach_7221_scene_0_seed_0_1000_dp3"
dataset_root=data/single_object_reach/lerobot/$exp_name
rm -rf outputs/train/dp3_$exp_name
lerobot-train \
  --policy.type=dp3 \
  --batch_size=128 \
  --steps=300000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=100000 \
  --job_name=dp3_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=true \
  --output_dir=outputs/train/dp3_$exp_name \
  --dataset.preload=true


###############################################################
# Example usage: Utils of lerobot-dataset
###############################################################

# Local Viusalization
lerobot-dataset-viz \
    --repo-id single_object_open_test \
    --root data/single_object_open_test/lerobot/single_object_open_test \
    --episode-index 0 \
    --video-backend pyav

# Remote Viusalization
lerobot-dataset-viz \
    --repo-id multi_object_open_7221_scene_0_seed_0 \
    --root data/automoma-docker-1/multi_object_open/lerobot/multi_object_open_7221_scene_0_seed_0 \
    --episode-index 0 \
    --video-backend pyav \
    --save 1 \
    --output-dir ./viz_results


# Split Dataset
exp_name="multi_object_open_7221_scene_0_seed_0"
dataset_root="$(pwd)/data/multi_object_open/lerobot/$exp_name"
python -m lerobot.scripts.lerobot_edit_dataset \
  --repo_id $dataset_root \
  --operation.type split \
  --operation.splits '{"50": 0.1, "val": 0.1, "train": 0.8}'


# Remove Feature (for dp3)
# p.s. repo_id needs to be absolute path
exp_name="single_object_reach_7221_scene_0_seed_0_dp3"
dataset_root="$(pwd)/data/single_object_reach/lerobot/$exp_name"
python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id $dataset_root \
        --operation.type remove_feature \
        --operation.feature_names "['observation.images.ego_topdown', 'observation.images.ego_wrist', 'observation.images.fix_local', 'observation.depth.ego_topdown', 'observation.depth.ego_wrist', 'observation.depth.fix_local', 'observation.eef']"


exp_name="single_object_reach_7221_scene_0_seed_0"
dataset_root="$(pwd)/data/single_object_reach/lerobot/$exp_name"
python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id $dataset_root \
        --operation.type remove_feature \
        --operation.feature_names "['observation.depth.ego_topdown', 'observation.depth.ego_wrist', 'observation.depth.fix_local', 'observation.eef']"