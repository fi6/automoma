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
  --save_freq=5000 \
  --job_name=act_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.optimizer_lr=1e-4 \
  --policy.push_to_hub=false \
  --policy.device=cuda:6 \
  --wandb.enable=true \
  --output_dir=outputs/train/act_$exp_name 

# (2) DP3

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
  --output_dir=outputs/train/dp_$exp_name


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
python scripts/pipeline/2_render_dataset.py --exp single_object_reach --scene scene_0_seed_0 --object 7221 --max-episodes 100 --headless

# Step 3: Train policies

# Step 4: Evaluate policies
python scripts/pipeline/4_evaluate.py \
    --run-dir outputs/train/dp3_single_object_reach_7221_scene_0_seed_0 \
    --dataset_root data/single_object_reach/lerobot/single_object_reach_7221_scene_0_seed_0


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

