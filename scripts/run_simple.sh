python scripts/pipeline/1_generate_plans.py --exp multi_object_open --scene scene_3_seed_3 --object 11622

python scripts/pipeline/2_render_dataset.py --exp single_object_open_test --scene scene_0_seed_0 --object 7221 --headless
python scripts/pipeline/2_render_dataset.py --exp single_object_open_test --scene scene_1_seed_1 --object 7221 --headless

python scripts/pipeline/2_render_dataset.py --exp single_object_open_test --headless --max-episodes 10

python scripts/pipeline/2_render_dataset.py --exp single_object_open_test --max-episodes 10

lerobot-dataset-viz \
    --repo-id single_object_open_test \
    --root data/single_object_open_test/lerobot/single_object_open_test \
    --episode-index 0 \
    --video-backend pyav

python third_party/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id single_object_open_test \
    --root data/single_object_open_test/lerobot/single_object_open_test \
    --episode-index 0 \
    --video-backend pyav


# Remote Viusalization
python third_party/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id multi_object_open_7221_scene_0_seed_0 \
    --root data/automoma-docker-1/multi_object_open/lerobot/multi_object_open_7221_scene_0_seed_0 \
    --episode-index 0 \
    --video-backend pyav \
    --save 1 \
    --output-dir ./viz_results


# Remote Viusalization
python third_party/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id multi_object_open_46197_scene_30_seed_30 \
    --root data/automoma-docker-4/multi_object_open/lerobot/multi_object_open_46197_scene_30_seed_30 \
    --episode-index 0 \
    --video-backend pyav 
    
lerobot-train \
  --policy.type=act \
  --batch_size=128 \
  --steps=10000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=1000 \
  --job_name=act_single_object_open_test \
  --dataset.repo_id=single_object_open_test \
  --dataset.root=data/single_object_open_test/lerobot/single_object_open_test \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.optimizer_lr=1e-4 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=true \
  --output_dir=outputs/train/act_single_object_open_test_2

lerobot-train \
  --policy.type=diffusion \
  --batch_size=128 \
  --steps=10000 \
  --log_freq=50 \
  --eval_freq=100 \
  --save_freq=1000 \
  --job_name=dp_single_object_open_test \
  --dataset.repo_id=single_object_open_test \
  --dataset.root=data/single_object_open_test/lerobot/single_object_open_test \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=true \
  --output_dir=outputs/train/dp_single_object_open_test_2


exp_name="multi_object_open_7221_scene_0_seed_0"
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
  --dataset.root=data/automoma-docker-1/multi_object_open/lerobot/$exp_name \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.optimizer_lr=1e-4 \
  --policy.push_to_hub=false \
  --policy.device=cuda:6 \
  --wandb.enable=true \
  --output_dir=outputs/train/act_$exp_name 


lerobot-train \
  --policy.type=diffusion \
  --batch_size=128 \
  --steps=10000 \
  --log_freq=50 \
  --eval_freq=100 \
  --save_freq=1000 \
  --job_name=dp_multi_object_open_7221_scene_0_seed_0 \
  --dataset.repo_id=multi_object_open_7221_scene_0_seed_0 \
  --dataset.root=data/automoma-docker-1/multi_object_open/lerobot/multi_object_open_7221_scene_0_seed_0 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=true \
  --output_dir=outputs/train/dp_multi_object_open_7221_scene_0_seed_0


python scripts/pipeline/4_evaluate.py \
  --exp single_object_open_test  \
  --headless  \
  --policy-type act   \
  --checkpoint outputs/train/act_single_object_open_test_2/checkpoints/last/pretrained_model  \
  --initial-state-path data/single_object_open_test/traj/summit_franka/scene_0_seed_0/7221/grasp_0000/stage_0/start_iks.pt


python scripts/pipeline/4_evaluate.py \
  --exp single_object_open_test  \
  --policy-type act   \
  --checkpoint outputs/train/act_single_object_open_test_2/checkpoints/last/pretrained_model  \
  --initial-state-path data/single_object_open_test/traj/summit_franka/scene_0_seed_0/7221/grasp_0000/stage_0/start_iks.pt


# Evaluate with multi_object_open configuration
python scripts/pipeline/4_evaluate.py \
  --exp multi_object_open  \
  --policy-type act


python scripts/pipeline/4_evaluate.py \
  --exp single_object_open_test  \
  --policy-type act   \
  --checkpoint outputs/train/act_single_object_open_test_2/checkpoints/last/pretrained_model  \
  --initial-state-path data/single_object_open_test/traj/summit_franka/scene_0_seed_0/7221/grasp_0000/stage_0/start_iks.pt