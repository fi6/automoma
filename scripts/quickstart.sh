#!/usr/bin/env bash
# Print categorized AutoMoMa command examples. Run from the repository root.

set -euo pipefail

cat <<'COMMANDS'
# AutoMoMa quickstart command reference
# Run from the repository root. Replace object/scene/episode values as needed.
# Default example: microwave_7221, scene_0_seed_0, 10 episodes.

# =============================================================================
# 0. Environment
# =============================================================================
conda activate automoma

# Optional: configure Isaac Sim / IsaacLab / asset roots for interactive sim tools.
# If Isaac Sim is installed outside the active Python env, set IsaacSim_ROOT first.
# export IsaacSim_ROOT=/path/to/isaac-sim
source scripts/setup_sim_env.sh

# Show this command reference.
bash scripts/quickstart.sh

# =============================================================================
# 1. Asset preparation helpers
# =============================================================================
# Fix object URDF/mesh names only, without launching Isaac Sim.
python tools/assets/prepare_object.py --object_type Microwave --object_id 7221 --fix_only

# Fix object assets and convert URDF -> USD. Requires IsaacLab / Isaac Sim runtime.
python tools/assets/prepare_object.py --object_type Microwave --object_id 7221

# Prepare one scene or all scenes.
python tools/assets/prepare_scene.py --scene_name scene_0_seed_0
python tools/assets/prepare_scene.py --scene_name all

# Run the standalone scene preparation pipeline.
python tools/assets/prepare_scene_pipeline.py --scene-name scene_0_seed_0

# =============================================================================
# 2. Planning
# =============================================================================
# Use configs/plan.yaml defaults.
python scripts/plan.py

# Plan train/test trajectories for one object and scene.
python scripts/plan.py object_id=7221 scene_name=scene_0_seed_0 mode=train
python scripts/plan.py object_id=7221 scene_name=scene_0_seed_0 mode=test

# Equivalent shell wrapper, with logs under logs/plan/<object_id>/<scene_name>/.
bash scripts/run_pipeline.sh plan 7221 scene_0_seed_0 train
bash scripts/run_pipeline.sh plan 7221 scene_0_seed_0 test

# Small smoke planning run into an isolated output root.
python scripts/plan.py object_id=7221 scene_name=scene_0_seed_0 mode=train \
  output_dir=data/smoke/trajs \
  planner.output.max_successful_trajectories=10 \
  resume=false

# Debug planner collision visualization.
python scripts/plan.py object_id=7221 scene_name=scene_0_seed_0 \
  planner.visualize_collision=true

# =============================================================================
# 3. Record, replay, and trajectory debug
# =============================================================================
# Record planned train trajectories to one merged HDF5.
# Base actions are absolute by default; add --mobile_base_relative only for relative-delta datasets.
bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 10 --headless

# Record with explicit trajectory and output HDF5 paths.
bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 10 \
  --traj_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train/traj_data_train.pt \
  --dataset_file data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5 \
  --headless

# Record one HDF5 per episode instead of one merged file.
bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 10 \
  --split_episodes \
  --dataset_file data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10 \
  --headless

# Record while filtering demos with the eval-style success rule.
bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 10 \
  --validate_record_success \
  --headless

# Replay trajectories without writing HDF5 demos.
bash scripts/run_pipeline.sh replay microwave_7221 scene_0_seed_0 4 --headless

# Replay a targeted subset and write lightweight metrics.
bash scripts/run_pipeline.sh replay microwave_7221 scene_0_seed_0 4 \
  --metrics \
  --metrics_file debug/replay_probe/microwave_scene0.csv \
  --episode_indices 0,1,2,3 \
  --headless

# GUI replay for visual inspection on a machine with a display.
bash scripts/run_pipeline.sh replay microwave_7221 scene_0_seed_0 1 \
  --no-headless \
  --enable_cameras \
  --episode_indices 0

# Debug IK, per-grasp, or merged trajectory .pt files in IsaacLab-Arena.
bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 \
  --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train/grasp_0000/ik_data.pt
bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 \
  --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train/grasp_0000/traj_data.pt
bash scripts/run_pipeline.sh debug microwave_7221 scene_0_seed_0 \
  --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train/traj_data_train.pt

# =============================================================================
# 4. Dataset conversion and visualization
# =============================================================================
# Convert raw AutoMoMa HDF5 to LeRobot format.
bash scripts/run_pipeline.sh convert lerobot microwave_7221 scene_0_seed_0 10
# Split per-episode HDF5 directories are converted directly without a temporary merged HDF5.

# Convert with explicit input/output locations and repo id.
bash scripts/run_pipeline.sh convert lerobot microwave_7221 scene_0_seed_0 10 \
  --data_root data/automoma \
  --hdf5_name summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5 \
  --repo_id automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10 \
  --output_dir data/lerobot/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10

# Visualize a LeRobot dataset.
lerobot-dataset-viz \
  --repo-id automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10 \
  --root data/lerobot/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10 \
  --episode-index 0 \
  --video-backend pyav

# Visualize raw AutoMoMa HDF5 recordings in Rerun.
python tools/dataset/viz_hdf5.py \
  data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5
python tools/dataset/viz_hdf5.py \
  data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5 \
  --demo-index 0,3-5
python tools/dataset/viz_hdf5.py \
  data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5 \
  --demo-start 10 \
  --demo-count 4 \
  --save \
  --output-dir outputs/viz/raw_hdf5

# Convert between merged HDF5 and per-episode HDF5 layouts.
python tools/dataset/convert_hdf5_layout.py \
  data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5 \
  data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10-split \
  --direction split \
  --mode copy \
  --overwrite
python tools/dataset/convert_hdf5_layout.py \
  data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10-split \
  data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10-merged.hdf5 \
  --direction merge \
  --mode copy \
  --overwrite

# Convert legacy/raw planner trajectories to current 12D train/test files.
python tools/dataset/prepare_traj.py \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0 \
  --mode train

# =============================================================================
# 5. LeRobot training and evaluation
# =============================================================================
# Train ACT / diffusion policies.
bash scripts/run_pipeline.sh train lerobot act microwave_7221 scene_0_seed_0 10
bash scripts/run_pipeline.sh train lerobot diffusion microwave_7221 scene_0_seed_0 10

# Short smoke training run with custom output directory and wandb disabled.
WANDB_MODE=disabled bash scripts/run_pipeline.sh train lerobot act microwave_7221 scene_0_seed_0 10 \
  --dataset.repo_id automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10 \
  --dataset.root data/lerobot/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10 \
  --output_dir outputs/train/smoke/act_microwave_7221_scene_0_seed_0_10 \
  --job_name act_microwave_7221_scene_0_seed_0_smoke \
  --steps 10 \
  --batch_size 2 \
  --wandb.enable=false

# Evaluate a trained LeRobot policy. Base actions are absolute by default.
bash scripts/run_pipeline.sh eval lerobot act microwave_7221 scene_0_seed_0 10 --headless

# Evaluate with explicit checkpoint, test trajectory, and output directory.
bash scripts/run_pipeline.sh eval lerobot act microwave_7221 scene_0_seed_0 10 \
  --policy.path outputs/train/lerobot/act_summit_franka_open-microwave_7221-scene_0_seed_0-10/checkpoints/last/pretrained_model \
  --traj_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/test/traj_data_test.pt \
  --output_dir outputs/eval/lerobot/act_summit_franka_open-microwave_7221-scene_0_seed_0-10 \
  --headless \
  --eval.n_episodes=10

# Evaluate a recorded HDF5 dataset with the runtime eval rule.
bash scripts/run_pipeline.sh record_dataset_eval microwave_7221 scene_0_seed_0 10 \
  --dataset_file data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5 \
  --traj_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/test/traj_data_test.pt \
  --output_dir outputs/eval/record_dataset/microwave_7221_scene_0_seed_0_10 \
  --headless

# =============================================================================
# 6. Diagnostics and plots
# =============================================================================
# Run replay metrics over sampled trajectories for grasp-filter diagnostics.
python tools/debug/run_grasp_filter_metrics.py \
  --max-objects 5 \
  --scenes-per-object 1 \
  --max-episodes-per-scene 50 \
  --interpolated 5 \
  --interpolation-type cubic \
  --decimation 1 \
  --init-steps 5 \
  --keep-going

# Visualize grasp poses and wrist-camera views.
python tools/debug/grasp/viz_grasp_pose.py \
  --camera \
  --object-id 7221 \
  --grasp-ids 0-19 \
  --ik-seeds 64 \
  summit_franka_grasp_viz \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0 \
  --object_center

# Headless grasp camera contact sheets.
python tools/debug/grasp/viz_grasp_pose.py \
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

# Plot action-trace diagnostics if eval/record was run with trace output enabled.
python tools/debug/plots/plot_action_trace_joint_states.py \
  --csv outputs/eval/lerobot/act_summit_franka_open-microwave_7221-scene_0_seed_0-10/action_trace_joint_states.csv \
  --per_episode \
  --output_dir outputs/eval/lerobot/act_summit_franka_open-microwave_7221-scene_0_seed_0-10/trace_plots
python tools/debug/plots/plot_ee_pose_trace.py \
  --trace_csv outputs/eval/lerobot/act_summit_franka_open-microwave_7221-scene_0_seed_0-10/action_trace_joint_states.csv \
  --per_episode_csv outputs/eval/lerobot/act_summit_franka_open-microwave_7221-scene_0_seed_0-10/per_episode_results.csv \
  --output_dir outputs/eval/lerobot/act_summit_franka_open-microwave_7221-scene_0_seed_0-10/ee_pose_plots

# Compare record and eval artifacts for alignment debugging.
python tools/debug/eval_align/compare_record_eval.py \
  --dataset_file data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5 \
  --eval_output_dir outputs/eval/record_dataset/microwave_7221_scene_0_seed_0_10 \
  --output_dir outputs/eval_alignment/record_vs_eval
python tools/debug/eval_align/compare_policy_eval.py \
  --run policy=outputs/eval/lerobot/act_summit_franka_open-microwave_7221-scene_0_seed_0-10 \
  --output_dir outputs/eval_alignment/policy_eval

# =============================================================================
# 7. RoboTwin optional workflow
# =============================================================================
# Convert raw HDF5 for RoboTwin policy code.
bash scripts/run_pipeline.sh convert robotwin microwave_7221 scene_0_seed_0 10 --policy dp3

# Train/evaluate RoboTwin DP3 through the wrapper.
bash scripts/run_pipeline.sh train robotwin dp3 microwave_7221 scene_0_seed_0 10 \
  --gpu_id 0 \
  --output_dir outputs/train/robotwin/dp3_microwave_7221_scene_0_seed_0_10
bash scripts/run_pipeline.sh eval robotwin dp3 microwave_7221 scene_0_seed_0 10 \
  --gpu_id 0 \
  --checkpoint_root outputs/train/robotwin/dp3_microwave_7221_scene_0_seed_0_10 \
  --output_dir outputs/eval/robotwin/dp3_microwave_7221_scene_0_seed_0_10

# =============================================================================
# 8. Release and maintenance helpers
# =============================================================================
# Clean cuRobo editable-build artifacts before rebuilding.
bash tools/dev/clean_curobo_build.sh

# AutoMoMa-500k planning self-test and dry-run.
python tools/release/automoma-500k/plan_automoma_500k.py --self-test
python tools/release/automoma-500k/plan_automoma_500k.py --dry-run --scenes scene_0_seed_0

# Summarize planned trajectory counts.
python tools/release/automoma-500k/trajectory_statistics.py \
  --root data/trajs \
  --output-dir outputs/statistics/automoma-500k

# Verify per-object trajectory counts.
python tools/release/automoma-500k/verify_counts.py \
  --root data/trajs/summit_franka \
  --target-per-object 100000

# More detailed workflow notes live in docs/workflows.md and docs/tools.md.
COMMANDS
