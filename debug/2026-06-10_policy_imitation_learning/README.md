# Policy imitation learning debug task

Status: completed on 2026-06-11 using the `lerobot-arena` conda environment.

This task creates two datasets from:

`data/trajs/summit_franka/microwave_7221/scene_0_seed_0/train/traj_data_train.pt`

Settings:

- `same_30`: find the first source trajectory that can be successfully
  drive-recorded, then keep 30 successful recordings of that same trajectory.
- `similar_100`: select a dense cluster of similar successful trajectories,
  record candidates from the center outward, and keep the first 100 successful
  distinct source trajectories.

Recording uses absolute mobile-base actions, matching the successful
`2026-05-13_policy_overfit_single_traj` debug task. The direct recorder command
intentionally omits `--mobile_base_relative`.

Main entrypoint:

```bash
source /home/xinhai/miniconda3/etc/profile.d/conda.sh
conda activate lerobot-arena
source scripts/setup_sim_env.sh
bash debug/2026-06-10_policy_imitation_learning/scripts/run_record_and_convert.sh
```

The script fails fast before recording if `isaaclab` is not importable.

Remote training entrypoints are generated as checked-in scripts:

```bash
bash debug/2026-06-10_policy_imitation_learning/scripts/train_same_30_lerobot.sh
bash debug/2026-06-10_policy_imitation_learning/scripts/train_similar_100_lerobot.sh
```

Eval command templates:

```bash
bash debug/2026-06-10_policy_imitation_learning/scripts/eval_same_30_lerobot.sh act
bash debug/2026-06-10_policy_imitation_learning/scripts/eval_similar_100_lerobot.sh act
```

Final artifacts:

- `assets/source_traj_summary.json`
- `assets/same_30/traj_data_train_same_30.pt`
- `assets/same_30/same_30_mapping.json`
- `assets/same_30/dataset_abs_drive_same_30.hdf5`
- `assets/similar_100/traj_data_train_similar_100.pt`
- `assets/similar_100/similar_100_mapping.json`
- `assets/similar_100/dataset_abs_drive_similar_100.hdf5`
- `output/same_30/lerobot_dataset/`
- `output/similar_100/lerobot_dataset/`
- `output/run_record_and_convert.log`

Source summary:

- source trajectory count: 2500
- planner `traj_success` count: 2500
- source tensor shape: `traj_robot == (2500, 36, 12)`

`same_30` result:

- selected source index: 0
- record attempts: 45
- successful recordings: 45
- saved final demos: 30
- final HDF5 samples: 5280
- final trajectory shape: `traj_robot == (30, 36, 12)`

`similar_100` cluster selection:

- selected cluster id: 4
- cluster size: 164
- center source index: 2145
- candidates written: 164

`similar_100` result:

- record attempts: 164
- successful recordings: 105
- saved final demos: 100
- failed/removed candidates: 59
- unique source indices in final mapping: 100
- final HDF5 samples: 17600
- final trajectory shape: `traj_robot == (100, 36, 12)`

Validation commands run:

```bash
python debug/2026-06-10_policy_imitation_learning/scripts/policy_il_tools.py validate-final \
  --traj debug/2026-06-10_policy_imitation_learning/assets/same_30/traj_data_train_same_30.pt \
  --mapping debug/2026-06-10_policy_imitation_learning/assets/same_30/same_30_mapping.json \
  --hdf5 debug/2026-06-10_policy_imitation_learning/assets/same_30/dataset_abs_drive_same_30.hdf5 \
  --expected-count 30

python debug/2026-06-10_policy_imitation_learning/scripts/policy_il_tools.py validate-final \
  --traj debug/2026-06-10_policy_imitation_learning/assets/similar_100/traj_data_train_similar_100.pt \
  --mapping debug/2026-06-10_policy_imitation_learning/assets/similar_100/similar_100_mapping.json \
  --hdf5 debug/2026-06-10_policy_imitation_learning/assets/similar_100/dataset_abs_drive_similar_100.hdf5 \
  --expected-count 100 \
  --require-distinct
```

Both validations returned `ok: true` with no problems.

## PI-FAST same_30 recovery

Status: completed on 2026-06-12.

Root cause found while debugging PI-FAST:

- The original H100 PI-FAST checkpoint used `policy.use_relative_actions=true`.
- Relative actions were normalized with absolute-action dataset statistics, producing normalized values far outside the FAST tokenizer range.
- The FAST tokenizer round trip clipped these targets, so the old checkpoint could have low teacher loss on clipped tokens but could not recover the intended absolute actions during eval.

Fix used for the successful same_30 run:

- Retrained full PI-FAST on H100 with `--policy.use_relative_actions=false`.
- Used checkpoint:
  `output/same_30/train/lerobot/pi0_fast_abs_no_relative_v2/checkpoints/020000/pretrained_model`
- Copied it locally and created a hardlinked local eval checkpoint with local HF cache paths and safer decoding settings:
  `output/tmp/pi0_fast_abs_same30_020000_no_kv64/pretrained_model`
- Local eval checkpoint settings:
  `use_relative_actions=false`, `use_kv_cache=false`, `max_decoding_steps=64`.

Validation run:

- H100 reload teacher-force diagnostic on checkpoint 020000:
  loss mean `0.045825`, min `4.17e-06`, max `0.184313` over 10 sampled frames.
- Local reload teacher-force diagnostic:
  loss mean `0.039852`, min `6.64e-06`, max `0.112668` over the same 10 sampled frames.
- Local autoregressive action generation sanity check on training frame 0:
  max absolute action error `0.004243`, mean absolute action error `0.001695`.
- Local same_30 smoke eval:
  `1/1`, `pc_success=100.0`.
- Local same_30 full eval:
  `10/10`, `pc_success=100.0`.

Final PI-FAST same_30 eval artifacts:

- `output/eval_local/pi_fast_abs020_same30_20260612_000827/same_30/pi_fast/eval_info.json`
- `output/eval_local/pi_fast_abs020_same30_20260612_000827/same_30/pi_fast/per_episode_results.csv`
- `output/eval_local/pi_fast_abs020_same30_20260612_000827/same_30/pi_fast/action_trace_joint_states.csv`
- `output/eval_local/pi_fast_abs020_same30_20260612_000827/same_30/pi_fast/videos/`

Third-party changes made during PI-FAST debugging are committed separately in
`third_party/lerobot` and checkpointed in the top-level repo. They cover PI-FAST
token decoding compatibility, causal-mask compatibility, and processor construction
for PI-FAST pretrained training.

## PI-FAST similar_100 eval

Status: completed on 2026-06-12.

Checkpoint and configuration:

- H100 training job: `pi0fast_abs_sim100_v2`, 30000 steps.
- H100 checkpoint:
  `output/similar_100/train/lerobot/pi0_fast_abs_no_relative_v2/checkpoints/030000/pretrained_model`
- Local eval checkpoint:
  `output/tmp/pi0_fast_abs_similar100_030000_no_kv64/pretrained_model`
- Local eval settings:
  `use_relative_actions=false`, `use_kv_cache=false`, `max_decoding_steps=64`.
- Eval trajectory file:
  `assets/eval/similar_100/traj_data_eval_random50_seed0.pt`

Teacher-force diagnostics:

- H100 checkpoint 030000 diagnostic:
  mean loss `0.339955`, min `0.076783`, max `0.850763` over 11 sampled frames.
- Local checkpoint 030000 diagnostic:
  mean loss `0.339158`, min `0.076294`, max `0.849588` over the same sampled frames.

Eval runs:

- Smoke eval:
  `output/eval_local/pi_fast_abs030_similar100_no_kv64_smoke_20260612_043655/similar_100/pi_fast/`
  returned `1/1`, `pc_success=100.0`.
- Initial fixed-subset eval using the historical random-with-replacement wrapper behavior:
  `output/eval_local/pi_fast_abs030_similar100_no_kv64_20260612_044058/similar_100/pi_fast/`
  returned `5/50`, `pc_success=10.0`.
  This run loaded the fixed random50 subset but sampled start states with replacement
  (`50` resets, `33` unique trajectory starts), so it is kept as a diagnostic rather
  than the primary fixed-50 result.
- Primary sequential fixed-50 eval:
  `output/eval_local/pi_fast_abs030_similar100_no_kv64_sequential50_20260612_060328/similar_100/pi_fast/`
  returned `8/50`, `pc_success=16.0`.

Primary sequential eval details:

- Trajectory start indices were verified as the exact prefix `0..49`.
- Success episodes: `[19, 28, 29, 31, 32, 33, 44, 49]`.
- `final_door_open`: `16/50`.
- `final_engaged`: `21/50`.
- Both open and engaged: `8/50`.
- Failure categories:
  open without final engagement `8`, engaged without final open `13`, neither `21`.
- Mean final door openness: `0.2345`.
- Mean final handle distance: `0.4083`.
- Action trace:
  `output/eval_local/pi_fast_abs030_similar100_no_kv64_sequential50_20260612_060328/similar_100/pi_fast/action_trace_joint_states.csv`
- Per-episode CSV:
  `output/eval_local/pi_fast_abs030_similar100_no_kv64_sequential50_20260612_060328/similar_100/pi_fast/per_episode_results.csv`
- Eval info:
  `output/eval_local/pi_fast_abs030_similar100_no_kv64_sequential50_20260612_060328/similar_100/pi_fast/eval_info.json`

Eval-only code change:

- Added opt-in `traj_selection_mode=sequential` for IsaacLab-Arena trajectory
  initial states. The default remains the previous random sampling behavior.
- Top-level eval CLI now records `traj_selection_mode` in `eval_info.json`.
