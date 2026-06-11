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
