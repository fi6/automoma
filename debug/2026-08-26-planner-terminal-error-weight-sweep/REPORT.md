# AKR object stability and base-yaw sweep

Date: 2026-08-26 UTC

## Scope

- Scene `scene_0_seed_0`, microwave `7221`, grasp `2`, goal angles `1.333` and
  `1.230` rad.
- cuRobo collision checking enabled, 12 seeds, 32 trajectory steps.
- Physical GPU2 only (`CUDA_VISIBLE_DEVICES=2`); production jobs were not
  stopped or modified.
- Docker image:
  `sha256:94f7c92d91c3fc8a5a7beaee6cd69d54444c836df96f1614dc6373d697651550`.
- AKR `ee_link=link_1`, so the running `pose_cfg` constrains the microwave
  body/base pose over the whole trajectory.

## Pose-weight sweep

Values are per-trajectory maxima. Position is metres and rotation is radians.
Arm d2/d3 are finite differences per saved planning step because optimized
per-trajectory dt is not persisted in `traj_data.pt`.

| Pose weight | Exported | Object pos p95 | Object rot p95 | Object rot p99 | Base yaw p95 | Arm d2 p95 | Arm d3 p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline `2000/50000` | 261 | 0.000763 | 0.037980 | 0.048243 | 4.8618 | 0.04379 | 0.04040 |
| 2x `4000/100000` | 271 | 0.000784 | 0.031103 | 0.038337 | 4.5825 | 0.04857 | 0.04451 |
| 5x `10000/250000` | 256 | 0.000685 | 0.017854 | 0.030897 | 4.7325 | 0.05374 | 0.04914 |
| 10x `20000/500000` | 215 | 0.000384 | 0.011649 | 0.019155 | 4.2595 | 0.05532 | 0.05239 |
| 20x `40000/1000000` | 196 | 0.000371 | 0.010707 | 0.037363 | 4.2320 | 0.07106 | 0.06617 |

The 20x setting is past the useful knee: relative to 10x it improves rotation
p95 by only about 8%, worsens rotation p99 by about 95%, exports fewer
trajectories, and increases arm d2/d3. The 10x setting still improves rotation
but reduces exported yield by about 18% from baseline. The 5x setting is the
best conservative pose-weight candidate.

## Base-yaw experiments on 5x pose weight

`base_z` is a bounded revolute joint (`[-6, 6]` rad). Its raw coordinate delta,
not a wrapped angle delta, represents the commanded chassis turn.

| Variant | Exported | Object rot p95 | Object rot p99 | Base yaw p50 | Base yaw p95 | Base yaw max | Arm d2 p95 | Arm d3 p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5x only | 256 | 0.017854 | 0.030897 | 1.5619 | 4.7325 | 5.4431 | 0.05374 | 0.04914 |
| 5x + c-space yaw weight 10 | 247 | 0.021278 | 0.035656 | 1.6711 | 4.7941 | 5.4686 | 0.06779 | 0.06233 |
| 5x + raw yaw pair <= 2 rad | 121 | 0.017619 | 0.025801 | 0.8032 | 1.7740 | 2.1030 | 0.04220 | 0.04056 |
| 5x + raw yaw pair <= 3 rad | 175 | 0.028818 | 0.041414 | 1.3303 | 2.8316 | 2.9725 | 0.05387 | 0.04262 |

Increasing `cspace_distance_weight[base_z]` did not reduce chassis rotation in
this cuRobo version and degraded yield and finite-difference smoothness. The
effective fix is to reject large raw start/goal yaw differences after the IK
Cartesian product and before TrajOpt. The 2 rad limit also selected trajectories
with better object-rotation tails and smoother arm motion than the 3 rad limit.

## Recommended isolated-branch candidate

- Running AKR object pose weights: orientation `10000`, position `250000` (5x
  the installed fixed-base defaults).
- Reject IK pairs with raw `abs(goal_base_z - start_base_z) > 2.0` rad before
  TrajOpt.
- Planner AKR waypoint filter: object-body position `< 0.001 m`, quaternion
  angle `< 0.02 rad`.

The 2 rad run produced 121 trajectories under the old 1 cm / 0.05 rad planner
filter. Applying the proposed 1 mm / 0.02 rad bounds post hoc retains 116
(95.9%). Intermediate TrajOpt overshoot can reach 2.103 rad even though endpoint
pairs are limited to 2 rad; an explicit waypoint base-yaw boundary remains a
separate follow-up.

## Important filter semantics and remaining work

- Existing postprocess robot-EE position/rotation thresholds measure the
  standard Summit-Franka terminal EE pose. The object threshold measures the
  microwave door joint. They do not measure AKR `link_1` body/base drift.
- The planner's position/rotation filter does measure AKR `link_1` at every
  waypoint, because the grasp model configures `ee_link=link_1`.
- Postprocess must gain a separate per-waypoint AKR object-body FK check and
  repeat it after KS. This is recorded in the repository backlog.
- `base_rotation_limit` is currently not consumed by `filter_traj()`; the
  backlog records this explicitly.
- All runs had collision checking enabled, but this sweep did not export a
  comparable numeric collision-clearance margin.
- Results cover one grasp and two goal angles. Validate the selected candidate
  across multiple grasps before using it for a production wave.

## Validation

- `test_base_yaw_pair_filter.py`: 2 tests passed.
- `test_grasp_result_aggregation.py`: 1 test passed.
- Python compileall passed with a temporary pycache.
- Recommended YAML and `configs/plan.yaml` values loaded and asserted.
- Full test discovery is independently blocked by the existing image lacking
  `automoma.assets`, required by `tests/test_akr_assets.py`.
