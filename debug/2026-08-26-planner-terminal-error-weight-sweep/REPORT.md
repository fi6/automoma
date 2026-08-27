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

## Final orientation-only candidate

The follow-up sweep separated orientation and position weights. With orientation
at 10x (`20000`) and position left at its installed default 1x (`50000`), the
fixed all-waypoint filter retained 188 trajectories from 198 raw successes:

| Metric (per-trajectory maximum) | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| Object-body position | 0.264 mm | 0.988 mm | 0.998 mm | 0.999 mm |
| Object-body rotation | 0.00183 rad | 0.00644 rad | 0.00982 rad | 0.01056 rad |
| Base yaw excursion | 0.821 rad | 1.933 rad | 2.180 rad | 2.243 rad |

All 188 retained trajectories satisfy both the 1 mm position and 0.02 rad
rotation bounds at every waypoint. Before strict filtering, comparable runs
already had about 96--98% of trajectories below 1 mm, so increasing the
position cost was unnecessary. Orientation 10x materially improves the 5x/5x
candidate's rotation tail (p95 0.01762 and p99 0.02580 rad) without the yield
and smoothness degradation observed when both weights were raised to 20x.

## Recommended isolated-branch candidate

- Running AKR object pose weights: orientation `20000` (10x), position `50000`
  (unchanged 1x default).
- Reject IK pairs with raw `abs(goal_base_z - start_base_z) > 2.0` rad before
  TrajOpt.
- Planner AKR waypoint filter: object-body position `< 0.001 m`, quaternion
  angle `< 0.02 rad`.

The final run produced 198 raw successes and retained 188 after filtering.
Independent FK analysis confirmed that every retained trajectory satisfies the
1 mm / 0.02 rad limits. Intermediate base-yaw overshoot can reach 2.243 rad even
though endpoint pairs are limited to 2 rad; an explicit waypoint base-yaw
boundary remains a separate follow-up.

## Important filter semantics and remaining work

- Existing postprocess robot-EE position/rotation thresholds measure the
  standard Summit-Franka terminal EE pose. The object threshold measures the
  microwave door joint. They do not measure AKR `link_1` body/base drift.
- The planner's position/rotation filter does measure AKR `link_1` at every
  waypoint, because the grasp model configures `ee_link=link_1`.
- The first strict-filter run exposed that cuRobo FK reused its output buffer:
  waypoint FK calls overwrote the saved goal pose, making the position test
  ineffective. The planner now clones the goal position and quaternion before
  walking the trajectory. The final 188-trajectory result is from the fixed
  implementation and was independently rechecked from exported tensors.
- Postprocess must gain a separate per-waypoint AKR object-body FK check and
  repeat it after KS. This is recorded in the repository backlog.
- `base_rotation_limit` is currently not consumed by `filter_traj()`; the
  backlog records this explicitly.
- All runs had collision checking enabled, but this sweep did not export a
  comparable numeric collision-clearance margin.
- Results cover one grasp and two goal angles. Validate the selected candidate
  across multiple grasps before using it for a production wave.

At a representative 0.30--0.38 m radius from the microwave body rotation axis
to the door handle/edge, 0.02 rad corresponds to about 0.60--0.76 cm of lateral
point displacement (`2 r sin(theta/2)`, approximately `r theta`). This is an
object-body orientation-drift interpretation, not the microwave door-joint
opening error.

## Validation

- `test_base_yaw_pair_filter.py`: 2 tests passed.
- `test_grasp_result_aggregation.py`: 1 test passed.
- Python compileall passed with a temporary pycache.
- Recommended YAML and `configs/plan.yaml` values loaded and asserted.
- Full test discovery is independently blocked by the existing image lacking
  `automoma.assets`, required by `tests/test_akr_assets.py`.
