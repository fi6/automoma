# Articulated-object terminal goals

AKR planning keeps the object's fixed body at its scene anchor while allowing
the robot to choose a different terminal IK configuration. This remains a pose
planning problem: the robot's complete goal joint vector is a trajectory seed,
not a joint configuration that must be tracked.

cuRobo pose solves discard `Goal.goal_state` before optimization and disable
the cspace goal cost. An anchor-only `Goal.goal_pose` therefore does not specify
the desired object joint coordinate, even when the IK seed was calculated at
that coordinate. Correct anchor FK alone cannot distinguish a 54-degree door
from the requested 60-degree door.

AutoMoMa supplies the moving-side link's terminal pose through
`Goal.links_goal_pose`, in addition to the existing fixed-body anchor target.
For the reversed AKR object joint, this link is its URDF parent. The link is
added to the stored FK links without changing the URDF, collision geometry,
robot DOFs, or trajectory serialization. The joint seed remains free to change
as long as the object pose constraints are satisfied. The maintained gradient
profile has a terminal link-pose cost; its zero running link-pose weights allow
the object to move throughout the trajectory rather than demanding the open
pose at every waypoint.

Filtering also requires the final object joint coordinate to match its goal
within `filter.object_joint_tolerance` (default `0.001`: radians for revolute
objects, metres for prismatic objects). Solver success and anchor FK alone are
insufficient. This intentionally rejects trajectories that previously passed
despite missing their object goal; existing datasets are not rewritten.

## Validation

The Lift2/Kitchen060 60-degree passive microwave reproduction used identical
initial and goal IK arrays and the same seed (`1264830514`):

| Selected first candidate | Before | After |
| --- | ---: | ---: |
| Planned terminal door angle | 54.212936 degrees | 60.049065 degrees |
| Physically recorded terminal angle | 54.216040 degrees | 60.032869 degrees |

The corrected physical recording retained the grasp, completed the final hold,
and passed independent contact acceptance without forbidden contact. All
canonical opening-state differences were zero. Six corrected candidates passed
the object endpoint, base excursion, and waypoint anchor gates. The 14 focused
tests cover moving-side link selection (including prismatic joints), rejection
of wrong/nonfinite object goals despite perfect anchor FK, preservation of
alternative robot terminal configurations, base gates, and planning metadata.

This changes planning targets and acceptance, not replay or data formats.
The existing position-command resampling's acceleration violations remain
separately reported; this fix does not claim to resolve them. Other articulated
assets have not been physically revalidated, and the full production matrix
has not been rerun.
