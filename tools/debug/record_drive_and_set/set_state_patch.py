"""Local set-state action patching for AutoMoMa debug wrappers."""

from __future__ import annotations

from typing import Sequence


def parse_joint_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    return names or None


def patch_summit_franka_set_state_object_joints(object_joint_names: Sequence[str] | None) -> None:
    """Limit SummitFranka AutoMoMa set-state object joints for 13D trajectories.

    IsaacLab-Arena's default set-state action uses all joints on the object when
    ``object_joint_names`` is unset. Some AutoMoMa objects, such as ovens, carry
    extra articulated joints even though the planner trajectory only controls
    ``joint_0``. This patch preserves the existing action class and config, but
    changes the wrapper config constructor to pass the requested object joint
    list.
    """

    if not object_joint_names:
        return

    import isaaclab_arena.embodiments.summit_franka.summit_franka as summit_franka

    names = list(object_joint_names)

    class PatchedSummitFrankaAutomomaSetStateActionsCfg:
        set_state_action = summit_franka.SummitFrankaAutomomaSetStateActionCfg(
            asset_name="robot",
            joint_names=summit_franka._SUMMIT_FRANKA_ACTION_JOINT_NAMES,
            object_asset_name=None,
            object_joint_names=names,
        )

        def __init__(self, object_asset_name: str | None = None):
            self.set_state_action = summit_franka.SummitFrankaAutomomaSetStateActionCfg(
                asset_name="robot",
                joint_names=summit_franka._SUMMIT_FRANKA_ACTION_JOINT_NAMES,
                object_asset_name=object_asset_name,
                object_joint_names=names,
            )

    summit_franka.SummitFrankaAutomomaSetStateActionsCfg = PatchedSummitFrankaAutomomaSetStateActionsCfg

