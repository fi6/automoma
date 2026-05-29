#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ISAAC_ARENA_ROOT = REPO_ROOT / "third_party" / "IsaacLab-Arena"
ISAAC_ENV_ROOT = ISAAC_ARENA_ROOT / "isaaclab-arena-envs"
LEROBOT_SRC_ROOT = REPO_ROOT / "third_party" / "lerobot" / "src"

for path in (str(ISAAC_ARENA_ROOT), str(ISAAC_ENV_ROOT), str(LEROBOT_SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from lerobot.envs import make_env, preprocess_observation
from lerobot.envs.configs import IsaaclabArenaEnv
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

from ee_pose_trace import EE_TRACE_COLUMNS, ee_trace_values, empty_ee_trace_values, make_ee_fk


PER_EPISODE_CSV_COLUMNS = [
    "episode_ix",
    "demo_key",
    "seed",
    "record_success",
    "success",
    "final_door_open",
    "final_door_openness",
    "final_engaged",
    "final_handle_distance",
    "steps",
    "policy_steps",
    "dataset_steps",
    "max_abs_eval_vs_record_state",
    "mean_abs_eval_vs_record_state",
    "final_abs_eval_vs_record_state",
    "final_max_abs_eval_vs_record_state",
    "final_mean_abs_eval_vs_record_state",
    "video_path",
]

ACTION_TRACE_CSV_COLUMNS = [
    "episode_ix",
    "demo_key",
    "policy_step_ix",
    "sim_trace_step_ix",
    "sim_substep_ix",
    "sim_substeps_for_policy",
    "is_policy_action_sample",
    "terminated",
    "truncated",
    "joint_ix",
    "joint_name",
    "dataset_action_raw",
    "prepared_action_abs",
    "interpolated_action_abs",
    "eval_sim_joint_pos_after",
    "record_obs_joint_pos",
    "record_state_joint_pos",
    "abs_eval_vs_record_state",
    *EE_TRACE_COLUMNS,
]

SUMMIT_FRANKA_ACTION_JOINT_NAMES = (
    "base_x",
    "base_y",
    "base_z",
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate record HDF5 actions through the IsaacLab-Arena eval policy path."
    )
    parser.add_argument("--dataset_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--object_name", required=True)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--traj_file", default=None)
    parser.add_argument("--traj_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--policy.device", "--policy_device", dest="policy_device", default="cuda")
    parser.add_argument("--env.device", "--env_device", dest="env_device", default="cuda:0")
    parser.add_argument("--eval.n_episodes", "--n_episodes", dest="n_episodes", type=int, default=None)
    parser.add_argument("--start_demo", type=int, default=0)
    parser.add_argument("--env.episode_length", "--max_steps", dest="max_steps", type=int, default=None)
    parser.add_argument("--max_episodes_rendered", type=int, default=10)
    parser.add_argument("--headless", type=str2bool, default=True)
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument("--init_steps", type=int, default=1)
    parser.add_argument("--interpolated", type=int, default=1)
    parser.add_argument("--interpolation_type", default="linear")
    parser.add_argument("--action_key", default="actions", choices=("actions", "processed_actions", "obs/actions"))
    parser.add_argument("--stop_on_dataset_end", type=str2bool, default=True)
    parser.add_argument("--mobile_base_relative", type=str2bool, default=True)
    parser.add_argument(
        "--base_relative_reference",
        default="eval_state",
        choices=("eval_state", "record_obs"),
        help=(
            "How to reconstruct mobile-base-relative HDF5 actions. "
            "eval_state matches normal policy eval. record_obs is a diagnostic mode "
            "that reconstructs the original absolute base target from HDF5 obs/joint_pos."
        ),
    )
    parser.add_argument("--openness_threshold", type=float, default=0.3)
    parser.add_argument("--handle_distance_threshold", type=float, default=0.1)
    parser.add_argument("--proximity_threshold", type=float, default=0.12)
    parser.add_argument("--proximity_window_steps", type=int, default=8)
    parser.add_argument("--proximity_required_steps", type=int, default=5)
    parser.add_argument("--disable_fingertip_proximity", type=str2bool, default=False)
    parser.add_argument("--debug_visualize_handle", type=str2bool, default=False)
    parser.add_argument("--debug_record_handle_diagnostics", type=str2bool, default=False)
    parser.add_argument("--debug_marker_scale", type=float, default=1.0)
    parser.add_argument("--robot_object_static_friction", type=float, default=None)
    parser.add_argument("--robot_object_dynamic_friction", type=float, default=None)
    parser.add_argument(
        "--disable_success_termination",
        type=str2bool,
        default=True,
        help=(
            "Disable the env's runtime success termination during HDF5 replay. "
            "Record replay disables this termination so the full trajectory is recorded; "
            "the final eval success rule is still computed after replay."
        ),
    )
    parser.add_argument("--debug_action_trace", type=str2bool, default=True)
    parser.add_argument("--action_trace_csv", default=None)
    return parser.parse_args()


def _episode_sort_key(episode_name: str) -> tuple[int, str]:
    try:
        return int(episode_name.rsplit("_", 1)[1]), episode_name
    except (IndexError, ValueError):
        return 1_000_000_000, episode_name


@dataclass
class RecordEpisode:
    demo_key: str
    actions: np.ndarray
    obs_joint_pos: np.ndarray
    state_joint_pos: np.ndarray | None
    object_initial_joint_pos: np.ndarray | None
    object_initial_root_pose: np.ndarray | None
    object_initial_root_velocity: np.ndarray | None
    record_success: bool | None

    @property
    def n_steps(self) -> int:
        return int(self.actions.shape[0])


def _read_dataset(group: h5py.Group, key: str) -> np.ndarray | None:
    if key in group:
        return group[key][:]
    cur: h5py.Group | h5py.Dataset = group
    for part in key.split("/"):
        if not isinstance(cur, h5py.Group) or part not in cur:
            return None
        cur = cur[part]
    return cur[:] if isinstance(cur, h5py.Dataset) else None


def load_record_metadata(dataset_file: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"env_args": {}, "sim_args": {}}
    with h5py.File(dataset_file, "r") as f:
        data = f.get("data")
        if data is None:
            return metadata
        raw_env_args = data.attrs.get("env_args")
        if isinstance(raw_env_args, bytes):
            raw_env_args = raw_env_args.decode("utf-8")
        if isinstance(raw_env_args, str) and raw_env_args:
            try:
                env_args = json.loads(raw_env_args)
            except json.JSONDecodeError:
                logging.warning("Could not parse HDF5 env_args JSON: %s", raw_env_args)
            else:
                metadata["env_args"] = env_args
                metadata["sim_args"] = env_args.get("sim_args", {}) if isinstance(env_args, dict) else {}
    return metadata


def load_record_episodes(dataset_file: Path, action_key: str) -> list[RecordEpisode]:
    with h5py.File(dataset_file, "r") as f:
        if "data" not in f:
            raise KeyError(f"HDF5 file has no 'data' group: {dataset_file}")
        data = f["data"]
        episodes: list[RecordEpisode] = []
        for demo_key in sorted(data.keys(), key=_episode_sort_key):
            demo = data[demo_key]
            actions = _read_dataset(demo, action_key)
            obs_joint_pos = _read_dataset(demo, "obs/joint_pos")
            if actions is None:
                raise KeyError(f"{demo_key} missing action dataset '{action_key}'")
            if obs_joint_pos is None:
                raise KeyError(f"{demo_key} missing 'obs/joint_pos'")

            state_joint_pos = _read_dataset(demo, "states/articulation/robot/joint_position")
            object_initial_joint_pos = None
            object_initial_root_pose = None
            object_initial_root_velocity = None
            initial_articulation = demo.get("initial_state/articulation")
            if isinstance(initial_articulation, h5py.Group):
                for articulation_name in sorted(initial_articulation.keys()):
                    if articulation_name == "robot":
                        continue
                    object_group = initial_articulation[articulation_name]
                    object_initial_joint_pos = _read_dataset(object_group, "joint_position")
                    object_initial_root_pose = _read_dataset(object_group, "root_pose")
                    object_initial_root_velocity = _read_dataset(object_group, "root_velocity")
                    break

            states_articulation = demo.get("states/articulation")
            if object_initial_joint_pos is None and isinstance(states_articulation, h5py.Group):
                for articulation_name in sorted(states_articulation.keys()):
                    if articulation_name == "robot":
                        continue
                    object_group = states_articulation[articulation_name]
                    object_joint_pos = _read_dataset(object_group, "joint_position")
                    object_root_pose = _read_dataset(object_group, "root_pose")
                    object_root_velocity = _read_dataset(object_group, "root_velocity")
                    if object_joint_pos is not None:
                        object_initial_joint_pos = object_joint_pos[:1]
                    if object_root_pose is not None:
                        object_initial_root_pose = object_root_pose[:1]
                    if object_root_velocity is not None:
                        object_initial_root_velocity = object_root_velocity[:1]
                    break

            record_success = demo.attrs.get("success")
            if record_success is not None:
                record_success = bool(record_success)

            episodes.append(
                RecordEpisode(
                    demo_key=demo_key,
                    actions=np.asarray(actions, dtype=np.float32),
                    obs_joint_pos=np.asarray(obs_joint_pos, dtype=np.float32),
                    state_joint_pos=None if state_joint_pos is None else np.asarray(state_joint_pos, dtype=np.float32),
                    object_initial_joint_pos=None
                    if object_initial_joint_pos is None
                    else np.asarray(object_initial_joint_pos, dtype=np.float32),
                    object_initial_root_pose=None
                    if object_initial_root_pose is None
                    else np.asarray(object_initial_root_pose, dtype=np.float32),
                    object_initial_root_velocity=None
                    if object_initial_root_velocity is None
                    else np.asarray(object_initial_root_velocity, dtype=np.float32),
                    record_success=record_success,
                )
            )
    return episodes


class IdentityProcessor:
    def __call__(self, value: Any) -> Any:
        return value


class Hdf5ActionPolicy(torch.nn.Module):
    """A LeRobot-shaped policy that returns actions from one HDF5 demo."""

    def __init__(self, device: str = "cuda", base_relative_reference: str = "eval_state", base_dof: int = 3) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.base_relative_reference = base_relative_reference
        self.base_dof = int(base_dof)
        self.episode: RecordEpisode | None = None
        self.step_ix = 0

    def set_episode(self, episode: RecordEpisode) -> None:
        self.episode = episode
        self.reset()

    def reset(self) -> None:
        self.step_ix = 0

    @property
    def has_action(self) -> bool:
        return self.episode is not None and self.step_ix < self.episode.n_steps

    def select_action(self, observation: dict[str, Any]) -> torch.Tensor:  # noqa: ARG002
        if self.episode is None:
            raise RuntimeError("No HDF5 episode has been selected.")
        if self.step_ix >= self.episode.n_steps:
            action = np.zeros((1, self.episode.actions.shape[-1]), dtype=np.float32)
        else:
            action = self.episode.actions[self.step_ix].copy()
            if self.base_relative_reference == "record_obs" and self.base_dof > 0:
                action[: self.base_dof] = (
                    self.episode.obs_joint_pos[self.step_ix, : self.base_dof] + action[: self.base_dof]
                )
            action = action[None, :]
        self.step_ix += 1
        return torch.as_tensor(action, dtype=torch.float32, device=self.device)


class ActionTraceLogger:
    def __init__(self, csv_path: Path, joint_names: list[str]) -> None:
        self.csv_path = csv_path
        self.joint_names = joint_names
        self.sim_trace_step_ix = 0
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=ACTION_TRACE_CSV_COLUMNS)
        self._writer.writeheader()
        self.ee_fk = make_ee_fk(joint_names)
        self.abs_errors: list[float] = []
        self.last_step_errors: list[float] = []

    def close(self) -> None:
        self._file.close()

    def write_substep(
        self,
        *,
        episode_ix: int,
        demo_key: str,
        policy_step_ix: int,
        sim_substep_ix: int,
        sim_substeps_for_policy: int,
        terminated: bool,
        truncated: bool,
        dataset_action_raw: np.ndarray,
        prepared_action_abs: np.ndarray,
        interpolated_action_abs: np.ndarray,
        eval_sim_joint_pos_after: np.ndarray,
        record_obs_joint_pos: np.ndarray | None,
        record_state_joint_pos: np.ndarray | None,
    ) -> None:
        width = min(
            len(self.joint_names),
            int(dataset_action_raw.shape[0]),
            int(prepared_action_abs.shape[0]),
            int(interpolated_action_abs.shape[0]),
            int(eval_sim_joint_pos_after.shape[0]),
        )
        is_policy_action_sample = sim_substep_ix == sim_substeps_for_policy - 1
        step_errors: list[float] = []
        ee_values = (
            empty_ee_trace_values()
            if terminated or truncated
            else ee_trace_values(self.ee_fk, interpolated_action_abs[:width], eval_sim_joint_pos_after[:width])
        )
        for joint_ix in range(width):
            record_state_value = np.nan
            abs_error = np.nan
            if record_state_joint_pos is not None and joint_ix < record_state_joint_pos.shape[0]:
                record_state_value = float(record_state_joint_pos[joint_ix])
                abs_error = abs(float(eval_sim_joint_pos_after[joint_ix]) - record_state_value)
                self.abs_errors.append(abs_error)
                step_errors.append(abs_error)

            record_obs_value = np.nan
            if record_obs_joint_pos is not None and joint_ix < record_obs_joint_pos.shape[0]:
                record_obs_value = float(record_obs_joint_pos[joint_ix])

            self._writer.writerow(
                {
                    "episode_ix": episode_ix,
                    "demo_key": demo_key,
                    "policy_step_ix": policy_step_ix,
                    "sim_trace_step_ix": self.sim_trace_step_ix,
                    "sim_substep_ix": sim_substep_ix,
                    "sim_substeps_for_policy": sim_substeps_for_policy,
                    "is_policy_action_sample": int(is_policy_action_sample),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "joint_ix": joint_ix,
                    "joint_name": self.joint_names[joint_ix],
                    "dataset_action_raw": float(dataset_action_raw[joint_ix]),
                    "prepared_action_abs": float(prepared_action_abs[joint_ix]),
                    "interpolated_action_abs": float(interpolated_action_abs[joint_ix]),
                    "eval_sim_joint_pos_after": float(eval_sim_joint_pos_after[joint_ix]),
                    "record_obs_joint_pos": record_obs_value,
                    "record_state_joint_pos": record_state_value,
                    "abs_eval_vs_record_state": abs_error,
                    **ee_values,
                }
            )
        self.sim_trace_step_ix += 1
        if step_errors:
            self.last_step_errors = step_errors

    def flush_episode_errors(self) -> dict[str, float | str]:
        if not self.abs_errors:
            return {
                "max_abs_eval_vs_record_state": "",
                "mean_abs_eval_vs_record_state": "",
                "final_abs_eval_vs_record_state": "",
                "final_max_abs_eval_vs_record_state": "",
                "final_mean_abs_eval_vs_record_state": "",
            }
        errors = np.asarray(self.abs_errors, dtype=np.float64)
        final_errors = np.asarray(self.last_step_errors, dtype=np.float64)
        final_max = float(np.nanmax(final_errors)) if final_errors.size else ""
        final_mean = float(np.nanmean(final_errors)) if final_errors.size else ""
        summary = {
            "max_abs_eval_vs_record_state": float(np.nanmax(errors)),
            "mean_abs_eval_vs_record_state": float(np.nanmean(errors)),
            "final_abs_eval_vs_record_state": final_max,
            "final_max_abs_eval_vs_record_state": final_max,
            "final_mean_abs_eval_vs_record_state": final_mean,
        }
        self.abs_errors.clear()
        self.last_step_errors.clear()
        return summary


def make_isaaclab_arena_cfg(args: argparse.Namespace, episode_length: int) -> IsaaclabArenaEnv:
    kwargs = {
        "object_name": args.object_name,
        "scene_name": args.scene_name,
        "object_center": True,
        "mobile_base_relative": bool(args.mobile_base_relative and args.base_relative_reference == "eval_state"),
        "traj_seed": args.traj_seed,
        "interpolated": args.interpolated,
        "interpolation_type": args.interpolation_type,
        "openness_threshold": args.openness_threshold,
        "handle_distance_threshold": args.handle_distance_threshold,
        "proximity_threshold": args.proximity_threshold,
        "proximity_window_steps": args.proximity_window_steps,
        "proximity_required_steps": args.proximity_required_steps,
        "disable_fingertip_proximity": args.disable_fingertip_proximity,
        "debug_visualize_handle": args.debug_visualize_handle,
        "debug_record_handle_diagnostics": args.debug_record_handle_diagnostics,
        "debug_marker_scale": args.debug_marker_scale,
    }
    if args.robot_object_static_friction is not None:
        kwargs["robot_object_static_friction"] = args.robot_object_static_friction
    if args.robot_object_dynamic_friction is not None:
        kwargs["robot_object_dynamic_friction"] = args.robot_object_dynamic_friction
    if args.traj_file:
        kwargs["traj_file"] = str(Path(args.traj_file).resolve())
    if args.decimation is not None:
        kwargs["decimation"] = args.decimation

    return IsaaclabArenaEnv(
        hub_path=str(ISAAC_ENV_ROOT),
        episode_length=episode_length,
        environment="summit_franka_open_door_eval",
        headless=bool(args.headless),
        enable_cameras=True,
        state_keys="joint_pos",
        camera_keys="ego_topdown_rgb,ego_wrist_rgb,fix_local_rgb",
        state_dim=12,
        action_dim=12,
        camera_height=240,
        camera_width=320,
        device=str(args.env_device),
        kwargs=kwargs,
    )


def apply_decimation_override(env: Any, decimation: int | None) -> None:
    if decimation is None:
        return
    raw_env = getattr(env, "_env", None)
    raw_cfg = getattr(raw_env, "cfg", None)
    if raw_cfg is None:
        logging.warning("Could not apply decimation override: raw env cfg is unavailable.")
        return
    raw_cfg.decimation = int(decimation)
    sim_cfg = getattr(raw_cfg, "sim", None)
    if sim_cfg is not None and hasattr(sim_cfg, "render_interval"):
        sim_cfg.render_interval = int(decimation)


def sync_episode_budget(env: Any, total_steps: int) -> None:
    if hasattr(env, "_episode_length"):
        env._episode_length = int(total_steps)

    raw_env = getattr(env, "_env", None)
    raw_cfg = getattr(raw_env, "cfg", None)
    if raw_env is None or raw_cfg is None or not hasattr(raw_cfg, "episode_length_s"):
        return

    step_dt = float(getattr(raw_env, "step_dt", 0.02) or 0.02)
    raw_cfg.episode_length_s = int(total_steps) * step_dt


def disable_termination_terms(env: Any, term_names: tuple[str, ...] = ("success",)) -> list[str]:
    """Neutralize runtime termination terms from an already-created IsaacLab env.

    This is intentionally scoped to record-dataset replay. The recording script
    disables success termination before creating the env so trajectories are not
    cut short by the task's stability detector; here the env already exists via
    the LeRobot eval path, so we patch the manager in-place for equivalent replay.
    The term names are kept so IsaacLab recorder terms that expect a ``success``
    termination can still export/reset cleanly.
    """

    raw_env = getattr(env, "_env", None)
    manager = getattr(raw_env, "termination_manager", None)
    if raw_env is None or manager is None:
        return []

    existing = list(getattr(manager, "_term_names", []))
    remove = {name for name in term_names if name in existing}
    if not remove:
        return []

    for name in remove:
        term_ix = existing.index(name)
        term_cfg = manager._term_cfgs[term_ix]

        def _never_terminate(patched_env: Any, **_: Any) -> torch.Tensor:
            return torch.zeros(patched_env.num_envs, dtype=torch.bool, device=patched_env.device)

        term_cfg.func = _never_terminate
        term_cfg.params = {}

    logging.info("Disabled runtime termination term(s) for record replay: %s", ", ".join(sorted(remove)))
    return sorted(remove)


def recompute_observations(env: Any) -> dict[str, Any] | None:
    raw_env = getattr(env, "_env", None)
    if raw_env is None or not hasattr(raw_env, "observation_manager"):
        return None
    obs = raw_env.observation_manager.compute()
    raw_env.obs_buf = obs
    return obs


def _scene_articulation_for_object(env: Any) -> Any | None:
    raw_env = getattr(env, "_env", None)
    scene = getattr(raw_env, "scene", None)
    if scene is None:
        return None
    for key in scene.keys():
        if key == "robot":
            continue
        entity = scene[key]
        if hasattr(entity, "write_joint_state_to_sim") and hasattr(entity, "num_joints"):
            return entity
    return None


def set_hdf5_initial_state(env: Any, episode: RecordEpisode, init_steps: int, render: bool = True) -> dict[str, Any] | None:
    raw_env = getattr(env, "_env", None)
    scene = getattr(raw_env, "scene", None)
    if raw_env is None or scene is None:
        return None

    robot = scene["robot"] if "robot" in scene.keys() else None
    object_entity = _scene_articulation_for_object(env)
    dev = torch.device(getattr(raw_env, "device", "cpu"))

    robot_pos_np = episode.obs_joint_pos[0]
    object_pos_np = episode.object_initial_joint_pos[0] if episode.object_initial_joint_pos is not None else None
    object_root_pose_np = episode.object_initial_root_pose[0] if episode.object_initial_root_pose is not None else None
    object_root_velocity_np = (
        episode.object_initial_root_velocity[0] if episode.object_initial_root_velocity is not None else None
    )

    steps = max(int(init_steps), 1)
    for _ in range(steps):
        if robot is not None:
            n_joints = min(robot_pos_np.shape[0], robot.num_joints)
            joint_pos = torch.as_tensor(robot_pos_np[:n_joints], dtype=torch.float32, device=dev).unsqueeze(0)
            joint_vel = torch.zeros_like(joint_pos)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            if hasattr(robot, "set_joint_position_target"):
                robot.set_joint_position_target(joint_pos)
            if hasattr(robot, "set_joint_velocity_target"):
                robot.set_joint_velocity_target(joint_vel)

        if object_entity is not None:
            if object_root_pose_np is not None and hasattr(object_entity, "write_root_pose_to_sim"):
                root_pose = torch.as_tensor(object_root_pose_np, dtype=torch.float32, device=dev).unsqueeze(0)
                object_entity.write_root_pose_to_sim(root_pose)
            if object_root_velocity_np is not None and hasattr(object_entity, "write_root_velocity_to_sim"):
                root_velocity = torch.as_tensor(object_root_velocity_np, dtype=torch.float32, device=dev).unsqueeze(0)
                object_entity.write_root_velocity_to_sim(root_velocity)
            if object_pos_np is not None:
                n_joints = min(object_pos_np.shape[0], object_entity.num_joints)
                joint_pos = torch.as_tensor(object_pos_np[:n_joints], dtype=torch.float32, device=dev).unsqueeze(0)
                joint_vel = torch.zeros_like(joint_pos)
                object_entity.write_joint_state_to_sim(joint_pos, joint_vel)
                if hasattr(object_entity, "set_joint_position_target"):
                    object_entity.set_joint_position_target(joint_pos)
                if hasattr(object_entity, "set_joint_velocity_target"):
                    object_entity.set_joint_velocity_target(joint_vel)

        if hasattr(scene, "write_data_to_sim"):
            scene.write_data_to_sim()
        sim = getattr(raw_env, "sim", None)
        if sim is not None:
            if hasattr(sim, "step"):
                sim.step(render=render)
            elif render and hasattr(sim, "render"):
                sim.render()
        if hasattr(scene, "update"):
            scene.update(getattr(raw_env, "physics_dt", 0.0))

    return recompute_observations(env)


def extract_sim_joint_pos(env: Any, fallback_obs: dict[str, Any] | None = None) -> np.ndarray:
    raw_env = getattr(env, "_env", None)
    scene = getattr(raw_env, "scene", None)
    if raw_env is not None and scene is not None and "robot" in scene.keys():
        robot = scene["robot"]
        joint_pos = robot.data.joint_pos
        if hasattr(robot.data, "joint_names"):
            name_to_ix = {name: ix for ix, name in enumerate(robot.data.joint_names)}
            if all(name in name_to_ix for name in SUMMIT_FRANKA_ACTION_JOINT_NAMES):
                indices = [name_to_ix[name] for name in SUMMIT_FRANKA_ACTION_JOINT_NAMES]
                joint_pos = joint_pos[:, indices]
        joint_pos_np = joint_pos.detach().cpu().numpy()
        return joint_pos_np[0].astype(np.float32) if joint_pos_np.ndim == 2 else joint_pos_np.astype(np.float32)
    if fallback_obs is None:
        raise RuntimeError("Could not read robot joint state from simulator.")
    joint_pos = fallback_obs["policy"]["joint_pos"]
    if isinstance(joint_pos, torch.Tensor):
        joint_pos = joint_pos.detach().cpu().numpy()
    return joint_pos[0].astype(np.float32) if joint_pos.ndim == 2 else joint_pos.astype(np.float32)


def render_env(env: Any) -> np.ndarray | None:
    try:
        frames = env.call("render")
    except Exception:
        return None
    if not frames:
        return None
    return frames[0]


def append_per_episode_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_EPISODE_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in PER_EPISODE_CSV_COLUMNS})


def first_scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    return value


def evaluate_current_success(env: Any, args: argparse.Namespace) -> dict[str, Any]:
    raw_env = getattr(env, "_env", None)
    openable_object = None
    try:
        arena_env = getattr(getattr(raw_env, "cfg", None), "isaaclab_arena_env", None)
        openable_object = getattr(getattr(arena_env, "task", None), "openable_object", None)
    except Exception:
        openable_object = None

    if raw_env is None or openable_object is None:
        return {
            "success": False,
            "final_door_open": False,
            "final_door_openness": np.nan,
            "final_engaged": False,
            "final_handle_distance": np.nan,
        }

    from isaaclab_arena.metrics.handle_proximity_rate import (
        compute_open_while_engaged,
        get_cached_handle_proximity_diagnostics,
    )

    compute_open_while_engaged(
        env=raw_env,
        openable_object=openable_object,
        proximity_threshold=args.proximity_threshold,
        proximity_window_steps=args.proximity_window_steps,
        proximity_required_steps=args.proximity_required_steps,
        use_fingertips=not args.disable_fingertip_proximity,
        openness_threshold=args.openness_threshold,
        debug_visualize_handle=args.debug_visualize_handle,
        debug_marker_scale=args.debug_marker_scale,
    )
    diagnostics = get_cached_handle_proximity_diagnostics(raw_env, openable_object) or {}

    def tensor_value(key: str, default: Any) -> Any:
        value = diagnostics.get(key)
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            return value.detach().cpu().flatten()[0].item()
        return first_scalar(value, default)

    final_door_open = bool(tensor_value("door_open", False))
    final_openness = float(tensor_value("openness", np.nan))
    final_handle_distance = float(tensor_value("handle_distance", np.nan))
    final_engaged = bool(np.isfinite(final_handle_distance) and final_handle_distance <= args.handle_distance_threshold)
    return {
        "success": bool(final_door_open and final_engaged),
        "final_door_open": final_door_open,
        "final_door_openness": final_openness,
        "final_engaged": final_engaged,
        "final_handle_distance": final_handle_distance,
    }


def prepare_policy_observation(
    env: Any,
    observation: dict[str, Any],
    env_preprocessor: Any,
    preprocessor: Any,
) -> dict[str, Any]:
    processed = preprocess_observation(observation)
    try:
        processed["task"] = list(env.call("task_description"))
    except (AttributeError, NotImplementedError):
        try:
            processed["task"] = list(env.call("task"))
        except (AttributeError, NotImplementedError):
            processed["task"] = [""] * env.num_envs
    processed = env_preprocessor(processed)
    return preprocessor(processed)


def run_episode(
    *,
    env: Any,
    policy: Hdf5ActionPolicy,
    episode: RecordEpisode,
    episode_ix: int,
    seed: int,
    max_steps: int,
    init_steps: int,
    args: argparse.Namespace,
    action_executor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    render_video: bool,
    stop_on_dataset_end: bool,
    action_trace_logger: ActionTraceLogger | None,
) -> dict[str, Any]:
    policy.set_episode(episode)
    action_executor.reset()
    observation, _ = env.reset(seed=seed)
    settled = set_hdf5_initial_state(env, episode, init_steps=init_steps, render=True)
    if settled is not None:
        observation = settled

    frames: list[np.ndarray] = []
    if render_video:
        frame = render_env(env)
        if frame is not None:
            frames.append(frame)

    done = False
    sim_steps = 0
    policy_steps = 0

    while not done and sim_steps < max_steps:
        if stop_on_dataset_end and not policy.has_action:
            break

        policy_obs = prepare_policy_observation(env, observation, env_preprocessor, preprocessor)
        with torch.inference_mode():
            action = policy.select_action(policy_obs)
        action = postprocessor(action)
        action_transition = env_postprocessor({ACTION: action})
        action_np = action_transition[ACTION].to("cpu").numpy()
        raw_dataset_action = action_np[0].astype(np.float32)

        step_results = action_executor.execute(action_np)

        for step_result in step_results:
            observation = step_result.obs
            done = bool(step_result.terminated[0] or step_result.truncated[0])
            sim_state_after = step_result.sim_state_after
            if sim_state_after is None:
                sim_state_after = extract_sim_joint_pos(env, observation)

            record_ix = sim_steps
            record_obs_joint_pos = episode.obs_joint_pos[record_ix] if record_ix < episode.obs_joint_pos.shape[0] else None
            record_state_joint_pos = None
            if episode.state_joint_pos is not None and record_ix < episode.state_joint_pos.shape[0]:
                record_state_joint_pos = episode.state_joint_pos[record_ix]

            if action_trace_logger is not None:
                action_trace_logger.write_substep(
                    episode_ix=episode_ix,
                    demo_key=episode.demo_key,
                    policy_step_ix=policy_steps,
                    sim_substep_ix=step_result.sim_substep_ix,
                    sim_substeps_for_policy=step_result.sim_substeps_for_policy,
                    terminated=bool(step_result.terminated[0]),
                    truncated=bool(step_result.truncated[0]),
                    dataset_action_raw=raw_dataset_action,
                    prepared_action_abs=step_result.policy_action,
                    interpolated_action_abs=step_result.interpolated_action,
                    eval_sim_joint_pos_after=sim_state_after,
                    record_obs_joint_pos=record_obs_joint_pos,
                    record_state_joint_pos=record_state_joint_pos,
                )

            sim_steps += 1
            if render_video:
                frame = render_env(env)
                if frame is not None:
                    frames.append(frame)
            if done or sim_steps >= max_steps:
                break

        policy_steps += 1

    metrics = evaluate_current_success(env, args=args)
    trace_summary = action_trace_logger.flush_episode_errors() if action_trace_logger is not None else {}
    return {
        **metrics,
        "sim_steps": sim_steps,
        "policy_steps": policy_steps,
        "frames": frames,
        "trace_summary": trace_summary,
    }


def main() -> None:
    init_logging()
    args = parse_args()

    if args.interpolated < 1:
        raise ValueError("--interpolated must be >= 1.")
    if args.init_steps < 1:
        raise ValueError("--init_steps must be >= 1.")
    if args.decimation is not None and args.decimation < 1:
        raise ValueError("--decimation must be >= 1.")

    dataset_file = Path(args.dataset_file)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Record HDF5 not found: {dataset_file}")

    record_metadata = load_record_metadata(dataset_file)
    hdf5_decimation = record_metadata.get("sim_args", {}).get("decimation")
    decimation_source = "cli"
    if args.decimation is None and hdf5_decimation is not None:
        args.decimation = int(hdf5_decimation)
        decimation_source = "hdf5_env_args"
        logging.info("Using HDF5 record decimation=%s from data.attrs['env_args'].", args.decimation)
    elif args.decimation is None:
        decimation_source = "env_default"
    if args.decimation is not None and args.decimation < 1:
        raise ValueError("--decimation must be >= 1.")

    episodes = load_record_episodes(dataset_file, args.action_key)
    selected = episodes[args.start_demo :]
    if not selected:
        raise ValueError(f"No HDF5 demos available from --start_demo={args.start_demo}.")
    n_episodes = len(selected) if args.n_episodes is None else min(args.n_episodes, len(selected))
    selected = selected[:n_episodes]

    max_dataset_steps = max(ep.n_steps for ep in selected)
    interpolation_factor = max(int(args.interpolated), 1) if args.interpolation_type != "none" else 1
    default_max_steps = max_dataset_steps * interpolation_factor
    max_steps = int(args.max_steps or default_max_steps)
    env_episode_length = max_steps + 1 if args.stop_on_dataset_end else max_steps

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_episode_results.csv"
    eval_info_path = output_dir / "eval_info.json"
    for path in (csv_path, eval_info_path):
        if path.exists():
            path.unlink()

    set_seed(args.seed)

    env_cfg = make_isaaclab_arena_cfg(args, episode_length=env_episode_length)
    env_map = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=True)
    env = env_map["summit_franka_open_door_eval"][0]
    apply_decimation_override(env, args.decimation)
    sync_episode_budget(env, env_episode_length)
    disabled_termination_terms = (
        disable_termination_terms(env, ("success",)) if args.disable_success_termination else []
    )

    if hasattr(env, "make_action_executor"):
        action_executor = env.make_action_executor(
            interpolation_factor=args.interpolated,
            interpolation_type=args.interpolation_type,
            state_fn=lambda step_obs: extract_sim_joint_pos(env, step_obs),
        )
    else:
        from isaaclab_arena.utils.action_execution import InterpolatedActionExecutor

        action_executor = InterpolatedActionExecutor(
            env,
            interpolation_factor=args.interpolated,
            interpolation_type=args.interpolation_type,
            state_fn=lambda step_obs: extract_sim_joint_pos(env, step_obs),
        )

    action_trace_path = Path(args.action_trace_csv) if args.action_trace_csv else output_dir / "action_trace_joint_states.csv"
    action_trace_logger = (
        ActionTraceLogger(action_trace_path, list(SUMMIT_FRANKA_ACTION_JOINT_NAMES))
        if args.debug_action_trace
        else None
    )

    policy_device = args.policy_device
    if policy_device == "cuda" and not torch.cuda.is_available():
        policy_device = "cpu"
    policy = Hdf5ActionPolicy(device=policy_device, base_relative_reference=args.base_relative_reference)
    policy.eval()
    env_preprocessor = IdentityProcessor()
    env_postprocessor = IdentityProcessor()
    preprocessor = IdentityProcessor()
    postprocessor = IdentityProcessor()

    videos_dir = output_dir / "videos" / "summit_franka_open_door_eval_0"
    per_episode: list[dict[str, Any]] = []
    successes: list[bool] = []
    video_paths: list[str] = []
    start = time.time()

    try:
        for episode_ix, episode in enumerate(selected):
            episode_seed = args.seed + episode_ix
            render_video = episode_ix < args.max_episodes_rendered
            result = run_episode(
                env=env,
                policy=policy,
                episode=episode,
                episode_ix=episode_ix,
                seed=episode_seed,
                max_steps=max_steps,
                init_steps=args.init_steps,
                args=args,
                action_executor=action_executor,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                render_video=render_video,
                stop_on_dataset_end=args.stop_on_dataset_end,
                action_trace_logger=action_trace_logger,
            )

            video_path = ""
            if render_video and result["frames"]:
                videos_dir.mkdir(parents=True, exist_ok=True)
                video_file = videos_dir / f"eval_episode_{episode_ix}.mp4"
                fps = int(getattr(env, "metadata", {}).get("render_fps", 50))
                write_video(str(video_file), np.stack(result["frames"], axis=0), fps)
                video_path = str(video_file)
                video_paths.append(video_path)

            row = {
                "episode_ix": episode_ix,
                "demo_key": episode.demo_key,
                "seed": episode_seed,
                "record_success": "" if episode.record_success is None else bool(episode.record_success),
                "success": bool(result["success"]),
                "final_door_open": result["final_door_open"],
                "final_door_openness": result["final_door_openness"],
                "final_engaged": result["final_engaged"],
                "final_handle_distance": result["final_handle_distance"],
                "steps": f"{result['sim_steps']}/{max_steps}",
                "policy_steps": result["policy_steps"],
                "dataset_steps": episode.n_steps,
                "video_path": video_path,
                **result["trace_summary"],
            }
            append_per_episode_csv_row(csv_path, row)
            per_episode.append(
                {
                    "episode_ix": episode_ix,
                    "demo_key": episode.demo_key,
                    "record_success": row["record_success"],
                    "success": row["success"],
                    "final_door_openness": row["final_door_openness"],
                    "final_handle_distance": row["final_handle_distance"],
                    "max_abs_eval_vs_record_state": row.get("max_abs_eval_vs_record_state", ""),
                    "mean_abs_eval_vs_record_state": row.get("mean_abs_eval_vs_record_state", ""),
                    "final_abs_eval_vs_record_state": row.get("final_abs_eval_vs_record_state", ""),
                    "final_max_abs_eval_vs_record_state": row.get("final_max_abs_eval_vs_record_state", ""),
                    "final_mean_abs_eval_vs_record_state": row.get("final_mean_abs_eval_vs_record_state", ""),
                    "seed": episode_seed,
                }
            )
            successes.append(bool(row["success"]))
            logging.info(
                "episode=%s demo=%s success=%s record_success=%s steps=%s max_joint_err=%s",
                episode_ix,
                episode.demo_key,
                row["success"],
                row["record_success"],
                row["steps"],
                row.get("max_abs_eval_vs_record_state", ""),
            )

    finally:
        if action_trace_logger is not None:
            action_trace_logger.close()

    elapsed = time.time() - start
    info = {
        "per_episode": per_episode,
        "aggregated": {
            "pc_success": float(np.mean(successes) * 100.0) if successes else 0.0,
            "eval_s": elapsed,
            "eval_ep_s": elapsed / max(len(successes), 1),
        },
        "video_paths": video_paths,
        "record_dataset_eval": {
            "dataset_file": str(dataset_file.resolve()),
            "action_key": args.action_key,
            "start_demo": args.start_demo,
            "n_episodes": n_episodes,
            "max_steps": max_steps,
            "env_episode_length": env_episode_length,
            "stop_on_dataset_end": bool(args.stop_on_dataset_end),
            "mobile_base_relative": bool(args.mobile_base_relative),
            "env_mobile_base_relative": bool(args.mobile_base_relative and args.base_relative_reference == "eval_state"),
            "base_relative_reference": args.base_relative_reference,
            "disable_success_termination": bool(args.disable_success_termination),
            "disabled_termination_terms": disabled_termination_terms,
            "decimation": args.decimation,
            "decimation_source": decimation_source,
            "record_env_args": record_metadata.get("env_args", {}),
            "init_steps": args.init_steps,
            "interpolated": args.interpolated,
            "interpolation_type": args.interpolation_type,
            "traj_file": str(Path(args.traj_file).resolve()) if args.traj_file else "",
            "traj_seed": args.traj_seed,
            "action_trace_csv": str(action_trace_path) if args.debug_action_trace else "",
        },
    }
    eval_info_path.write_text(json.dumps(info, indent=2))
    print(json.dumps(info["aggregated"], indent=2), flush=True)
    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
