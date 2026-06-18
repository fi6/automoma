#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ISAAC_ENV_ROOT = REPO_ROOT / "third_party" / "IsaacLab-Arena" / "isaaclab-arena-envs"
LEROBOT_SRC_ROOT = REPO_ROOT / "third_party" / "lerobot" / "src"

for path in (str(ISAAC_ENV_ROOT), str(LEROBOT_SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.configs import IsaaclabArenaEnv
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


PER_EPISODE_CSV_COLUMNS = [
    "episode_ix",
    "seed",
    "success",
    "final_door_open",
    "final_door_openness",
    "final_engaged",
    "final_handle_distance",
    "steps",
    "policy_steps",
    "video_path",
]

ACTION_TRACE_CSV_COLUMNS = [
    "episode_ix",
    "policy_step_ix",
    "sim_trace_step_ix",
    "sim_substep_ix",
    "sim_substeps_for_policy",
    "is_policy_action_sample",
    "terminated",
    "truncated",
    "joint_ix",
    "joint_name",
    "raw_policy_action",
    "policy_action_abs",
    "interpolated_action_abs",
    "sim_joint_pos_before",
    "sim_joint_pos_after",
    "raw_policy_step_delta_abs",
    "policy_action_step_delta_abs",
    "sim_joint_step_delta_abs",
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
        description="Evaluate a LeRobot policy on IsaacLab-Arena with AutoMoMa record/eval timing alignment."
    )
    parser.add_argument("--policy.path", "--policy_path", dest="policy_path", required=True)
    parser.add_argument("--policy.device", "--policy_device", dest="policy_device", default="cuda")
    parser.add_argument("--env.device", "--env_device", dest="env_device", default="cuda:0")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--object_name", required=True)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--traj_file", required=True)
    parser.add_argument("--traj_seed", type=int, default=42)
    parser.add_argument("--traj_selection_mode", choices=("random", "sequential"), default="random")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--eval.n_episodes", "--n_episodes", dest="n_episodes", type=int, default=50)
    parser.add_argument("--env.episode_length", "--max_steps", dest="max_steps", type=int, default=300)
    parser.add_argument("--max_episodes_rendered", type=int, default=10)
    parser.add_argument("--headless", type=str2bool, default=True)
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument("--init_steps", type=int, default=1)
    parser.add_argument("--interpolated", type=int, default=1)
    parser.add_argument("--interpolation_type", default="linear")
    parser.add_argument("--mobile_base_relative", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--openness_threshold", type=float, default=0.3)
    parser.add_argument("--proximity_threshold", type=float, default=0.12)
    parser.add_argument("--proximity_window_steps", type=int, default=8)
    parser.add_argument("--proximity_required_steps", type=int, default=5)
    parser.add_argument("--disable_fingertip_proximity", type=str2bool, default=False)
    parser.add_argument("--debug_visualize_handle", type=str2bool, default=False)
    parser.add_argument("--debug_record_handle_diagnostics", type=str2bool, default=False)
    parser.add_argument("--debug_marker_scale", type=float, default=1.0)
    parser.add_argument("--robot_object_static_friction", type=float, default=None)
    parser.add_argument("--robot_object_dynamic_friction", type=float, default=None)
    parser.add_argument("--debug_action_trace", type=str2bool, default=False)
    parser.add_argument("--action_trace_csv", default=None)
    return parser.parse_args()


class ActionTraceLogger:
    def __init__(self, csv_path: Path, joint_names: list[str]) -> None:
        self.csv_path = csv_path
        self.joint_names = joint_names
        self.sim_trace_step_ix = 0
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=ACTION_TRACE_CSV_COLUMNS)
        self._writer.writeheader()
        self._last_raw_policy_action: dict[int, np.ndarray] = {}
        self._last_policy_action: dict[int, np.ndarray] = {}
        self._last_sim_joint_pos_after: dict[int, np.ndarray] = {}
        self.max_raw_policy_step_delta_abs = 0.0
        self.max_policy_action_step_delta_abs = 0.0
        self.max_sim_joint_step_delta_abs = 0.0
        self.max_raw_policy_base_step_delta_abs = 0.0
        self.max_policy_action_base_step_delta_abs = 0.0
        self.max_sim_base_step_delta_abs = 0.0

    def close(self) -> None:
        self._file.close()

    def write_substep(
        self,
        *,
        episode_ix: int,
        policy_step_ix: int,
        sim_substep_ix: int,
        sim_substeps_for_policy: int,
        terminated: bool,
        truncated: bool,
        raw_policy_action: np.ndarray,
        policy_action_abs: np.ndarray,
        interpolated_action_abs: np.ndarray,
        sim_joint_pos_before: np.ndarray,
        sim_joint_pos_after: np.ndarray,
    ) -> None:
        width = min(
            len(self.joint_names),
            int(raw_policy_action.shape[0]),
            int(policy_action_abs.shape[0]),
            int(interpolated_action_abs.shape[0]),
            int(sim_joint_pos_before.shape[0]),
            int(sim_joint_pos_after.shape[0]),
        )
        is_policy_action_sample = sim_substep_ix == sim_substeps_for_policy - 1
        prev_raw = self._last_raw_policy_action.get(episode_ix)
        prev_policy = self._last_policy_action.get(episode_ix)
        prev_sim = self._last_sim_joint_pos_after.get(episode_ix)
        raw_delta = (
            np.abs(raw_policy_action[:width] - prev_raw[:width])
            if prev_raw is not None and prev_raw.shape[0] >= width
            else np.full(width, np.nan, dtype=np.float32)
        )
        policy_delta = (
            np.abs(policy_action_abs[:width] - prev_policy[:width])
            if prev_policy is not None and prev_policy.shape[0] >= width
            else np.full(width, np.nan, dtype=np.float32)
        )
        sim_delta = (
            np.abs(sim_joint_pos_after[:width] - prev_sim[:width])
            if prev_sim is not None and prev_sim.shape[0] >= width
            else np.full(width, np.nan, dtype=np.float32)
        )
        if not (terminated or truncated):
            self._update_delta_summary(raw_delta, policy_delta, sim_delta)
        for joint_ix in range(width):
            self._writer.writerow(
                {
                    "episode_ix": episode_ix,
                    "policy_step_ix": policy_step_ix,
                    "sim_trace_step_ix": self.sim_trace_step_ix,
                    "sim_substep_ix": sim_substep_ix,
                    "sim_substeps_for_policy": sim_substeps_for_policy,
                    "is_policy_action_sample": int(is_policy_action_sample),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "joint_ix": joint_ix,
                    "joint_name": self.joint_names[joint_ix],
                    "raw_policy_action": float(raw_policy_action[joint_ix]),
                    "policy_action_abs": float(policy_action_abs[joint_ix]),
                    "interpolated_action_abs": float(interpolated_action_abs[joint_ix]),
                    "sim_joint_pos_before": float(sim_joint_pos_before[joint_ix]),
                    "sim_joint_pos_after": float(sim_joint_pos_after[joint_ix]),
                    "raw_policy_step_delta_abs": "" if np.isnan(raw_delta[joint_ix]) else float(raw_delta[joint_ix]),
                    "policy_action_step_delta_abs": ""
                    if np.isnan(policy_delta[joint_ix])
                    else float(policy_delta[joint_ix]),
                    "sim_joint_step_delta_abs": "" if np.isnan(sim_delta[joint_ix]) else float(sim_delta[joint_ix]),
                }
            )
        self._last_raw_policy_action[episode_ix] = raw_policy_action.copy()
        self._last_policy_action[episode_ix] = policy_action_abs.copy()
        self._last_sim_joint_pos_after[episode_ix] = sim_joint_pos_after.copy()
        self.sim_trace_step_ix += 1

    def _update_delta_summary(
        self,
        raw_delta: np.ndarray,
        policy_delta: np.ndarray,
        sim_delta: np.ndarray,
    ) -> None:
        for attr, values in (
            ("max_raw_policy_step_delta_abs", raw_delta),
            ("max_policy_action_step_delta_abs", policy_delta),
            ("max_sim_joint_step_delta_abs", sim_delta),
        ):
            finite = values[np.isfinite(values)]
            if finite.size:
                setattr(self, attr, max(float(getattr(self, attr)), float(np.max(finite))))
        base_width = min(3, raw_delta.shape[0], policy_delta.shape[0], sim_delta.shape[0])
        if base_width <= 0:
            return
        for attr, values in (
            ("max_raw_policy_base_step_delta_abs", raw_delta[:base_width]),
            ("max_policy_action_base_step_delta_abs", policy_delta[:base_width]),
            ("max_sim_base_step_delta_abs", sim_delta[:base_width]),
        ):
            finite = values[np.isfinite(values)]
            if finite.size:
                setattr(self, attr, max(float(getattr(self, attr)), float(np.max(finite))))

    def summary(self) -> dict[str, float | str]:
        return {
            "csv_path": str(self.csv_path),
            "max_raw_policy_step_delta_abs": self.max_raw_policy_step_delta_abs,
            "max_policy_action_step_delta_abs": self.max_policy_action_step_delta_abs,
            "max_sim_joint_step_delta_abs": self.max_sim_joint_step_delta_abs,
            "max_raw_policy_base_step_delta_abs": self.max_raw_policy_base_step_delta_abs,
            "max_policy_action_base_step_delta_abs": self.max_policy_action_base_step_delta_abs,
            "max_sim_base_step_delta_abs": self.max_sim_base_step_delta_abs,
        }


def make_isaaclab_arena_cfg(args: argparse.Namespace) -> IsaaclabArenaEnv:
    kwargs = {
        "object_name": args.object_name,
        "scene_name": args.scene_name,
        "object_center": True,
        "mobile_base_relative": args.mobile_base_relative,
        "traj_file": str(Path(args.traj_file).resolve()),
        "traj_seed": args.traj_seed,
        "traj_selection_mode": args.traj_selection_mode,
        "interpolated": args.interpolated,
        "interpolation_type": args.interpolation_type,
        "openness_threshold": args.openness_threshold,
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
    if args.decimation is not None:
        kwargs["decimation"] = args.decimation

    return IsaaclabArenaEnv(
        hub_path=str(ISAAC_ENV_ROOT),
        episode_length=args.max_steps,
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
    logging.info("Applied eval decimation override: decimation=%s step_dt=%s", decimation, getattr(raw_env, "step_dt", None))


def sync_episode_budget(env: Any, max_steps: int) -> None:
    if hasattr(env, "_episode_length"):
        env._episode_length = int(max_steps)

    raw_env = getattr(env, "_env", None)
    raw_cfg = getattr(raw_env, "cfg", None)
    if raw_env is None or raw_cfg is None or not hasattr(raw_cfg, "episode_length_s"):
        return

    step_dt = float(getattr(raw_env, "step_dt", 0.02) or 0.02)
    raw_cfg.episode_length_s = int(max_steps) * step_dt
    logging.info(
        "Synced eval budget: wrapper_steps=%s raw_episode_length_s=%.6f raw_step_dt=%.6f",
        max_steps,
        raw_cfg.episode_length_s,
        step_dt,
    )


def recompute_observations(env: Any) -> dict[str, Any] | None:
    raw_env = getattr(env, "_env", None)
    if raw_env is None or not hasattr(raw_env, "observation_manager"):
        return None
    obs = raw_env.observation_manager.compute()
    raw_env.obs_buf = obs
    return obs


def settle_initial_state(env: Any, init_steps: int, render: bool = True) -> dict[str, Any] | None:
    if init_steps <= 0:
        return None
    raw_env = getattr(env, "_env", None)
    if raw_env is None or not hasattr(raw_env, "scene"):
        return None

    scene = raw_env.scene
    robot = scene["robot"] if "robot" in scene.keys() else None
    object_entity = None
    for key in scene.keys():
        if key == "robot":
            continue
        entity = scene[key]
        if hasattr(entity, "write_joint_state_to_sim") and hasattr(entity, "num_joints"):
            object_entity = entity
            break

    for _ in range(init_steps):
        if robot is not None:
            joint_pos = robot.data.joint_pos.clone()
            joint_vel = torch.zeros_like(joint_pos)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.set_joint_position_target(joint_pos)
            robot.set_joint_velocity_target(joint_vel)
        if object_entity is not None:
            joint_pos = object_entity.data.joint_pos.clone()
            joint_vel = torch.zeros_like(joint_pos)
            object_entity.write_joint_state_to_sim(joint_pos, joint_vel)
            if hasattr(object_entity, "set_joint_position_target"):
                object_entity.set_joint_position_target(joint_pos)
            if hasattr(object_entity, "set_joint_velocity_target"):
                object_entity.set_joint_velocity_target(joint_vel)

        scene.write_data_to_sim()
        raw_env.sim.step(render=render)
        scene.update(raw_env.physics_dt)

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


def success_from_final_info(final_info: dict[str, Any]) -> bool:
    if "is_success" in final_info:
        return bool(first_scalar(final_info["is_success"], False))
    return bool(first_scalar(final_info.get("final_door_open"), False)) and bool(
        first_scalar(final_info.get("final_engaged"), False)
    )


def build_policy_and_processors(args: argparse.Namespace, env_cfg: IsaaclabArenaEnv):
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = Path(args.policy_path)
    policy_cfg.device = args.policy_device

    rename_map = {
        "observation.images.ego_topdown_rgb": "observation.images.ego_topdown",
        "observation.images.ego_wrist_rgb": "observation.images.ego_wrist",
        "observation.images.fix_local_rgb": "observation.images.fix_local",
    }

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    return policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor


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
    episode_ix: int,
    env: Any,
    policy: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    seed: int,
    max_steps: int,
    init_steps: int,
    action_executor: Any,
    action_trace_logger: ActionTraceLogger | None,
    render_video: bool,
) -> dict[str, Any]:
    policy.reset()
    action_executor.reset()
    observation, _ = env.reset(seed=seed)
    settled = settle_initial_state(env, init_steps, render=True)
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
    latest_final_info: dict[str, Any] = {}

    while not done and sim_steps < max_steps:
        policy_obs = prepare_policy_observation(env, observation, env_preprocessor, preprocessor)
        with torch.inference_mode():
            action = policy.select_action(policy_obs)
        action = postprocessor(action)
        action_transition = env_postprocessor({ACTION: action})
        action_np = action_transition[ACTION].to("cpu").numpy()
        raw_policy_action = action_np[0].astype(np.float32) if action_np.ndim == 2 else action_np.astype(np.float32)
        sim_state_before = extract_sim_joint_pos(env, observation) if action_trace_logger is not None else None

        step_results = action_executor.execute(action_np)
        policy_steps += 1

        for step_result in step_results:
            observation = step_result.obs
            sim_steps += 1
            latest_final_info = step_result.info.get("final_info", latest_final_info)
            done = bool(step_result.terminated[0] or step_result.truncated[0])
            if action_trace_logger is not None:
                sim_state_after = step_result.sim_state_after
                if sim_state_after is None:
                    sim_state_after = extract_sim_joint_pos(env, observation)
                if sim_state_before is None:
                    sim_state_before = sim_state_after
                action_trace_logger.write_substep(
                    episode_ix=episode_ix,
                    policy_step_ix=policy_steps - 1,
                    sim_substep_ix=step_result.sim_substep_ix,
                    sim_substeps_for_policy=step_result.sim_substeps_for_policy,
                    terminated=bool(step_result.terminated[0]),
                    truncated=bool(step_result.truncated[0]),
                    raw_policy_action=raw_policy_action,
                    policy_action_abs=step_result.policy_action,
                    interpolated_action_abs=step_result.interpolated_action,
                    sim_joint_pos_before=sim_state_before,
                    sim_joint_pos_after=sim_state_after,
                )
                sim_state_before = sim_state_after
            if render_video:
                frame = render_env(env)
                if frame is not None:
                    frames.append(frame)
            if done or sim_steps >= max_steps:
                break

    return {
        "success": success_from_final_info(latest_final_info),
        "final_info": latest_final_info,
        "sim_steps": sim_steps,
        "policy_steps": policy_steps,
        "frames": frames,
    }


def main() -> None:
    init_logging()
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("--max_steps/--env.episode_length must be >= 1.")
    if args.n_episodes < 1:
        raise ValueError("--n_episodes/--eval.n_episodes must be >= 1.")
    if args.interpolated < 1:
        raise ValueError("--interpolated must be >= 1.")
    if args.init_steps < 0:
        raise ValueError("--init_steps must be >= 0.")
    if args.decimation is not None and args.decimation < 1:
        raise ValueError("--decimation must be >= 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_episode_results.csv"
    eval_info_path = output_dir / "eval_info.json"
    action_trace_path = Path(args.action_trace_csv) if args.action_trace_csv else output_dir / "action_trace_joint_states.csv"
    for path in (csv_path, eval_info_path):
        if path.exists():
            path.unlink()
    if args.debug_action_trace and action_trace_path.exists():
        action_trace_path.unlink()

    device = get_safe_torch_device(args.policy_device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    env_cfg = make_isaaclab_arena_cfg(args)
    policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor = build_policy_and_processors(
        args, env_cfg
    )

    env_map = make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=True)
    env = env_map["summit_franka_open_door_eval"][0]
    apply_decimation_override(env, args.decimation)
    sync_episode_budget(env, args.max_steps)

    action_trace_logger = ActionTraceLogger(action_trace_path, list(SUMMIT_FRANKA_ACTION_JOINT_NAMES)) if args.debug_action_trace else None
    if action_trace_logger is not None:
        logging.info("Writing action trace to %s", action_trace_path)

    executor_kwargs = {
        "interpolation_factor": args.interpolated,
        "interpolation_type": args.interpolation_type,
        "state_fn": (lambda step_obs: extract_sim_joint_pos(env, step_obs)) if action_trace_logger is not None else None,
    }
    if hasattr(env, "make_action_executor"):
        action_executor = env.make_action_executor(**executor_kwargs)
    else:
        from isaaclab_arena.utils.action_execution import InterpolatedActionExecutor

        action_executor = InterpolatedActionExecutor(
            env,
            **executor_kwargs,
        )

    videos_dir = output_dir / "videos" / "summit_franka_open_door_eval_0"
    per_episode: list[dict[str, Any]] = []
    successes: list[bool] = []
    video_paths: list[str] = []
    start = time.time()

    amp_context = torch.autocast(device_type=device.type) if policy.config.use_amp else nullcontext()
    try:
        with torch.no_grad(), amp_context:
            for episode_ix in range(args.n_episodes):
                episode_seed = args.seed + episode_ix
                render_video = episode_ix < args.max_episodes_rendered
                result = run_episode(
                    episode_ix=episode_ix,
                    env=env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    seed=episode_seed,
                    max_steps=args.max_steps,
                    init_steps=args.init_steps,
                    action_executor=action_executor,
                    action_trace_logger=action_trace_logger,
                    render_video=render_video,
                )

                video_path = ""
                if render_video and result["frames"]:
                    videos_dir.mkdir(parents=True, exist_ok=True)
                    video_file = videos_dir / f"eval_episode_{episode_ix}.mp4"
                    fps = int(getattr(env, "metadata", {}).get("render_fps", 50))
                    write_video(str(video_file), np.stack(result["frames"], axis=0), fps)
                    video_path = str(video_file)
                    video_paths.append(video_path)

                final_info = result["final_info"]
                row = {
                    "episode_ix": episode_ix,
                    "seed": episode_seed,
                    "success": bool(result["success"]),
                    "final_door_open": first_scalar(final_info.get("final_door_open"), ""),
                    "final_door_openness": first_scalar(final_info.get("final_door_openness"), ""),
                    "final_engaged": first_scalar(final_info.get("final_engaged"), ""),
                    "final_handle_distance": first_scalar(final_info.get("final_handle_distance"), ""),
                    "steps": f"{result['sim_steps']}/{args.max_steps}",
                    "policy_steps": result["policy_steps"],
                    "video_path": video_path,
                }
                append_per_episode_csv_row(csv_path, row)
                per_episode.append(
                    {
                        "episode_ix": episode_ix,
                        "seed": episode_seed,
                        "success": row["success"],
                        "final_door_openness": row["final_door_openness"],
                        "final_handle_distance": row["final_handle_distance"],
                    }
                )
                successes.append(bool(row["success"]))
                logging.info(
                    "episode=%s success=%s steps=%s final_open=%s final_engaged=%s",
                    episode_ix,
                    row["success"],
                    row["steps"],
                    row["final_door_open"],
                    row["final_engaged"],
                )
    finally:
        if action_trace_logger is not None:
            action_trace_logger.close()

    info = {
        "per_episode": per_episode,
        "aggregated": {
            "pc_success": float(np.mean(successes) * 100.0) if successes else 0.0,
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / max(len(successes), 1),
        },
        "video_paths": video_paths,
        "alignment": {
            "max_steps": args.max_steps,
            "decimation": args.decimation,
            "init_steps": args.init_steps,
            "interpolated": args.interpolated,
            "interpolation_type": args.interpolation_type,
            "mobile_base_relative": bool(args.mobile_base_relative),
            "traj_file": str(Path(args.traj_file).resolve()),
            "traj_seed": args.traj_seed,
            "traj_selection_mode": args.traj_selection_mode,
            "debug_action_trace": bool(args.debug_action_trace),
            "action_trace_csv": str(action_trace_path) if args.debug_action_trace else "",
        },
    }
    if action_trace_logger is not None:
        info["action_trace_summary"] = action_trace_logger.summary()
    eval_info_path.write_text(json.dumps(info, indent=2))
    print(json.dumps(info["aggregated"], indent=2), flush=True)

    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
