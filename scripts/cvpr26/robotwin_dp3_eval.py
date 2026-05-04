from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import dill
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DP3_ROOT = REPO_ROOT / "third_party" / "RoboTwin" / "policy" / "DP3"
DP3_DIFFUSION_ROOT = DP3_ROOT / "3D-Diffusion-Policy"
ISAAC_ENV_ROOT = REPO_ROOT / "third_party" / "IsaacLab-Arena" / "isaaclab-arena-envs"
LEROBOT_SRC_ROOT = REPO_ROOT / "third_party" / "lerobot" / "src"

for path in (
    str(DP3_ROOT),
    str(DP3_ROOT / "scripts"),
    str(DP3_DIFFUSION_ROOT),
    str(ISAAC_ENV_ROOT),
    str(LEROBOT_SRC_ROOT),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from automoma_dp3_utils import PointCloudConfig, rgbd_to_pointcloud
from diffusion_policy_3d.env_runner.robot_runner import RobotRunner
from lerobot.utils.io_utils import write_video
from train_dp3 import TrainDP3Workspace


PER_EPISODE_CSV_COLUMNS = [
    "episode_ix",
    "seed",
    "success",
    "final_door_open",
    "final_door_openness",
    "final_engaged",
    "final_handle_distance",
    "preclose_gripper_left",
    "preclose_gripper_right",
    "preclose_base_arm_max_abs_delta",
    "preclose_diagnostics_path",
    "steps",
    "video_path",
]


def load_module(module_name: str, file_path: Path):
    spec = __import__("importlib.util").util.spec_from_file_location(module_name, file_path)
    module = __import__("importlib.util").util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


env_module = load_module("automoma_isaac_env", ISAAC_ENV_ROOT / "env.py")

CAMERA_CHOICES = ("ego_topdown", "ego_wrist", "fix_local")
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


def append_per_episode_csv_row(csv_path: Path, row: dict[str, object]) -> None:
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_EPISODE_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in PER_EPISODE_CSV_COLUMNS})


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


class SimpleEnvConfig:
    def __init__(self, **kwargs):
        self.environment = kwargs.pop("environment")
        self.embodiment = kwargs.pop("embodiment", "gr1_pink")
        self.object = kwargs.pop("object", "power_drill")
        self.mimic = kwargs.pop("mimic", False)
        self.teleop_device = kwargs.pop("teleop_device", None)
        self.seed = kwargs.pop("seed", 42)
        self.device = kwargs.pop("device", "cuda:0")
        self.disable_fabric = kwargs.pop("disable_fabric", False)
        self.enable_cameras = kwargs.pop("enable_cameras", True)
        self.headless = kwargs.pop("headless", True)
        self.enable_pinocchio = kwargs.pop("enable_pinocchio", True)
        self.episode_length = kwargs.pop("episode_length", 300)
        self.state_dim = kwargs.pop("state_dim", 12)
        self.action_dim = kwargs.pop("action_dim", 12)
        self.camera_height = kwargs.pop("camera_height", 240)
        self.camera_width = kwargs.pop("camera_width", 320)
        self.video = kwargs.pop("video", False)
        self.video_length = kwargs.pop("video_length", 100)
        self.video_interval = kwargs.pop("video_interval", 200)
        self.state_keys = kwargs.pop("state_keys", "joint_pos")
        self.camera_keys = kwargs.pop("camera_keys", "ego_topdown_rgb,ego_wrist_rgb,fix_local_rgb")
        self.task = kwargs.pop("task", "Reach out to the microwave and open it.")
        self.disable_env_checker = kwargs.pop("disable_env_checker", True)
        self.fps = kwargs.pop("fps", 30)
        self.features = kwargs.pop("features", None)
        self.features_map = kwargs.pop("features_map", None)
        self.max_parallel_tasks = kwargs.pop("max_parallel_tasks", 1)
        self.kwargs = kwargs.pop("kwargs", None)
        for key, value in kwargs.items():
            setattr(self, key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RoboTwin DP3 on IsaacLab-Arena AutoMoMa env")
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--task_config", required=True)
    parser.add_argument("--expert_data_num", type=int, required=True)
    parser.add_argument("--ckpt_setting", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--checkpoint_num", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_task_name", type=str, default=None)
    parser.add_argument("--checkpoint_setting", type=str, default=None)
    parser.add_argument("--checkpoint_expert_data_num", type=int, default=None)
    parser.add_argument("--legacy_cvpr26", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--camera_view", choices=CAMERA_CHOICES, default="ego_topdown")
    parser.add_argument("--n_points", type=int, default=None)
    parser.add_argument("--random_drop_points", type=int, default=None)
    parser.add_argument("--use_fps", type=str2bool, default=None)
    parser.add_argument("--use_rgb", type=str2bool, default=None)
    parser.add_argument("--fov_deg", type=float, default=60.0)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--eval.n_episodes", dest="n_episodes", type=int, default=10)
    parser.add_argument("--checkpoint_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--traj_file", type=str, default=None)
    parser.add_argument("--traj_seed", type=int, default=42)
    parser.add_argument("--interpolated", type=int, default=1)
    parser.add_argument("--interpolation_type", type=str, default="linear")
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument("--init_steps", type=int, default=1)
    parser.add_argument("--headless", type=str2bool, default=True)
    parser.add_argument("--env.headless", dest="headless", type=str2bool)
    parser.add_argument("--max_episodes_rendered", type=int, default=10)
    parser.add_argument("--debug_visualize_handle", type=str2bool, default=False)
    parser.add_argument("--debug_record_handle_diagnostics", type=str2bool, default=False)
    parser.add_argument("--handle_distance_threshold", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--preclose_steps", type=int, default=None)
    parser.add_argument("--preclose_gripper_value", type=float, default=None)
    parser.add_argument("--preclose_lock_base_arm", type=str2bool, default=None)
    parser.add_argument("--obs_gripper_value", type=float, default=None)
    parser.add_argument("--action_gripper_value", type=float, default=None)
    return parser.parse_args()


def _unique_existing(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    existing: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        existing.append(resolved)
    return existing


def _numeric_ckpt_key(path: Path) -> tuple[int, str]:
    try:
        number = int(path.stem)
    except ValueError:
        number = -1
    return number, str(path)


def _checkpoint_dir_name(task_name: str, ckpt_setting: str, expert_data_num: int, seed: int, use_pc_color: bool) -> str:
    return TrainDP3Workspace.checkpoint_dir_name(
        task_name=task_name,
        ckpt_setting=ckpt_setting,
        expert_data_num=expert_data_num,
        seed=seed,
        use_pc_color=use_pc_color,
    )


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint_path not found: {checkpoint_path}")
        return checkpoint_path

    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve() if args.checkpoint_root else None
    if checkpoint_root is None:
        if args.legacy_cvpr26:
            checkpoint_root = REPO_ROOT / "outputs" / "train" / "debug_eval" / "robotwin" / "dp3"
        else:
            checkpoint_root = REPO_ROOT / "outputs" / "train" / "robotwin" / f"dp3_{args.task_name}-{args.task_config}-{args.expert_data_num}"

    if checkpoint_root.is_file():
        return checkpoint_root
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"checkpoint_root not found: {checkpoint_root}")

    expert_data_num = args.checkpoint_expert_data_num or args.expert_data_num
    checkpoint_num = args.checkpoint_num if args.checkpoint_num is not None else (None if args.legacy_cvpr26 else 3000)
    model_use_pc_color = bool(args.use_rgb) if args.use_rgb is not None and not args.legacy_cvpr26 else False

    task_names = []
    if args.checkpoint_task_name:
        task_names.append(args.checkpoint_task_name)
    if args.legacy_cvpr26:
        task_names.append("automoma_manip_summit_franka")
    task_names.append(args.task_name)

    settings = []
    if args.checkpoint_setting:
        settings.append(args.checkpoint_setting)
    settings.append(args.ckpt_setting)
    if args.legacy_cvpr26:
        settings.extend(
            [
                "task_1object_1scene_20pose",
                "task_1object_30scene_20pose",
                "task_5object_30scene_20pose",
            ]
        )

    candidates: list[Path] = []
    if checkpoint_num is not None:
        for task_name in dict.fromkeys(task_names):
            for setting in dict.fromkeys(settings):
                dirname = _checkpoint_dir_name(
                    task_name,
                    setting,
                    expert_data_num,
                    args.seed,
                    model_use_pc_color,
                )
                candidates.append(checkpoint_root / "checkpoints" / dirname / f"{checkpoint_num}.ckpt")
                candidates.append(checkpoint_root / dirname / f"{checkpoint_num}.ckpt")

    existing = _unique_existing(candidates)
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise RuntimeError("Ambiguous checkpoint candidates:\n" + "\n".join(str(path) for path in existing))

    globbed = sorted(checkpoint_root.glob("**/*.ckpt"))
    if checkpoint_num is not None:
        globbed = [path for path in globbed if path.name == f"{checkpoint_num}.ckpt"]

    expert_seed_token = f"-{expert_data_num}_{args.seed}"
    filtered = [path for path in globbed if expert_seed_token in path.parent.name]
    if not filtered:
        expert_token = f"-{expert_data_num}_"
        filtered = [path for path in globbed if expert_token in path.parent.name]
    if args.checkpoint_task_name:
        filtered = [path for path in filtered if args.checkpoint_task_name in path.parent.name]
    if args.checkpoint_setting:
        filtered = [path for path in filtered if args.checkpoint_setting in path.parent.name]
    if not filtered and globbed and not args.legacy_cvpr26:
        filtered = globbed

    if not filtered:
        details = (
            f"root={checkpoint_root}, expert_data_num={expert_data_num}, "
            f"seed={args.seed}, checkpoint_num={checkpoint_num}"
        )
        raise FileNotFoundError(f"No DP3 checkpoint found ({details})")

    filtered = sorted(filtered, key=_numeric_ckpt_key)
    return filtered[-1].resolve()


def load_checkpoint_payload(checkpoint_path: Path) -> dict:
    return torch.load(checkpoint_path.open("rb"), pickle_module=dill, map_location="cpu")


def make_cfg(args: argparse.Namespace, checkpoint_payload: dict | None = None):
    if checkpoint_payload is not None and checkpoint_payload.get("cfg") is not None:
        cfg = copy.deepcopy(checkpoint_payload["cfg"])
        OmegaConf.set_struct(cfg, False)
        cfg.training.device = f"cuda:{args.gpu_id}"
        OmegaConf.set_struct(cfg, True)
        return cfg

    config_path = DP3_DIFFUSION_ROOT / "diffusion_policy_3d" / "config"
    with __import__("hydra").initialize_config_dir(config_dir=str(config_path), version_base="1.2"):
        cfg = __import__("hydra").compose(config_name="robot_dp3.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.task_name = args.task_name
    cfg.expert_data_num = args.expert_data_num
    cfg.setting = args.ckpt_setting
    cfg.raw_task_name = args.task_name
    cfg.training.device = f"cuda:{args.gpu_id}"
    cfg.policy.use_pc_color = bool(args.use_rgb) if args.use_rgb is not None else False
    cfg.task.shape_meta.obs.agent_pos.shape = [12]
    cfg.task.shape_meta.action.shape = [12]
    OmegaConf.set_struct(cfg, True)
    return cfg


def get_policy_dims(cfg) -> tuple[int, int]:
    agent_pos_dim = int(cfg.task.shape_meta.obs.agent_pos.shape[0])
    point_count = int(cfg.task.shape_meta.obs.point_cloud.shape[0])
    return agent_pos_dim, point_count


def get_point_count(cfg) -> int:
    return int(cfg.task.shape_meta.obs.point_cloud.shape[0])


def configure_runtime_defaults(args: argparse.Namespace, cfg) -> None:
    cfg_point_count = get_point_count(cfg)
    if args.n_points is None:
        args.n_points = 4096 if args.legacy_cvpr26 else cfg_point_count
    if args.random_drop_points is None:
        if args.legacy_cvpr26:
            args.random_drop_points = min(int(0.25 * 240 * 320), 2 * args.n_points)
        else:
            args.random_drop_points = 5000
    if args.use_fps is None:
        args.use_fps = False if args.legacy_cvpr26 else True
    if args.use_rgb is None:
        args.use_rgb = True if args.legacy_cvpr26 else bool(cfg.policy.use_pc_color)
    if args.preclose_steps is None:
        args.preclose_steps = 60 if args.legacy_cvpr26 else 0
    if args.preclose_gripper_value is None:
        args.preclose_gripper_value = 0.0 if args.legacy_cvpr26 else None
    if args.preclose_lock_base_arm is None:
        args.preclose_lock_base_arm = bool(args.legacy_cvpr26)
    if args.obs_gripper_value is None:
        args.obs_gripper_value = 0.02 if args.legacy_cvpr26 else None
    if args.action_gripper_value is None:
        args.action_gripper_value = 0.0 if args.legacy_cvpr26 else None


def load_policy(cfg, checkpoint_payload: dict, checkpoint_path: Path):
    workspace = TrainDP3Workspace(cfg)
    workspace.load_payload(checkpoint_payload)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    policy.cuda()

    env_runner = RobotRunner(
        n_obs_steps=int(cfg.n_obs_steps),
        n_action_steps=int(cfg.n_action_steps),
    )
    print(f"Loaded DP3 checkpoint: {checkpoint_path}", flush=True)
    return policy, env_runner


def parse_env_identifiers(task_name: str, task_config: str) -> tuple[str, str]:
    parts = task_config.split("-")
    if len(parts) >= 3:
        object_name = parts[0]
        scene_name = parts[1]
        return object_name, scene_name
    if len(parts) == 2:
        object_name, scene_name = parts
        return object_name, scene_name
    return task_name, task_config


def build_env(args: argparse.Namespace):
    object_name, scene_name = parse_env_identifiers(args.task_name, args.task_config)
    if args.interpolated < 1:
        raise ValueError("--interpolated must be >= 1.")
    if args.decimation is not None and args.decimation < 1:
        raise ValueError("--decimation must be >= 1.")
    if args.init_steps < 1:
        raise ValueError("--init_steps must be >= 1.")
    traj_file = Path(args.traj_file) if args.traj_file else REPO_ROOT / "data" / "trajs" / "summit_franka" / object_name / scene_name / "test" / "traj_data_test.pt"
    preclose_steps = max(int(args.preclose_steps or 0), 0)
    episode_length = int(args.max_steps) + preclose_steps
    cfg = SimpleEnvConfig(
        environment="summit_franka_open_door_eval",
        headless=args.headless,
        enable_cameras=True,
        state_keys="joint_pos",
        camera_keys="ego_topdown_rgb,ego_wrist_rgb,fix_local_rgb",
        state_dim=12,
        action_dim=12,
        camera_height=240,
        camera_width=320,
        episode_length=episode_length,
        object_name=object_name,
        scene_name=scene_name,
        object_center=True,
        mobile_base_relative=True,
        traj_file=str(traj_file),
        traj_seed=args.traj_seed,
        interpolated=args.interpolated,
        interpolation_type=args.interpolation_type,
        decimation=args.decimation,
        openness_threshold=0.3,
        handle_distance_threshold=args.handle_distance_threshold,
        proximity_threshold=0.12,
        proximity_window_steps=8,
        proximity_required_steps=5,
        disable_fingertip_proximity=False,
        debug_visualize_handle=args.debug_visualize_handle,
        debug_record_handle_diagnostics=args.debug_record_handle_diagnostics,
        debug_marker_scale=1.0,
    )
    env_map = env_module.make_env(n_envs=1, cfg=cfg)
    env = env_map["summit_franka_open_door_eval"][0]
    apply_decimation_override(env, args.decimation)
    sync_episode_budget(env, episode_length)
    return env


def apply_decimation_override(env, decimation: int | None) -> None:
    if decimation is None:
        return
    raw_env = getattr(env, "_env", None)
    raw_cfg = getattr(raw_env, "cfg", None)
    if raw_cfg is None:
        return
    raw_cfg.decimation = int(decimation)
    sim_cfg = getattr(raw_cfg, "sim", None)
    if sim_cfg is not None and hasattr(sim_cfg, "render_interval"):
        sim_cfg.render_interval = int(decimation)


def sync_episode_budget(env, total_steps: int) -> None:
    if hasattr(env, "_episode_length"):
        env._episode_length = int(total_steps)

    raw_env = getattr(env, "_env", None)
    raw_cfg = getattr(raw_env, "cfg", None)
    if raw_env is None or raw_cfg is None or not hasattr(raw_cfg, "episode_length_s"):
        return

    step_dt = float(getattr(raw_env, "step_dt", 0.02) or 0.02)
    raw_cfg.episode_length_s = int(total_steps) * step_dt


def extract_point_cloud(obs: dict, camera_view: str, pc_cfg: PointCloudConfig, rng: np.random.Generator) -> np.ndarray:
    rgb = obs["camera_obs"][f"{camera_view}_rgb"]
    depth = obs["camera_obs"][f"{camera_view}_depth"]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    rgb = rgb[0] if rgb.ndim == 4 else rgb
    depth = depth[0] if depth.ndim == 4 else depth
    return rgbd_to_pointcloud(rgb, depth, pc_cfg, rng)


def extract_joint_pos(obs: dict) -> np.ndarray:
    joint_pos = obs["policy"]["joint_pos"]
    if isinstance(joint_pos, torch.Tensor):
        joint_pos = joint_pos.detach().cpu().numpy()
    return joint_pos[0].astype(np.float32) if joint_pos.ndim == 2 else joint_pos.astype(np.float32)


def get_sim_joint_names(env) -> list[str]:
    raw_env = getattr(env, "_env", None)
    scene = getattr(raw_env, "scene", None)
    if raw_env is not None and scene is not None and "robot" in scene.keys():
        robot = scene["robot"]
        if hasattr(robot.data, "joint_names"):
            return list(robot.data.joint_names)
    return []


def get_action_joint_names(env) -> list[str]:
    raw_env = getattr(env, "_env", None)
    action_manager = getattr(raw_env, "action_manager", None)
    if action_manager is None:
        return []
    joint_names: list[str] = []
    for term in getattr(action_manager, "_terms", {}).values():
        names = getattr(term, "_joint_names", None)
        if names is not None:
            joint_names.extend(list(names))
    return joint_names


def extract_sim_joint_pos(env, fallback_obs: dict | None = None) -> np.ndarray:
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
        raise RuntimeError("Could not read robot joint state from simulator")
    return extract_joint_pos(fallback_obs)


def _scene_articulations(env):
    raw_env = getattr(env, "_env", None)
    scene = getattr(raw_env, "scene", None)
    if raw_env is None or scene is None:
        return raw_env, []
    entities = []
    for key in scene.keys():
        entity = scene[key]
        if hasattr(entity, "write_joint_state_to_sim") and hasattr(entity, "data"):
            entities.append(entity)
    return raw_env, entities


def settle_initial_state(env, init_steps: int) -> dict | None:
    steps = max(int(init_steps), 1)
    if steps <= 1:
        return None

    raw_env, entities = _scene_articulations(env)
    if raw_env is None or not entities:
        return None

    states = []
    for entity in entities:
        states.append((entity, entity.data.joint_pos.clone(), entity.data.joint_vel.clone() * 0.0))

    for _ in range(steps - 1):
        for entity, joint_pos, joint_vel in states:
            entity.write_joint_state_to_sim(joint_pos, joint_vel)
        scene = getattr(raw_env, "scene", None)
        if hasattr(scene, "write_data_to_sim"):
            scene.write_data_to_sim()
        sim = getattr(raw_env, "sim", None)
        if hasattr(sim, "step"):
            sim.step(render=True)
        elif hasattr(sim, "render"):
            sim.render()
        if hasattr(scene, "update"):
            scene.update(getattr(raw_env, "physics_dt", 0.0))

    observation_manager = getattr(raw_env, "observation_manager", None)
    if observation_manager is None:
        return None
    obs = observation_manager.compute()
    raw_env.obs_buf = obs
    return obs


def lock_base_arm_state(env, reference_joint_pos: np.ndarray) -> dict | None:
    raw_env = getattr(env, "_env", None)
    scene = getattr(raw_env, "scene", None)
    if raw_env is None or scene is None or "robot" not in scene.keys():
        return None

    robot = scene["robot"]
    if not hasattr(robot, "write_joint_state_to_sim") or not hasattr(robot, "data"):
        return None

    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    freeze_dim = min(max(int(reference_joint_pos.shape[0]) - 2, 0), int(joint_pos.shape[1]))
    if freeze_dim <= 0:
        return None

    hold = torch.as_tensor(reference_joint_pos[:freeze_dim], dtype=joint_pos.dtype, device=joint_pos.device)
    joint_pos[:, :freeze_dim] = hold.unsqueeze(0)
    joint_vel[:, :freeze_dim] = 0.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    if hasattr(scene, "write_data_to_sim"):
        scene.write_data_to_sim()
    if hasattr(raw_env, "sim") and hasattr(raw_env.sim, "render"):
        raw_env.sim.render()
    if hasattr(scene, "update"):
        scene.update(getattr(raw_env, "physics_dt", 0.0))

    observation_manager = getattr(raw_env, "observation_manager", None)
    if observation_manager is None:
        return None
    obs = observation_manager.compute()
    raw_env.obs_buf = obs
    return obs


def policy_agent_pos(joint_pos: np.ndarray, policy_state_dim: int, obs_gripper_value: float | None) -> np.ndarray:
    state = np.asarray(joint_pos, dtype=np.float32).copy()
    if obs_gripper_value is not None:
        if state.shape[0] >= 12:
            state[-2:] = obs_gripper_value
        elif state.shape[0] >= 11:
            state[-1] = obs_gripper_value

    if policy_state_dim == 11 and state.shape[0] >= 12:
        gripper = np.array([state[-2:].mean()], dtype=np.float32)
        state = np.concatenate([state[:10], gripper]).astype(np.float32)
    elif policy_state_dim == 12 and state.shape[0] == 11:
        state = np.concatenate([state, state[-1:]]).astype(np.float32)

    if state.shape[0] < policy_state_dim:
        state = np.pad(state, (0, policy_state_dim - state.shape[0]))
    elif state.shape[0] > policy_state_dim:
        state = state[:policy_state_dim]
    return state.astype(np.float32)


def policy_action_to_env_action(
    action: np.ndarray,
    env_action_dim: int,
    action_gripper_value: float | None,
) -> np.ndarray:
    env_action = np.asarray(action, dtype=np.float32).copy()
    if env_action.ndim != 1:
        env_action = env_action.reshape(-1)

    if env_action.shape[0] == 11 and env_action_dim == 12:
        env_action = np.concatenate([env_action, env_action[-1:]]).astype(np.float32)
    elif env_action.shape[0] < env_action_dim:
        env_action = np.pad(env_action, (0, env_action_dim - env_action.shape[0]))
    elif env_action.shape[0] > env_action_dim:
        env_action = env_action[:env_action_dim]

    if action_gripper_value is not None and env_action.shape[0] >= 2:
        env_action[-2:] = action_gripper_value
    return env_action.astype(np.float32)


def _disabled_termination(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def disable_success_termination(env):
    raw_env = getattr(env, "_env", None)
    termination_manager = getattr(raw_env, "termination_manager", None)
    if termination_manager is None or "success" not in getattr(termination_manager, "active_terms", []):
        return lambda: None

    term_cfg = termination_manager.get_term_cfg("success")
    original_func = term_cfg.func
    original_params = term_cfg.params
    term_cfg.func = _disabled_termination
    term_cfg.params = {}

    def restore() -> None:
        term_cfg.func = original_func
        term_cfg.params = original_params

    return restore


def make_dp3_obs(
    obs: dict,
    camera_view: str,
    pc_cfg: PointCloudConfig,
    rng: np.random.Generator,
    policy_state_dim: int,
    obs_gripper_value: float | None,
) -> dict[str, np.ndarray]:
    point_cloud = extract_point_cloud(obs, camera_view, pc_cfg, rng)
    joint_pos = extract_joint_pos(obs)
    return {
        "point_cloud": point_cloud.astype(np.float32),
        "agent_pos": policy_agent_pos(joint_pos, policy_state_dim, obs_gripper_value),
    }


def preclose_policy(
    joint_pos: np.ndarray,
    env_action_dim: int,
    close_value: float,
    mobile_base_relative: bool,
) -> np.ndarray:
    action = np.asarray(joint_pos, dtype=np.float32).copy()
    if action.shape[0] < env_action_dim:
        action = np.pad(action, (0, env_action_dim - action.shape[0]))
    elif action.shape[0] > env_action_dim:
        action = action[:env_action_dim]

    if mobile_base_relative and action.shape[0] >= 3:
        action[:3] = 0.0
    if action.shape[0] >= 2:
        action[-2:] = close_value
    return action.astype(np.float32)


def run_preclose(
    env,
    obs: dict,
    args: argparse.Namespace,
    frames: list[np.ndarray],
    episode_ix: int,
    reference_joint_pos: np.ndarray,
) -> tuple[dict, int, dict, dict[str, object]]:
    total_preclose_steps = max(int(args.preclose_steps or 0), 0)
    diagnostics: dict[str, object] = {
        "episode_ix": episode_ix,
        "preclose_steps": total_preclose_steps,
        "joint_names": list(SUMMIT_FRANKA_ACTION_JOINT_NAMES),
        "sim_joint_names": get_sim_joint_names(env),
        "action_joint_names": get_action_joint_names(env),
        "reference_joint_pos": np.asarray(reference_joint_pos, dtype=np.float32).tolist(),
        "steps": [],
    }
    if total_preclose_steps == 0:
        return obs, 0, {"is_success": np.array([False])}, diagnostics

    env_action_dim = env.action_space.shape[-1]
    mobile_base_relative = bool(getattr(env, "_mobile_base_relative", False))
    if args.preclose_gripper_value is None:
        close_value = 0.0 if args.action_gripper_value is None else float(args.action_gripper_value)
    else:
        close_value = float(args.preclose_gripper_value)

    freeze_dim = min(max(int(reference_joint_pos.shape[0]) - 2, 0), int(env_action_dim))
    diagnostics.update({
        "env_action_dim": int(env_action_dim),
        "mobile_base_relative": mobile_base_relative,
        "close_value": close_value,
        "preclose_lock_base_arm": bool(args.preclose_lock_base_arm),
        "freeze_dim": freeze_dim,
    })

    final_info = {"is_success": np.array([False])}
    restore_success_termination = disable_success_termination(env)
    try:
        for step_ix in range(total_preclose_steps):
            joint_pos_before = extract_sim_joint_pos(env, obs)
            action = preclose_policy(reference_joint_pos, env_action_dim, close_value, mobile_base_relative)
            obs, _reward, _terminated, _truncated, info = env.step(action[None, :])
            if args.preclose_lock_base_arm:
                locked_obs = lock_base_arm_state(env, reference_joint_pos)
                if locked_obs is not None:
                    obs = locked_obs
            joint_pos_after = extract_sim_joint_pos(env, obs)
            final_info = info.get("final_info", final_info)
            if freeze_dim > 0:
                base_arm_delta = np.abs(joint_pos_after[:freeze_dim] - reference_joint_pos[:freeze_dim])
                step_base_arm_max_abs_delta = float(np.max(base_arm_delta))
                step_base_arm_l2_delta = float(np.linalg.norm(base_arm_delta))
            else:
                step_base_arm_max_abs_delta = np.nan
                step_base_arm_l2_delta = np.nan
            diagnostics["steps"].append({
                "step_ix": step_ix,
                "joint_pos_before": np.asarray(joint_pos_before, dtype=np.float32).tolist(),
                "joint_pos_after": np.asarray(joint_pos_after, dtype=np.float32).tolist(),
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "base_arm_max_abs_delta_from_reset": step_base_arm_max_abs_delta,
                "base_arm_l2_delta_from_reset": step_base_arm_l2_delta,
                "gripper_after": np.asarray(joint_pos_after[-2:], dtype=np.float32).tolist() if joint_pos_after.shape[0] >= 2 else [],
            })
            if episode_ix < args.max_episodes_rendered:
                frame = render_frame(env)
                if frame is not None:
                    frames.append(frame)
    finally:
        restore_success_termination()

    return obs, total_preclose_steps, final_info, diagnostics


def get_success_metrics(final_info: dict) -> dict[str, float | bool]:
    def pick(key, default=np.nan):
        value = final_info.get(key)
        if value is None:
            return default
        if isinstance(value, np.ndarray):
            return value[0].item() if value.size else default
        return value

    return {
        "success": bool(pick("is_success", False)),
        "final_door_openness": float(pick("final_openness")),
        "final_door_open": bool(pick("final_door_open", False)),
        "final_engaged": bool(pick("final_engaged", False)),
        "final_handle_distance": float(pick("final_handle_distance")),
    }


def maybe_scalar(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return ""
    return value


def gripper_values_from_obs(obs: dict) -> tuple[float, float]:
    joint_pos = extract_joint_pos(obs)
    if joint_pos.shape[0] < 2:
        return np.nan, np.nan
    return float(joint_pos[-2]), float(joint_pos[-1])


def base_arm_max_abs_delta(obs: dict, reference_joint_pos: np.ndarray) -> float:
    joint_pos = extract_joint_pos(obs)
    freeze_dim = min(max(int(reference_joint_pos.shape[0]) - 2, 0), int(joint_pos.shape[0]))
    if freeze_dim <= 0:
        return np.nan
    return float(np.max(np.abs(joint_pos[:freeze_dim] - reference_joint_pos[:freeze_dim])))


def render_frame(env) -> np.ndarray | None:
    frame = env.render()
    if isinstance(frame, list):
        if not frame:
            return None
        frame = frame[0]
    if frame is None:
        return None
    return np.asarray(frame)


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(int(args.gpu_id))

    checkpoint_path = resolve_checkpoint_path(args)
    checkpoint_payload = load_checkpoint_payload(checkpoint_path)
    cfg = make_cfg(args, checkpoint_payload)
    configure_runtime_defaults(args, cfg)
    policy_state_dim, _cfg_point_count = get_policy_dims(cfg)
    policy, env_runner = load_policy(cfg, checkpoint_payload, checkpoint_path)
    env = build_env(args)

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "eval" / "robotwin" / f"dp3_{args.task_name}-{args.task_config}-{args.expert_data_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_episode_results.csv"
    videos_dir = output_dir / "videos"
    start_time = time.time()

    if csv_path.exists():
        csv_path.unlink()

    pc_cfg = PointCloudConfig(
        n_points=args.n_points,
        random_drop_points=args.random_drop_points,
        use_fps=args.use_fps,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        fov_deg=args.fov_deg,
        use_rgb=args.use_rgb,
    )
    rng = np.random.default_rng(args.seed)
    env_action_dim = env.action_space.shape[-1]

    per_episode: list[dict[str, object]] = []
    all_successes: list[bool] = []
    video_paths: list[str] = []

    try:
        for episode_ix in range(args.n_episodes):
            episode_seed = args.seed + episode_ix
            obs, _info = env.reset(seed=episode_seed)
            env_runner.reset_obs()
            settled_obs = settle_initial_state(env, args.init_steps)
            if settled_obs is not None:
                obs = settled_obs

            frames: list[np.ndarray] = []
            if episode_ix < args.max_episodes_rendered:
                frame = render_frame(env)
                if frame is not None:
                    frames.append(frame)

            preclose_reference_joint_pos = extract_joint_pos(obs)
            obs, preclose_count, final_info, preclose_diagnostics = run_preclose(
                env,
                obs,
                args,
                frames,
                episode_ix,
                preclose_reference_joint_pos,
            )
            done = False
            preclose_gripper_left, preclose_gripper_right = gripper_values_from_obs(obs)
            preclose_base_arm_delta = base_arm_max_abs_delta(obs, preclose_reference_joint_pos)
            preclose_diagnostics.update({
                "final_gripper": [preclose_gripper_left, preclose_gripper_right],
                "final_base_arm_max_abs_delta_from_reset": preclose_base_arm_delta,
            })

            policy_steps = 0
            while not done and policy_steps < args.max_steps:
                dp3_obs = make_dp3_obs(
                    obs,
                    args.camera_view,
                    pc_cfg,
                    rng,
                    policy_state_dim,
                    args.obs_gripper_value,
                )

                if len(env_runner.obs) == 0:
                    env_runner.update_obs(dp3_obs)
                    actions = env_runner.get_action(policy)
                else:
                    actions = env_runner.get_action(policy, dp3_obs)
                for action in actions:
                    env_action = policy_action_to_env_action(action, env_action_dim, args.action_gripper_value)
                    obs, _reward, terminated, truncated, info = env.step(env_action[None, :])
                    policy_steps += 1
                    done = bool(terminated[0] or truncated[0])
                    final_info = info.get("final_info", final_info)
                    if episode_ix < args.max_episodes_rendered:
                        frame = render_frame(env)
                        if frame is not None:
                            frames.append(frame)
                    env_runner.update_obs(
                        make_dp3_obs(
                            obs,
                            args.camera_view,
                            pc_cfg,
                            rng,
                            policy_state_dim,
                            args.obs_gripper_value,
                        )
                    )
                    if done or policy_steps >= args.max_steps:
                        break

            metrics = get_success_metrics(final_info)
            preclose_diagnostics_path = ""
            if preclose_count > 0:
                diagnostics_dir = output_dir / "preclose_diagnostics"
                diagnostics_dir.mkdir(parents=True, exist_ok=True)
                diagnostics_file = diagnostics_dir / f"preclose_diagnostics_episode_{episode_ix}.json"
                with diagnostics_file.open("w") as f:
                    json.dump(preclose_diagnostics, f, indent=2)
                preclose_diagnostics_path = str(diagnostics_file)

            video_path = ""
            if episode_ix < args.max_episodes_rendered and frames:
                videos_dir.mkdir(parents=True, exist_ok=True)
                video_file = videos_dir / f"eval_episode_{episode_ix}.mp4"
                fps = int(getattr(env, "metadata", {}).get("render_fps", 30))
                write_video(video_file, frames, fps)
                video_path = str(video_file)
                video_paths.append(video_path)

            row = {
                "episode_ix": episode_ix,
                "seed": episode_seed,
                "success": bool(metrics["success"]),
                "final_door_open": maybe_scalar(metrics["final_door_open"]),
                "final_door_openness": maybe_scalar(metrics["final_door_openness"]),
                "final_engaged": maybe_scalar(metrics["final_engaged"]),
                "final_handle_distance": maybe_scalar(metrics["final_handle_distance"]),
                "preclose_gripper_left": maybe_scalar(preclose_gripper_left),
                "preclose_gripper_right": maybe_scalar(preclose_gripper_right),
                "preclose_base_arm_max_abs_delta": maybe_scalar(preclose_base_arm_delta),
                "preclose_diagnostics_path": preclose_diagnostics_path,
                "steps": f"pre={preclose_count},policy={policy_steps}/{env_runner.n_action_steps}",
                "video_path": video_path,
            }
            append_per_episode_csv_row(csv_path, row)
            print(row, flush=True)

            per_episode.append({
                "episode_ix": episode_ix,
                "final_door_openness": maybe_scalar(metrics["final_door_openness"]),
                "final_handle_distance": maybe_scalar(metrics["final_handle_distance"]),
                "success": bool(metrics["success"]),
                "seed": episode_seed,
            })
            all_successes.append(bool(metrics["success"]))

        elapsed = time.time() - start_time
        eval_info = {
            "per_episode": per_episode,
            "aggregated": {
                "pc_success": float(np.nanmean(all_successes) * 100) if all_successes else 0.0,
            },
            "debug_eval": {
                "legacy_cvpr26": bool(args.legacy_cvpr26),
                "checkpoint_path": str(checkpoint_path),
                "policy_state_dim": policy_state_dim,
                "n_points": args.n_points,
                "use_rgb": bool(args.use_rgb),
                "use_fps": bool(args.use_fps),
                "random_drop_points": args.random_drop_points,
                "preclose_steps": args.preclose_steps,
                "preclose_gripper_value": args.preclose_gripper_value,
                "preclose_lock_base_arm": bool(args.preclose_lock_base_arm),
                "obs_gripper_value": args.obs_gripper_value,
                "action_gripper_value": args.action_gripper_value,
                "max_steps": args.max_steps,
                "episode_length_steps": int(args.max_steps) + max(int(args.preclose_steps or 0), 0),
                "episode_length_s": float(getattr(getattr(env, "_env", None), "max_episode_length_s", np.nan)),
            },
        }
        if video_paths:
            eval_info["video_paths"] = video_paths

        with (output_dir / "eval_info.json").open("w") as f:
            json.dump(eval_info, f, indent=2)

        print(f"saved eval results to {csv_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
