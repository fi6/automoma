from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
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
from lerobot.utils.io_utils import write_video
from train_dp3 import TrainDP3Workspace


PER_EPISODE_CSV_COLUMNS = [
    "episode_ix",
    "seed",
    "success",
    "video_path",
    "final_openness",
    "final_door_open",
    "final_engaged",
    "final_handle_distance",
]


def load_module(module_name: str, file_path: Path):
    spec = __import__("importlib.util").util.spec_from_file_location(module_name, file_path)
    module = __import__("importlib.util").util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


env_module = load_module("automoma_isaac_env", ISAAC_ENV_ROOT / "env.py")

CAMERA_CHOICES = ("ego_topdown", "ego_wrist", "fix_local")


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
    parser.add_argument("--checkpoint_num", type=int, default=3000)
    parser.add_argument("--camera_view", choices=CAMERA_CHOICES, default="ego_topdown")
    parser.add_argument("--n_points", type=int, default=1024)
    parser.add_argument("--random_drop_points", type=int, default=5000)
    parser.add_argument("--use_fps", type=str2bool, default=True)
    parser.add_argument("--use_rgb", type=str2bool, default=False)
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
    parser.add_argument("--headless", type=str2bool, default=True)
    parser.add_argument("--max_episodes_rendered", type=int, default=10)
    parser.add_argument("--debug_visualize_handle", type=str2bool, default=False)
    parser.add_argument("--debug_record_handle_diagnostics", type=str2bool, default=False)
    parser.add_argument("--handle_distance_threshold", type=float, default=0.1)
    return parser.parse_args()


def make_cfg(args: argparse.Namespace):
    config_path = DP3_DIFFUSION_ROOT / "diffusion_policy_3d" / "config"
    with __import__("hydra").initialize_config_dir(config_dir=str(config_path), version_base="1.2"):
        cfg = __import__("hydra").compose(config_name="robot_dp3.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.task_name = args.task_name
    cfg.expert_data_num = args.expert_data_num
    cfg.setting = args.ckpt_setting
    cfg.raw_task_name = args.task_name
    cfg.policy.use_pc_color = args.use_rgb
    cfg.task.shape_meta.obs.agent_pos.shape = [12]
    cfg.task.shape_meta.action.shape = [12]
    OmegaConf.set_struct(cfg, True)
    return cfg


def load_policy(cfg, args: argparse.Namespace):
    workspace = TrainDP3Workspace(cfg)
    usr_args = {
        "task_name": args.task_name,
        "ckpt_setting": args.ckpt_setting,
        "expert_data_num": args.expert_data_num,
        "seed": args.seed,
        "checkpoint_num": args.checkpoint_num,
        "output_dir": str(Path(args.checkpoint_root).resolve()) if args.checkpoint_root else None,
    }
    policy, env_runner = workspace.get_policy_and_runner(cfg, usr_args)
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
    traj_file = Path(args.traj_file) if args.traj_file else REPO_ROOT / "data" / "trajs" / "summit_franka" / object_name / scene_name / "test" / "traj_data_test.pt"
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
        episode_length=300,
        object_name=object_name,
        scene_name=scene_name,
        object_center=True,
        mobile_base_relative=True,
        traj_file=str(traj_file),
        traj_seed=args.traj_seed,
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
    return env_map["summit_franka_open_door_eval"][0]


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
        "final_openness": float(pick("final_openness")),
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

    cfg = make_cfg(args)
    policy, env_runner = load_policy(cfg, args)
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

    all_episode_metrics: list[dict[str, object]] = []
    all_successes: list[bool] = []
    all_seeds: list[int] = []
    video_paths: list[str] = []

    try:
        for episode_ix in range(args.n_episodes):
            episode_seed = args.seed + episode_ix
            obs, _info = env.reset(seed=episode_seed)
            env_runner.reset_obs()

            frames: list[np.ndarray] = []
            if episode_ix < args.max_episodes_rendered:
                frame = render_frame(env)
                if frame is not None:
                    frames.append(frame)

            done = False
            final_info = {"is_success": np.array([False])}
            steps = 0
            while not done and steps < 300:
                point_cloud = extract_point_cloud(obs, args.camera_view, pc_cfg, rng)
                joint_pos = extract_joint_pos(obs)
                dp3_obs = {
                    "point_cloud": point_cloud.astype(np.float32),
                    "agent_pos": joint_pos.astype(np.float32),
                }

                if len(env_runner.obs) == 0:
                    env_runner.update_obs(dp3_obs)
                    actions = env_runner.get_action(policy)
                else:
                    actions = env_runner.get_action(policy, dp3_obs)
                for action in actions:
                    obs, _reward, terminated, truncated, info = env.step(action[None, :])
                    steps += 1
                    done = bool(terminated[0] or truncated[0])
                    final_info = info.get("final_info", final_info)
                    if episode_ix < args.max_episodes_rendered:
                        frame = render_frame(env)
                        if frame is not None:
                            frames.append(frame)
                    point_cloud = extract_point_cloud(obs, args.camera_view, pc_cfg, rng)
                    joint_pos = extract_joint_pos(obs)
                    env_runner.update_obs(
                        {
                            "point_cloud": point_cloud.astype(np.float32),
                            "agent_pos": joint_pos.astype(np.float32),
                        }
                    )
                    if done or steps >= 300:
                        break

            metrics = get_success_metrics(final_info)
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
                "video_path": video_path,
                "final_openness": maybe_scalar(metrics["final_openness"]),
                "final_door_open": maybe_scalar(metrics["final_door_open"]),
                "final_engaged": maybe_scalar(metrics["final_engaged"]),
                "final_handle_distance": maybe_scalar(metrics["final_handle_distance"]),
            }
            append_per_episode_csv_row(csv_path, row)
            print(row, flush=True)

            all_episode_metrics.append({
                "final_openness": maybe_scalar(metrics["final_openness"]),
                "final_door_open": bool(metrics["final_door_open"]),
                "final_engaged": bool(metrics["final_engaged"]),
                "final_handle_distance": maybe_scalar(metrics["final_handle_distance"]),
            })
            all_successes.append(bool(metrics["success"]))
            all_seeds.append(episode_seed)

        elapsed = time.time() - start_time
        eval_info = {
            "per_episode": [
                {
                    "episode_ix": i,
                    "success": success,
                    "seed": seed,
                    **episode_metrics,
                }
                for i, (episode_metrics, success, seed) in enumerate(
                    zip(all_episode_metrics, all_successes, all_seeds, strict=True)
                )
            ],
            "aggregated": {
                "pc_success": float(np.nanmean(all_successes) * 100) if all_successes else 0.0,
                "eval_s": elapsed,
                "eval_ep_s": elapsed / args.n_episodes if args.n_episodes else 0.0,
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
