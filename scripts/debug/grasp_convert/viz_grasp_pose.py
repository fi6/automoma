#!/usr/bin/env python3
# Copyright (c) 2024-2026, AutoMoMa Authors. All rights reserved.
# SPDX-License-Identifier: MIT
"""Interactive IsaacSim viewer for AutoMoMa grasp poses.

The script loads grasp poses from an object's ``grasp/XXXX.npy`` files, solves
IK live for the selected pose, and lets the user page through them in a small
IsaacLab-Arena scene. It is intended for inspecting wrist-camera placement and
gripper roll conventions before changing grasp assets.

Example:
    conda activate lerobot-arena
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

    python scripts/debug/grasp_convert/viz_grasp_pose.py \\
        --camera \\
        --object-id 7221 \\
        --grasp-ids 0-19 \\
        --ik-seeds 64 \\
        summit_franka_grasp_viz \\
        --object_name microwave_7221 \\
        --scene_name scene_0_seed_0 \\
        --object_center

Controls:
    N / Right / Down: next grasp
    B / Left / Up: previous grasp
    R: refresh current grasp
    Q: quit
"""

from __future__ import annotations

import argparse
import contextlib
import os
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import (
    ExampleEnvironments,
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)
from isaaclab_arena.examples.example_environments.example_environment_base import (
    ExampleEnvironmentBase,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("AUTOMOMA_OBJECT_ROOT", str(REPO_ROOT / "assets" / "object"))
os.environ.setdefault("AUTOMOMA_SCENE_ROOT", str(REPO_ROOT / "assets" / "scene" / "infinigen" / "kitchen_1130"))
os.environ.setdefault("AUTOMOMA_ROBOT_ROOT", str(REPO_ROOT / "assets" / "robot"))


class SummitFrankaGraspVizEnvironment(ExampleEnvironmentBase):
    """Lightweight object+robot scene for grasp inspection."""

    name = "summit_franka_grasp_viz"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.assets.automoma_object_library import (
            get_automoma_object,
            get_object_pose_from_metadata,
            load_scene_metadata,
        )
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask
        from isaaclab_arena.utils.pose import Pose

        object_name = args_cli.object_name
        scene_name = args_cli.scene_name
        object_center = getattr(args_cli, "object_center", False)
        asset_id = object_name.split("_")[-1]
        asset_type = "_".join(object_name.split("_")[:-1]).capitalize()

        metadata = load_scene_metadata(scene_name)
        object_pose, object_scale = get_object_pose_from_metadata(
            metadata,
            asset_type=asset_type,
            asset_id=asset_id,
        )
        if object_center:
            object_pose = Pose(
                position_xyz=(0.0, 0.0, object_pose.position_xyz[2]),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )

        target_object = get_automoma_object(
            asset_type=asset_type,
            asset_id=asset_id,
            scale=object_scale,
        )
        target_object.set_initial_pose(object_pose)

        embodiment = self.asset_registry.get_asset_by_name("summit_franka")(
            enable_cameras=args_cli.enable_cameras
        )
        if args_cli.enable_cameras and embodiment.camera_config is not None:
            embodiment.camera_config.fix_local.prim_path = f"{target_object.get_prim_path()}/fix_local"

        embodiment.set_initial_pose(
            Pose(
                position_xyz=(object_pose.position_xyz[0], object_pose.position_xyz[1], 0.0),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        task = OpenDoorTask(
            target_object,
            openness_threshold=0.3,
            reset_openness=0.0,
            episode_length_s=3600.0,
            debug_visualize_handle=True,
        )
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[target_object]),
            task=task,
            teleop_device=None,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object_name", type=str, default="microwave_7221")
        parser.add_argument("--scene_name", type=str, default="scene_0_seed_0")
        parser.add_argument("--object_center", action="store_true", default=False)


ExampleEnvironments[SummitFrankaGraspVizEnvironment.name] = SummitFrankaGraspVizEnvironment


def _parse_grasp_ids(spec: str) -> list[int]:
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid grasp range: {part}")
            ids.extend(range(start, end + 1))
        else:
            ids.append(int(part))
    seen: set[int] = set()
    unique = []
    for grasp_id in ids:
        if grasp_id not in seen:
            seen.add(grasp_id)
            unique.append(grasp_id)
    return unique


def _default_object_id(object_name: str) -> str:
    return object_name.split("_")[-1]


def _default_asset_type(object_name: str) -> str:
    return "_".join(object_name.split("_")[:-1]).capitalize()


parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--robot-name", default="summit_franka", help="Robot name in configs/plan.yaml.")
parser.add_argument("--object-id", default=None, help="Object asset id, e.g. 7221. Defaults to suffix of object_name.")
parser.add_argument("--asset-type", default=None, help="Object asset type, e.g. Microwave. Defaults from object_name.")
parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "plan.yaml"), help="Planning config path.")
parser.add_argument("--grasp-ids", default="0-19", help="Comma/range list, e.g. 0-19 or 0,2,8.")
parser.add_argument("--grasp-dir", default=None, help="Override grasp directory. Defaults to object_dir/grasp.")
parser.add_argument("--object-dir", default=None, help="Override object asset directory.")
parser.add_argument("--ik-solution-index", type=int, default=0, help="Live IK solution index to visualize.")
parser.add_argument("--ik-seeds", type=int, default=64, help="Number of cuRobo IK seeds per displayed grasp.")
parser.add_argument("--ik-grad-iters", type=int, default=100, help="Gradient IK iterations per displayed grasp.")
parser.add_argument("--closed-gripper", type=float, default=0.0, help="Finger joint value for grasp display.")
parser.add_argument("--open-gripper", action="store_true", help="Show gripper open instead of closed.")
parser.add_argument("--object-joint", type=float, default=0.0, help="Object joint value for display.")
parser.add_argument("--hold-frames", type=int, default=4, help="Render/update frames after each state write.")
parser.add_argument("--once", action="store_true", help="Display the first requested grasp once and exit.")
parser.add_argument("--save-all", action="store_true", help="Display every requested grasp once and exit.")
parser.add_argument("--camera", action="store_true", help="Enable ego_topdown, ego_wrist, and fix_local cameras.")
parser.add_argument(
    "--save-camera-dir",
    default=None,
    help="Optional directory to save a contact sheet of camera RGB images for each displayed grasp.",
)
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

if args_cli.camera:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from yourdfpy import URDF  # noqa: E402

import isaaclab.envs.mdp as mdp_isaac_lab  # noqa: E402
import carb.input  # noqa: E402
import omni.appwindow  # noqa: E402
from curobo.types.base import TensorDeviceType  # noqa: E402
from curobo.types.math import Pose  # noqa: E402
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig  # noqa: E402

from automoma.utils.file_utils import load_object_from_metadata, load_robot_cfg, process_robot_cfg  # noqa: E402
from automoma.utils.math_utils import get_open_ee_pose  # noqa: E402
from isaaclab_arena.embodiments.summit_franka.summit_franka import (  # noqa: E402
    SummitFrankaJointSpaceActionsCfg,
)
from isaaclab_arena.utils.sim_utils import (  # noqa: E402
    deactivate_prims_by_name,
    disable_all_collisions,
    set_lighting_mode,
    sync_cameras_after_reset,
)


@dataclass
class GraspDisplayState:
    robot_joints: torch.Tensor | None
    object_joints: torch.Tensor
    source: str
    message: str = ""


class GraspStateResolver:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = self._load_plan_cfg(args.config)
        self.object_name = args.object_name
        self.object_id = args.object_id or _default_object_id(args.object_name)
        self.asset_type = args.asset_type or _default_asset_type(args.object_name)
        self.obj_meta = self.cfg["objects"][self.object_id]
        self.scale = float(self.obj_meta.get("scale", 1.0))
        self.handle_link = self.obj_meta.get("handle_link", "link_0")
        self.joint_name = self.obj_meta.get("joint_name", "joint_0")
        self.gripper_value = 0.04 if args.open_gripper else args.closed_gripper

        object_dir = Path(args.object_dir) if args.object_dir else REPO_ROOT / "assets" / "object" / self.asset_type / self.object_id
        self.grasp_dir = Path(args.grasp_dir) if args.grasp_dir else object_dir / "grasp"
        self.object_urdf_path = REPO_ROOT / self.obj_meta["urdf_path"]
        self.object_urdf = URDF.load(str(self.object_urdf_path), build_collision_scene_graph=False)
        self.object_pose = Pose.from_list(self._resolve_object_pose())
        self.tensor_args = TensorDeviceType()
        self.robot_cfg = self._load_robot_cfg()
        self.ik_solver = self._init_ik_solver()

    @staticmethod
    def _load_plan_cfg(config_path: str) -> dict:
        cfg = OmegaConf.load(config_path)
        return OmegaConf.to_container(cfg, resolve=True)

    def resolve(self, grasp_id: int) -> GraspDisplayState:
        grasp_path = self.grasp_dir / f"{grasp_id:04d}.npy"
        object_joints = torch.tensor([self.args.object_joint], dtype=torch.float32)
        if not grasp_path.exists():
            return GraspDisplayState(None, object_joints, "missing", f"Missing grasp file: {grasp_path}")

        grasp_raw = np.load(grasp_path).copy()
        grasp_raw[:3] *= self.scale
        grasp_pose = Pose.from_list(grasp_raw.tolist())
        default_joint_cfg = {self.joint_name: 0.0}
        target = get_open_ee_pose(
            self.object_pose,
            grasp_pose,
            self.object_urdf,
            self.handle_link,
            default_joint_cfg,
            default_joint_cfg,
        )
        iks = self._solve_ik(target)
        if iks is None or iks.shape[0] == 0:
            return GraspDisplayState(None, object_joints, "solve-empty", f"No IK solution for {grasp_path}")
        idx = min(max(self.args.ik_solution_index, 0), iks.shape[0] - 1)
        robot_joints = self._to_isaac_robot_joints(iks[idx].detach().cpu())
        return GraspDisplayState(robot_joints, object_joints, f"live-ik:{iks.shape[0]} solutions")

    def _resolve_object_pose(self) -> list[float]:
        scene_cfg = self.cfg.get("scene", {})
        scene_type = next(iter(scene_cfg))
        scene_defaults = scene_cfg[scene_type]
        metadata_subpath = scene_defaults.get("metadata_subpath", "info/metadata.json")
        metadata_path = Path(self.cfg.get("scene_dir", "")) / self.args.scene_name / metadata_subpath
        obj_cfg = {
            "path": self.obj_meta["urdf_path"],
            "asset_type": self.asset_type,
            "asset_id": self.object_id,
        }
        obj_cfg = load_object_from_metadata(str(metadata_path), obj_cfg)
        pose = list(obj_cfg["pose"])
        if getattr(self.args, "object_center", False):
            pose = [0.0, 0.0, pose[2], 1.0, 0.0, 0.0, 0.0]
        return pose

    def _load_robot_cfg(self) -> dict:
        robot_cfg_path = self.cfg["robot"][self.args.robot_name]["config_path"]
        return process_robot_cfg(load_robot_cfg(robot_cfg_path))

    def _init_ik_solver(self) -> IKSolver:
        print("[viz_grasp_pose] Initializing cuRobo IK solver...", flush=True)
        ik_cfg = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            None,
            self.tensor_args,
            num_seeds=int(self.args.ik_seeds),
            position_threshold=0.005,
            rotation_threshold=0.05,
            use_cuda_graph=False,
            collision_cache=None,
            collision_checker_type=None,
            self_collision_check=False,
            self_collision_opt=False,
            use_particle_opt=False,
            grad_iters=int(self.args.ik_grad_iters),
        )
        print("[viz_grasp_pose] IK solver ready.", flush=True)
        return IKSolver(ik_cfg)

    def _solve_ik(self, target: Pose) -> torch.Tensor | None:
        retract = self.tensor_args.to_device(self.robot_cfg["kinematics"]["cspace"]["retract_config"])
        print("[viz_grasp_pose] Solving IK...", flush=True)
        start_time = time.time()
        result = self.ik_solver.solve_single(
            goal_pose=target,
            retract_config=retract.unsqueeze(0),
            return_seeds=int(self.args.ik_seeds),
            num_seeds=int(self.args.ik_seeds),
            use_nn_seed=False,
            link_poses=None,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sol = result.get_unique_solution()
        print(
            f"[viz_grasp_pose] IK solve done in {time.time() - start_time:.3f}s: "
            f"{0 if sol is None else sol.shape[0]} unique solutions.",
            flush=True,
        )
        return sol

    def _to_isaac_robot_joints(self, ik: torch.Tensor) -> torch.Tensor:
        if ik.ndim != 1:
            ik = ik.flatten()
        if ik.numel() >= 12:
            robot = ik[:12].clone()
            robot[10] = self.gripper_value
            robot[11] = self.gripper_value
            return robot.float()
        if ik.numel() < 10:
            raise ValueError(f"Expected at least 10 IK values, got {ik.numel()}")
        gripper = torch.tensor([self.gripper_value, self.gripper_value], dtype=ik.dtype)
        return torch.cat([ik[:10], gripper], dim=0).float()


class GraspViewer:
    def __init__(self, env, resolver: GraspStateResolver, grasp_ids: list[int], args: argparse.Namespace):
        self.env = env
        self.resolver = resolver
        self.grasp_ids = grasp_ids
        self.args = args
        self.index = 0
        self.dirty = True
        self.quit = False
        self.obs = None

    @property
    def current_grasp_id(self) -> int:
        return self.grasp_ids[self.index]

    def next(self) -> None:
        self.index = (self.index + 1) % len(self.grasp_ids)
        self.dirty = True

    def prev(self) -> None:
        self.index = (self.index - 1) % len(self.grasp_ids)
        self.dirty = True

    def refresh(self) -> None:
        self.dirty = True

    def request_quit(self) -> None:
        self.quit = True

    def write_current_state(self) -> None:
        grasp_id = self.current_grasp_id
        state = self.resolver.resolve(grasp_id)
        print(
            f"[grasp {grasp_id:04d}] {self.index + 1}/{len(self.grasp_ids)} "
            f"source={state.source}"
        )
        if state.message:
            print(f"  {state.message}")
        if state.robot_joints is None:
            return

        robot = self.env.scene["robot"]
        robot_pos = state.robot_joints.unsqueeze(0).to(self.env.device)
        robot_vel = torch.zeros_like(robot_pos)
        robot.write_joint_state_to_sim(robot_pos, robot_vel)

        self._write_object_state(state.object_joints)
        self.env.scene.write_data_to_sim()
        self._render_frames()
        self._save_camera_sheet(grasp_id)

    def _write_object_state(self, object_joints: torch.Tensor) -> None:
        for key in self.env.scene.keys():
            if key == "robot":
                continue
            entity = self.env.scene[key]
            if hasattr(entity, "write_joint_state_to_sim") and hasattr(entity, "num_joints"):
                n_joints = min(object_joints.shape[0], entity.num_joints)
                pos = object_joints[:n_joints].unsqueeze(0).to(self.env.device)
                vel = torch.zeros_like(pos)
                entity.write_joint_state_to_sim(pos, vel)
                return

    def _render_frames(self) -> None:
        for _ in range(max(1, self.args.hold_frames)):
            self.env.scene.write_data_to_sim()
            if hasattr(self.env.sim, "render"):
                self.env.sim.render()
            self.env.scene.update(self.env.physics_dt)
            simulation_app.update()
        if self.args.camera:
            self.obs = sync_cameras_after_reset(self.env)

    def _save_camera_sheet(self, grasp_id: int) -> None:
        if not self.args.save_camera_dir or self.obs is None:
            return
        from PIL import Image

        camera_obs = self.obs.get("camera_obs", {}) if isinstance(self.obs, dict) else {}
        rgb_keys = [k for k in ("ego_topdown_rgb", "ego_wrist_rgb", "fix_local_rgb") if k in camera_obs]
        if not rgb_keys:
            print("  No RGB camera observations found to save.")
            return
        images = []
        for key in rgb_keys:
            img = camera_obs[key]
            if torch.is_tensor(img):
                img = img.detach().cpu().numpy()
            img = np.asarray(img)
            if img.ndim == 4:
                img = img[0]
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            images.append(Image.fromarray(img[..., :3]))
        width = sum(img.width for img in images)
        height = max(img.height for img in images)
        sheet = Image.new("RGB", (width, height))
        x = 0
        for img in images:
            sheet.paste(img, (x, 0))
            x += img.width
        out_dir = Path(self.args.save_camera_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"grasp_{grasp_id:04d}.png"
        sheet.save(out_path)
        print(f"  Saved camera sheet: {out_path}")


class OmniKeyboardControls:
    def __init__(self, viewer: GraspViewer):
        self.viewer = viewer
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        key = event.input.name
        if key in ("N", "RIGHT", "DOWN"):
            self.viewer.next()
        elif key in ("B", "LEFT", "UP"):
            self.viewer.prev()
        elif key == "R":
            self.viewer.refresh()
        elif key in ("Q", "ESCAPE"):
            self.viewer.request_quit()
        return True

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)


class TerminalKeyboardControls:
    def __init__(self, viewer: GraspViewer):
        self.viewer = viewer
        self._running = False
        self._thread: threading.Thread | None = None
        self._old_attrs = None
        if not sys.stdin.isatty():
            return
        self._old_attrs = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running and not self.viewer.quit:
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                seq = ch + sys.stdin.read(2)
                if seq in ("\x1b[B", "\x1b[C"):
                    self.viewer.next()
                elif seq in ("\x1b[A", "\x1b[D"):
                    self.viewer.prev()
                continue
            key = ch.lower()
            if key == "n":
                self.viewer.next()
            elif key == "b":
                self.viewer.prev()
            elif key == "r":
                self.viewer.refresh()
            elif key == "q":
                self.viewer.request_quit()

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        if self._old_attrs is not None:
            with contextlib.suppress(Exception):
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_attrs)


def _build_env(args):
    arena_builder = get_arena_builder_from_cli(args)
    env_name, env_cfg = arena_builder.build_registered()

    env_cfg.actions = SummitFrankaJointSpaceActionsCfg()
    env_cfg.observations.policy.joint_pos = mdp_isaac_lab.ObservationTermCfg(func=mdp_isaac_lab.joint_pos)
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    env_cfg.recorders = {}

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    if getattr(args, "object_name", None):
        deactivate_prims_by_name(args.object_name, exclude_paths=(), required_path_substrings=("/scene/",))
    set_lighting_mode(2)
    disable_all_collisions()
    return env


def main() -> int:
    if args_cli.robot_name != "summit_franka":
        raise ValueError("viz_grasp_pose.py currently supports robot-name=summit_franka only.")

    grasp_ids = _parse_grasp_ids(args_cli.grasp_ids)
    if not grasp_ids:
        raise ValueError("--grasp-ids resolved to an empty list")

    print("Launching grasp viewer")
    print(f"  object_name={args_cli.object_name}")
    print(f"  scene_name={args_cli.scene_name}")
    print(f"  grasp_ids={grasp_ids}")
    print(f"  cameras={'on' if args_cli.enable_cameras else 'off'}")

    env = _build_env(args_cli)
    resolver = GraspStateResolver(args_cli)
    viewer = GraspViewer(env, resolver, grasp_ids, args_cli)

    env.reset()
    if args_cli.camera:
        sync_cameras_after_reset(env)

    controls = []
    if not args_cli.headless and os.environ.get("HEADLESS", "0") in ("0", ""):
        controls.append(OmniKeyboardControls(viewer))
    terminal_controls = TerminalKeyboardControls(viewer)
    if terminal_controls._running:
        controls.append(terminal_controls)

    print("Controls: N/Right/Down=next, B/Left/Up=prev, R=refresh, Q/Esc=quit")
    with contextlib.suppress(KeyboardInterrupt):
        while simulation_app.is_running() and not simulation_app.is_exiting() and not viewer.quit:
            if viewer.dirty:
                viewer.write_current_state()
                if args_cli.save_all:
                    if viewer.index >= len(viewer.grasp_ids) - 1:
                        break
                    viewer.next()
                else:
                    viewer.dirty = False
                if args_cli.once and not args_cli.save_all:
                    break
            simulation_app.update()
            time.sleep(0.01)

    for control in controls:
        control.close()
    env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
