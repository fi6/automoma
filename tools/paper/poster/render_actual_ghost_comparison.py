#!/usr/bin/env python3
"""Render actual planned trajectory ghosts for the poster workspace comparison.

This script is intentionally separate from ``render_reach_comparison.py``:
it reads the already-planned iTHOR trajectories on disk and renders two actual
3D ghost panels with the same scene/camera:

* fixed-base Summit-Franka, with the Summit base visuals hidden
* mobile Summit-Franka, with colored solved base trajectories plus faint
  workspace ghosts
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from render_reach_comparison import (
    DOME_LIGHT_INTENSITY,
    FINGER_OPEN_RANGE,
    FRANKA_WORKSPACE_JOINT_RANGES,
    ROBOT_OUTPUT_NAMES,
    SCENE_RENDER_CONFIGS,
    add_repo_import_paths,
    bind_material_to_meshes,
    camera_for_view,
    check_inputs,
    create_camera,
    disable_collisions_under_prim,
    euler_deg_to_quat_wxyz,
    load_scene_specs,
    load_summit_defaults_from_yml,
    load_summit_joint_limits,
    load_yaml,
    make_contact_sheets,
    make_preview_material,
    render_one,
    repo_root,
    resolve_repo_path,
    sample_summit,
    save_rgb_image,
    spawn_static_scene,
    stable_seed,
    set_camera_view,
    set_prim_visibility,
    yaw_to_quat_wxyz,
)


DEFAULT_DATA_ROOT = (
    "/media/xinhai/GIANT/Research/AutoMoMa/data/ithor/data_250917/ithor_floorplan1_1"
)
DEFAULT_SCENE = "ithor_floorplan1_1"
DEFAULT_OBJECT_ID = "7221"
DEFAULT_OUTPUT_ROOT = "outputs/paper/poster/actual_ghost_comparison"
DEFAULT_TRAJECTORY_DISPLAY_OFFSETS = {
    # The old 7221 replay/HDF5 joint coordinates do not line up one-to-one with
    # the poster FloorPlan1 USD frame.  Keep this as a poster-only display
    # offset so the actual solved roots match the recorded top-down view near
    # the upper-right microwave without changing any planner/replay data.
    ("ithor_floorplan1_1", "7221"): (0.0, -1.56),
}

IMAGE_WIDTH = 2400
IMAGE_HEIGHT = 1600
RENDER_SETTLE_STEPS = 12

MOBILE_TRAJ_COUNT = 5
MOBILE_KEYFRAMES = 5
MOBILE_WORKSPACE_GHOSTS = 22
FIXED_WORKSPACE_GHOSTS = 18
FIXED_KEYFRAMES = 7
FIXED_ARM_GHOSTS = 22

GHOST_GREY = (0.78, 0.80, 0.78)
MOBILE_COLORS = [
    (0.00, 0.55, 0.52),
    (0.10, 0.45, 0.78),
    (0.92, 0.48, 0.08),
    (0.65, 0.34, 0.72),
    (0.24, 0.62, 0.28),
]
FIXED_COLOR = (0.90, 0.32, 0.13)
MOBILE_TRAJ_OPACITY = 0.34
MOBILE_WORKSPACE_OPACITY = 0.012
FIXED_WORKSPACE_OPACITY = 0.72
FIXED_TRAJ_OPACITY = 0.34
FIXED_RANDOM_OPACITY = 0.026
SOURCE_BACKGROUND_COLOR = (0.0, 1.0, 0.0)
SOURCE_BACKGROUND_SIZE = 80.0
SOURCE_BACKGROUND_DISTANCE = 30.0

SCENE_FOREGROUND_OCCLUDER_PATHS = [
    # Front wall/door occluders from the original iTHOR scene USD.
    "/World/Structure/FloorPlan1/Wall",
    "/World/Structure/FloorPlan1/Door",
    "/World/Structure/FloorPlan1/Door1",
    "/World/Structure/FloorPlan1/DoorFrame",
    "/World/Structure/FloorPlan1/DoorFrame1",
    "/World/Objects/LightSwitch_cfaab6e3",
    # Island/tabletop objects that should occlude robot-only layers later.
    "/World/Structure/FloorPlan1/StandardIslandHeight",
    "/World/Objects/Bowl_208f368b",
    "/World/Objects/Tomato_caaae6b0",
    "/World/Objects/ButterKnife_4ae287b7",
    "/World/Objects/Bread_c6b4566e",
    "/World/Objects/Apple_3fef4551",
    "/World/Objects/Book_e5ef3174",
    "/World/Structure/PaperClutter1",
    "/World/Objects/CreditCard_5e829d70",
]


@dataclasses.dataclass
class GhostSample:
    name: str
    joint_pos: dict[str, float]
    color: tuple[float, float, float]
    opacity: float
    hide_summit_base: bool = False
    urdf_path: str | None = None


def trajectory_dir(data_root: Path, robot_name: str, object_id: str) -> Path:
    return data_root / robot_name / object_id / "0" / "ik_traj_mobile_filter"


def load_filtered_trajs(data_root: Path, robot_name: str, object_id: str) -> torch.Tensor:
    """Load and concatenate successful raw cuRobo trajectories."""

    directory = trajectory_dir(data_root, robot_name, object_id)
    if not directory.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {directory}")

    chunks: list[torch.Tensor] = []
    for path in sorted(directory.glob("traj_mobile_*.pt")):
        data = torch.load(path, map_location="cpu")
        traj = data["traj_result"].float()
        success = data.get("success")
        if success is not None:
            traj = traj[success.bool()]
        if traj.shape[0] == 0:
            continue
        chunks.append(traj)
    if not chunks:
        raise RuntimeError(f"No successful trajectories found under {directory}")
    return torch.cat(chunks, dim=0)


def local_robot_cfg_path(root: Path, filename: str) -> Path:
    local = root / "tools" / "paper" / "poster" / "local" / "robots" / filename
    if local.exists():
        return local
    return root / "assets" / "robot" / "summit_franka" / filename


def ensure_local_urdfs(root: Path) -> tuple[Path, Path]:
    """Create poster-local URDF variants without touching global assets."""

    import xml.etree.ElementTree as ET

    source = root / "assets" / "robot" / "summit_franka" / "summit_franka.urdf"
    local_dir = root / "tools" / "paper" / "poster" / "local" / "robots"
    local_dir.mkdir(parents=True, exist_ok=True)
    mobile_urdf = local_dir / "summit_franka_full_base.urdf"
    fixed_urdf = local_dir / "summit_franka_empty_base.urdf"

    if not mobile_urdf.exists() or source.stat().st_mtime > mobile_urdf.stat().st_mtime:
        mobile_urdf.write_text(source.read_text(), encoding="utf-8")

    if not fixed_urdf.exists() or source.stat().st_mtime > fixed_urdf.stat().st_mtime:
        tree = ET.parse(source)
        xml_root = tree.getroot()
        for link in xml_root.findall("link"):
            if link.attrib.get("name") != "summit_base":
                continue
            for child in list(link):
                if child.tag in {"visual", "collision"}:
                    link.remove(child)
        tree.write(fixed_urdf, encoding="utf-8", xml_declaration=True)

    return mobile_urdf, fixed_urdf


def cfg_fixed_base_values(fixed_cfg_path: Path) -> tuple[float, float, float]:
    cfg = load_yaml(fixed_cfg_path)
    locks = cfg["robot_cfg"]["kinematics"].get("lock_joints", {})
    return (
        float(locks.get("base_x", 0.0)),
        float(locks.get("base_y", 0.0)),
        float(locks.get("base_z", 0.0)),
    )


def recorded_episode_paths(data_root: Path, robot_name: str, object_id: str) -> list[Path]:
    root = data_root / robot_name / object_id / "0" / "collect_data"
    if not root.exists():
        return []
    return sorted(root.glob("**/episode*.hdf5"))


def load_recorded_fixed_base(
    data_root: Path,
    robot_name: str,
    object_id: str,
) -> tuple[float, float, float] | None:
    """Read the fixed-base pose that IsaacLab actually recorded.

    The fixed-base ``traj_mobile_*.pt`` files only contain arm + object-joint
    states.  The robot root used for the old iTHOR runs is stored in the replay
    HDF5 files, so prefer it over any stale checked-in YAML lock joints.
    """

    try:
        import h5py
    except Exception as exc:
        print(f"[ghost] WARNING: h5py unavailable; cannot infer recorded fixed base: {exc}")
        return None

    samples: list[np.ndarray] = []
    for path in recorded_episode_paths(data_root, robot_name, object_id):
        try:
            with h5py.File(path, "r") as f:
                if "obs/joint/mobile_base" not in f:
                    continue
                mobile_base = np.asarray(f["obs/joint/mobile_base"][:], dtype=np.float64)
                if mobile_base.ndim == 2 and mobile_base.shape[1] >= 3:
                    samples.append(mobile_base[:, :3])
        except Exception as exc:
            print(f"[ghost] WARNING: failed to read fixed base from {path}: {exc}")
    if not samples:
        return None
    values = np.concatenate(samples, axis=0)
    base = np.median(values, axis=0)
    return (float(base[0]), float(base[1]), float(base[2]))


def load_recorded_fixed_arm_trajs(
    data_root: Path,
    robot_name: str,
    object_id: str,
) -> torch.Tensor | None:
    """Load fixed-base arm trajectories from recorded HDF5 demos when present."""

    try:
        import h5py
    except Exception as exc:
        print(f"[ghost] WARNING: h5py unavailable; cannot load recorded fixed arms: {exc}")
        return None

    arms: list[torch.Tensor] = []
    for path in recorded_episode_paths(data_root, robot_name, object_id):
        try:
            with h5py.File(path, "r") as f:
                if "obs/joint/arm" not in f:
                    continue
                arm = torch.as_tensor(np.asarray(f["obs/joint/arm"][:]), dtype=torch.float32)
                if arm.ndim == 2 and arm.shape[1] >= 7:
                    arms.append(arm[:, :7])
        except Exception as exc:
            print(f"[ghost] WARNING: failed to read fixed arm trajectory from {path}: {exc}")
    if not arms:
        return None
    lengths = {int(arm.shape[0]) for arm in arms}
    if len(lengths) != 1:
        min_len = min(lengths)
        arms = [arm[:min_len] for arm in arms]
    return torch.stack(arms, dim=0)


def fixed_base_values(
    fixed_cfg_path: Path,
    data_root: Path,
    robot_name: str,
    object_id: str,
) -> tuple[tuple[float, float, float], str]:
    recorded = load_recorded_fixed_base(data_root, robot_name, object_id)
    if recorded is not None:
        return recorded, "recorded_hdf5_obs_joint_mobile_base"
    return cfg_fixed_base_values(fixed_cfg_path), "robot_cfg_lock_joints"


def summit_joint_names() -> list[str]:
    return [
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
    ]


def mobile_state_to_joints(
    state: torch.Tensor,
    *,
    grip: float = 0.0,
    xy_offset: tuple[float, float] = (0.0, 0.0),
) -> dict[str, float]:
    values = state.detach().cpu().float().tolist()
    if len(values) < 10:
        raise ValueError(f"Mobile trajectory state must have at least 10 robot values, got {len(values)}")
    values[0] += float(xy_offset[0])
    values[1] += float(xy_offset[1])
    robot = values[:10] + [grip, grip]
    return dict(zip(summit_joint_names(), map(float, robot), strict=True))


def fixed_state_to_joints(
    state: torch.Tensor,
    fixed_base: tuple[float, float, float],
    *,
    grip: float = 0.0,
) -> dict[str, float]:
    values = state.detach().cpu().float().tolist()
    if len(values) < 7:
        raise ValueError(f"Fixed trajectory state must have at least 7 arm values, got {len(values)}")
    robot = list(fixed_base) + values[:7] + [grip, grip]
    return dict(zip(summit_joint_names(), map(float, robot), strict=True))


def farthest_indices(features: np.ndarray, count: int) -> list[int]:
    if features.shape[0] == 0:
        return []
    count = min(count, features.shape[0])
    scaled = features.astype(float)
    span = np.ptp(scaled, axis=0)
    scaled = (scaled - scaled.mean(axis=0)) / np.maximum(span, 1.0e-6)
    center = np.median(scaled, axis=0)
    selected = [int(np.argmax(np.linalg.norm(scaled - center, axis=1)))]
    min_dist = np.linalg.norm(scaled - scaled[selected[0]], axis=1)
    for _ in range(1, count):
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(scaled - scaled[idx], axis=1))
    return selected


def representative_mobile_indices(trajs: torch.Tensor, count: int) -> list[int]:
    base = trajs[:, :, :3].numpy()
    displacement = np.linalg.norm(base[:, -1, :2] - base[:, 0, :2], axis=1, keepdims=True)
    features = np.concatenate([base[:, 0, :3], base[:, -1, :3], displacement], axis=1)
    return farthest_indices(features, count)


def representative_fixed_index(trajs: torch.Tensor) -> int:
    arm = trajs[:, :, :7].numpy()
    motion = np.linalg.norm(arm[:, -1, :] - arm[:, 0, :], axis=1)
    return int(np.argsort(motion)[len(motion) // 2])


def keyframe_indices(num_steps: int, count: int) -> list[int]:
    if count <= 1:
        return [num_steps - 1]
    return sorted(set(int(round(v)) for v in np.linspace(0, num_steps - 1, count)))


def build_mobile_trajectory_samples(
    trajs: torch.Tensor,
    traj_count: int,
    keyframes: int,
    xy_offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[list[GhostSample], list[dict[str, Any]]]:
    selected = representative_mobile_indices(trajs, traj_count)
    samples: list[GhostSample] = []
    selected_meta: list[dict[str, Any]] = []
    for color_id, traj_idx in enumerate(selected):
        trajectory = trajs[traj_idx]
        color = MOBILE_COLORS[color_id % len(MOBILE_COLORS)]
        frames = keyframe_indices(trajectory.shape[0], keyframes)
        selected_meta.append({"trajectory_index": traj_idx, "keyframes": frames, "color": color})
        for frame in frames:
            samples.append(
                GhostSample(
                    name=f"mobile_traj{color_id:02d}_t{frame:02d}",
                    joint_pos=mobile_state_to_joints(trajectory[frame], xy_offset=xy_offset),
                    color=color,
                    opacity=MOBILE_TRAJ_OPACITY,
                )
            )
    return samples, selected_meta


def build_fixed_trajectory_samples(
    trajs: torch.Tensor,
    fixed_base: tuple[float, float, float],
    keyframes: int,
    trajectory_index: int | None = None,
) -> tuple[list[GhostSample], dict[str, Any]]:
    traj_idx = representative_fixed_index(trajs) if trajectory_index is None else int(trajectory_index)
    trajectory = trajs[traj_idx]
    frames = keyframe_indices(trajectory.shape[0], keyframes)
    samples = [
        GhostSample(
            name=f"fixed_traj_t{frame:02d}",
            joint_pos=fixed_state_to_joints(trajectory[frame], fixed_base),
            color=FIXED_COLOR,
            opacity=FIXED_TRAJ_OPACITY,
            hide_summit_base=True,
        )
        for frame in frames
    ]
    return samples, {"trajectory_index": traj_idx, "keyframes": frames, "color": FIXED_COLOR}


def representative_fixed_indices(trajs: torch.Tensor, count: int) -> list[int]:
    arm = trajs[:, :, :7].numpy()
    displacement = np.linalg.norm(arm[:, -1, :] - arm[:, 0, :], axis=1, keepdims=True)
    features = np.concatenate([arm[:, 0, :], arm[:, -1, :], displacement], axis=1)
    return farthest_indices(features, count)


def build_fixed_episode_trajectory_samples(
    trajs: torch.Tensor,
    fixed_base: tuple[float, float, float],
    episode_count: int,
    keyframes: int,
) -> tuple[list[GhostSample], list[dict[str, Any]]]:
    samples: list[GhostSample] = []
    metas: list[dict[str, Any]] = []
    for episode_id, traj_idx in enumerate(representative_fixed_indices(trajs, episode_count)):
        trajectory = trajs[traj_idx]
        color = MOBILE_COLORS[episode_id % len(MOBILE_COLORS)]
        frames = keyframe_indices(trajectory.shape[0], keyframes)
        metas.append({"trajectory_index": traj_idx, "keyframes": frames, "color": color})
        for frame in frames:
            samples.append(
                GhostSample(
                    name=f"fixed_traj{episode_id:02d}_t{frame:02d}",
                    joint_pos=fixed_state_to_joints(trajectory[frame], fixed_base),
                    color=color,
                    opacity=FIXED_TRAJ_OPACITY,
                    hide_summit_base=True,
                )
            )
    return samples, metas


def build_fixed_random_ghosts(
    trajs: torch.Tensor,
    fixed_base: tuple[float, float, float],
    count: int,
) -> list[GhostSample]:
    rng = np.random.default_rng(stable_seed("actual_ghost_fixed_random"))
    flat = trajs[:, :, :].reshape(-1, trajs.shape[-1])
    if flat.shape[0] == 0:
        return []
    ids = rng.choice(flat.shape[0], size=min(count, flat.shape[0]), replace=False)
    return [
        GhostSample(
            name=f"fixed_random_{i:02d}",
            joint_pos=fixed_state_to_joints(flat[int(idx)], fixed_base),
            color=GHOST_GREY,
            opacity=FIXED_RANDOM_OPACITY,
            hide_summit_base=True,
        )
        for i, idx in enumerate(ids)
    ]


def build_mobile_workspace_ghosts(spec: Any, count: int, summit_cfg: Path) -> list[GhostSample]:
    raw = sample_summit(spec, count, summit_cfg)
    samples: list[GhostSample] = []
    for index, sample in enumerate(raw):
        joints = dict(sample.joint_pos)
        joints["base_x"] = float(sample.position[0])
        joints["base_y"] = float(sample.position[1])
        joints["base_z"] = float(sample.yaw)
        samples.append(
            GhostSample(
                name=f"mobile_workspace_{index:02d}",
                joint_pos=joints,
                color=GHOST_GREY,
                opacity=MOBILE_WORKSPACE_OPACITY,
            )
        )
    return samples


def build_fixed_workspace_ghosts(
    fixed_base: tuple[float, float, float],
    count: int,
    fixed_cfg: Path,
    scene_name: str,
) -> list[GhostSample]:
    if count <= 0:
        return []
    # Match the older fixed_franka/detail_robots_only.png fan: one fixed base
    # with broad, radially distributed arm poses rather than local trajectory
    # perturbations.  The pose bank is adapted from the older
    # fixed_franka/detail_robots_only.png random workspace render: joint1 is
    # swept azimuthally, while joints 2-7 vary between low, side, and raised
    # elbow configurations so the silhouette reads like a rough sphere rather
    # than a flat flower.
    rng = np.random.default_rng(stable_seed(scene_name, "franka"))
    defaults = load_summit_defaults_from_yml(fixed_cfg)
    joint1_low, joint1_high = FRANKA_WORKSPACE_JOINT_RANGES["panda_joint1"]
    joint1_angles = np.linspace(joint1_low, joint1_high, count, endpoint=True)
    if count >= 14:
        # For poster source layers it is more important to communicate the
        # envelope than to leave a small visual gap at the joint-limit seam.
        joint1_angles = np.linspace(-math.pi, math.pi, count, endpoint=False)
    workspace_pose_bank = [
        {
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.81,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
        },
        {
            "panda_joint2": -1.070,
            "panda_joint3": -0.105,
            "panda_joint4": -2.349,
            "panda_joint5": -1.117,
            "panda_joint6": 3.271,
            "panda_joint7": -0.158,
        },
        {
            "panda_joint2": -1.074,
            "panda_joint3": -1.710,
            "panda_joint4": -0.548,
            "panda_joint5": 0.529,
            "panda_joint6": 1.588,
            "panda_joint7": 2.619,
        },
        {
            "panda_joint2": 0.236,
            "panda_joint3": -2.164,
            "panda_joint4": -0.875,
            "panda_joint5": 1.209,
            "panda_joint6": 2.605,
            "panda_joint7": 0.020,
        },
        {
            "panda_joint2": 0.068,
            "panda_joint3": -2.042,
            "panda_joint4": -1.803,
            "panda_joint5": -0.314,
            "panda_joint6": 2.622,
            "panda_joint7": -0.457,
        },
        {
            "panda_joint2": 0.465,
            "panda_joint3": 1.027,
            "panda_joint4": -1.441,
            "panda_joint5": -1.165,
            "panda_joint6": 2.288,
            "panda_joint7": 2.343,
        },
        {
            "panda_joint2": 0.171,
            "panda_joint3": -0.601,
            "panda_joint4": -1.756,
            "panda_joint5": 1.139,
            "panda_joint6": 1.234,
            "panda_joint7": -1.908,
        },
        {
            "panda_joint2": -0.869,
            "panda_joint3": -1.989,
            "panda_joint4": -1.673,
            "panda_joint5": 2.259,
            "panda_joint6": 2.612,
            "panda_joint7": 2.447,
        },
        {
            "panda_joint2": -0.18,
            "panda_joint3": 0.00,
            "panda_joint4": -1.28,
            "panda_joint5": 0.00,
            "panda_joint6": 1.36,
            "panda_joint7": 0.55,
        },
        {
            "panda_joint2": -0.24,
            "panda_joint3": 0.12,
            "panda_joint4": -1.20,
            "panda_joint5": -0.14,
            "panda_joint6": 1.44,
            "panda_joint7": 0.70,
        },
        {
            "panda_joint2": -0.10,
            "panda_joint3": -0.12,
            "panda_joint4": -1.36,
            "panda_joint5": 0.14,
            "panda_joint6": 1.28,
            "panda_joint7": 0.40,
        },
    ]
    samples: list[GhostSample] = []
    for index in range(count):
        joints = dict(defaults)
        joints.update(workspace_pose_bank[index % len(workspace_pose_bank)])
        joints["panda_joint1"] = float(joint1_angles[index])
        joints["panda_finger_joint1"] = float(rng.uniform(*FINGER_OPEN_RANGE))
        joints["panda_finger_joint2"] = float(rng.uniform(*FINGER_OPEN_RANGE))
        joints["base_x"] = float(fixed_base[0])
        joints["base_y"] = float(fixed_base[1])
        joints["base_z"] = float(fixed_base[2])
        samples.append(
            GhostSample(
                name=f"fixed_workspace_{index:02d}",
                joint_pos=joints,
                color=GHOST_GREY,
                opacity=FIXED_WORKSPACE_OPACITY,
                hide_summit_base=True,
            )
        )
    return samples


def make_robot_articulation(sample: GhostSample, prim_path: str) -> Any:
    from isaaclab.assets import Articulation
    import isaaclab.sim as sim_utils
    from isaaclab_arena.embodiments.summit_franka.summit_franka import SummitFrankaSceneCfg

    robot_cfg = SummitFrankaSceneCfg().robot.replace(prim_path=prim_path)
    visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=sample.color,
        opacity=sample.opacity,
        roughness=0.62,
    )
    if sample.urdf_path:
        robot_cfg.spawn = sim_utils.UrdfFileCfg(
            asset_path=str(Path(sample.urdf_path)),
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=False,
            visual_material=visual_material,
            visual_material_path=f"{prim_path}/GhostMaterial",
        )
    else:
        robot_cfg.spawn.activate_contact_sensors = False
        robot_cfg.spawn.visual_material = visual_material
        robot_cfg.spawn.visual_material_path = f"{prim_path}/GhostMaterial"
    robot_cfg.init_state.pos = (0.0, 0.0, 0.0)
    robot_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    return Articulation(cfg=robot_cfg)


def spawn_ghost_robots(group_path: str, samples: list[GhostSample]) -> dict[str, Any]:
    import isaacsim.core.utils.prims as prim_utils

    prim_utils.create_prim(group_path, "Xform")
    robots: dict[str, Any] = {}
    for sample in samples:
        robots[sample.name] = make_robot_articulation(sample, f"{group_path}/{sample.name}")
    return robots


def apply_ghost_samples(robots: dict[str, Any], samples: list[GhostSample], sim_dt: float) -> None:
    for sample in samples:
        robot = robots[sample.name]
        root_pose = robot.data.default_root_state[:, :7].clone()
        root_pose[:, :3] = torch.tensor((0.0, 0.0, 0.0), device=robot.device).unsqueeze(0)
        root_pose[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=robot.device).unsqueeze(0)
        root_vel = torch.zeros_like(robot.data.default_root_state[:, 7:])
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(root_vel)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        for name, value in sample.joint_pos.items():
            if name not in robot.joint_names:
                continue
            joint_pos[:, robot.joint_names.index(name)] = float(value)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
        robot.update(sim_dt)


def style_ghosts(stage: Any, group_path: str, samples: list[GhostSample]) -> None:
    import isaaclab.sim as sim_utils

    for sample in samples:
        material_path = f"{group_path}/{sample.name}/PosterGhostMaterial"
        material = make_preview_material(stage, material_path, sample.color, sample.opacity)
        try:
            sim_utils.bind_visual_material(
                f"{group_path}/{sample.name}",
                material_path,
                stage=stage,
                stronger_than_descendants=True,
            )
        except Exception as exc:
            print(f"[ghost] WARNING: material command bind failed for {sample.name}: {exc}")
        bind_material_to_meshes(stage, f"{group_path}/{sample.name}", material, sample.opacity)
    print(f"[ghost] styled {len(samples)} ghost robots under {group_path}")


def hide_fixed_base_visuals(stage: Any, group_path: str) -> int:
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(group_path)
    if not root or not root.IsValid():
        return 0
    hidden = 0
    needles = ("base_link_z", "summit_base", "summit_chassis", "franka_stand")
    for prim in Usd.PrimRange(root):
        path = prim.GetPath().pathString.lower()
        name = prim.GetName().lower()
        if any(needle in path or needle in name for needle in needles):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden += 1
    return hidden


def make_curve_material(stage: Any, prim_path: str, color: tuple[float, float, float]) -> Any:
    return make_preview_material(stage, prim_path, color, 1.0)


def add_base_path_curve(
    stage: Any,
    prim_path: str,
    xytheta: np.ndarray,
    color: tuple[float, float, float],
    width: float = 0.08,
    z: float = 0.18,
) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    points = [Gf.Vec3f(float(x), float(y), float(z)) for x, y in xytheta[:, :2]]
    curve = UsdGeom.BasisCurves.Define(stage, prim_path)
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr(points)
    curve.CreateWidthsAttr([width])
    curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    material = make_curve_material(stage, f"{prim_path}_Material", color)
    UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )


def add_marker_sphere(
    stage: Any,
    prim_path: str,
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    radius: float,
    opacity: float = 1.0,
) -> None:
    from pxr import Gf, UsdGeom, UsdShade

    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    sphere.CreateDisplayOpacityAttr([float(opacity)])
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*position))
    material = make_preview_material(stage, f"{prim_path}_Material", color, opacity)
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )


def add_mobile_base_footprint(
    stage: Any,
    prim_path: str,
    sample: GhostSample,
) -> None:
    """Add a lightweight Summit base proxy for poster-only mobile ghosts."""

    from pxr import Gf, UsdGeom, UsdShade

    x = float(sample.joint_pos.get("base_x", 0.0))
    y = float(sample.joint_pos.get("base_y", 0.0))
    yaw = float(sample.joint_pos.get("base_z", 0.0))
    opacity = 0.56 if sample.name.startswith("mobile_traj") else 0.13

    root = UsdGeom.Xform.Define(stage, prim_path)
    xform = UsdGeom.Xformable(root.GetPrim())
    xform.ClearXformOpOrder()
    # Float the proxy above the kitchen meshes for top-down poster readability:
    # the imported Summit chassis USD is unresolved in this asset set, and the
    # true floor-height proxy can be occluded by iTHOR cabinetry.
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(x, y, 0.92))
    xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(math.degrees(yaw))

    body_material = make_preview_material(stage, f"{prim_path}_BodyMaterial", sample.color, opacity)
    wheel_material = make_preview_material(stage, f"{prim_path}_WheelMaterial", (0.10, 0.10, 0.09), min(0.62, opacity + 0.18))

    def add_ellipsoid(
        name: str,
        local_pos: tuple[float, float, float],
        scale: tuple[float, float, float],
        color: tuple[float, float, float],
        material: Any,
        part_opacity: float,
    ) -> None:
        sphere = UsdGeom.Sphere.Define(stage, f"{prim_path}/{name}")
        sphere.CreateRadiusAttr(1.0)
        sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        sphere.CreateDisplayOpacityAttr([float(part_opacity)])
        sphere_xform = UsdGeom.Xformable(sphere.GetPrim())
        sphere_xform.ClearXformOpOrder()
        sphere_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*local_pos))
        sphere_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )

    # The Isaac USD currently drops the Summit chassis visual in headless
    # renders, so this proxy restores the mobile base without touching assets.
    add_ellipsoid("chassis", (0.0, 0.0, 0.0), (0.34, 0.24, 0.060), sample.color, body_material, opacity)
    add_ellipsoid("top_plate", (-0.02, 0.0, 0.12), (0.20, 0.15, 0.035), sample.color, body_material, opacity)
    for wheel_id, (wx, wy) in enumerate(((-0.31, -0.25), (-0.31, 0.25), (0.31, -0.25), (0.31, 0.25))):
        add_ellipsoid(
            f"wheel_{wheel_id:02d}",
            (wx, wy, -0.05),
            (0.065, 0.040, 0.070),
            (0.10, 0.10, 0.09),
            wheel_material,
            min(0.62, opacity + 0.18),
        )


def add_mobile_base_footprints(stage: Any, group_path: str, samples: list[GhostSample]) -> None:
    import isaacsim.core.utils.prims as prim_utils

    footprint_group = f"{group_path}/MobileBaseFootprints"
    prim_utils.create_prim(footprint_group, "Xform")
    visible_samples = [
        sample
        for sample in samples
        if not sample.hide_summit_base and sample.name.startswith("mobile_traj")
    ]
    for index, sample in enumerate(visible_samples):
        add_mobile_base_footprint(stage, f"{footprint_group}/{sample.name}_{index:02d}", sample)
    print(f"[ghost] added {len(visible_samples)} mobile base footprints under {footprint_group}")


def mobile_footprint_paths(group_path: str, samples: list[GhostSample]) -> dict[str, str]:
    visible_samples = [
        sample
        for sample in samples
        if not sample.hide_summit_base and sample.name.startswith("mobile_traj")
    ]
    return {
        sample.name: f"{group_path}/MobileBaseFootprints/{sample.name}_{index:02d}"
        for index, sample in enumerate(visible_samples)
    }


def project_world_to_pixel(
    point: tuple[float, float, float],
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    width: int,
    height: int,
) -> tuple[float, float] | None:
    """Project a world point with the poster camera convention."""

    eye_v = np.asarray(eye, dtype=float)
    target_v = np.asarray(target, dtype=float)
    point_v = np.asarray(point, dtype=float)
    forward = target_v - eye_v
    forward = forward / max(np.linalg.norm(forward), 1.0e-9)
    world_up = np.asarray((0.0, 0.0, 1.0), dtype=float)
    right = np.cross(forward, world_up)
    right = right / max(np.linalg.norm(right), 1.0e-9)
    up = np.cross(right, forward)
    delta = point_v - eye_v
    depth = float(np.dot(delta, forward))
    if depth <= 1.0e-6:
        return None
    focal_px = 28.0 / 20.955 * float(width)
    x = float(width) * 0.5 + focal_px * float(np.dot(delta, right)) / depth
    y = float(height) * 0.5 - focal_px * float(np.dot(delta, up)) / depth
    return x, y


def overlay_mobile_base_proxies(
    image_path: Path,
    samples: list[GhostSample],
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    width: int,
    height: int,
) -> None:
    """Draw mobile base overlays directly on the PNG when USD opacity is unreliable."""

    visible_samples = [
        sample
        for sample in samples
        if not sample.hide_summit_base and sample.name.startswith("mobile_traj")
    ]
    if not visible_samples:
        return
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"[ghost] WARNING: PIL unavailable; skipped mobile base PNG overlay: {exc}")
        return

    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    def as_rgba(color: tuple[float, float, float], alpha: int) -> tuple[int, int, int, int]:
        return tuple(int(np.clip(c, 0.0, 1.0) * 255) for c in color) + (alpha,)

    def project_xy(x: float, y: float, z: float = 0.95) -> tuple[float, float] | None:
        return project_world_to_pixel((x, y, z), eye, target, width, height)

    for sample in visible_samples:
        x = float(sample.joint_pos.get("base_x", 0.0))
        y = float(sample.joint_pos.get("base_y", 0.0))
        yaw = float(sample.joint_pos.get("base_z", 0.0))
        alpha = 64
        outline_alpha = 104
        color = sample.color
        c, s = math.cos(yaw), math.sin(yaw)

        def transform(local_x: float, local_y: float) -> tuple[float, float]:
            return (x + c * local_x - s * local_y, y + s * local_x + c * local_y)

        body = [transform(px, py) for px, py in ((-0.38, -0.30), (0.38, -0.30), (0.38, 0.30), (-0.38, 0.30))]
        body_px = [project_xy(px, py) for px, py in body]
        if all(point is not None for point in body_px):
            draw.polygon(body_px, fill=as_rgba(color, alpha), outline=as_rgba(color, outline_alpha))

        for wx, wy in ((-0.33, -0.31), (-0.33, 0.31), (0.33, -0.31), (0.33, 0.31)):
            wheel_center = transform(wx, wy)
            center_px = project_xy(*wheel_center)
            edge_px = project_xy(*(transform(wx + 0.08, wy + 0.04)))
            if center_px is None or edge_px is None:
                continue
            radius = max(2.0, math.dist(center_px, edge_px))
            draw.ellipse(
                (
                    center_px[0] - radius,
                    center_px[1] - radius,
                    center_px[0] + radius,
                    center_px[1] + radius,
                ),
                fill=(24, 24, 22, 92),
            )

    Image.alpha_composite(image, overlay).convert("RGB").save(image_path)


def add_fixed_base_marker(
    stage: Any,
    group_path: str,
    fixed_base: tuple[float, float, float],
) -> None:
    add_marker_sphere(
        stage,
        f"{group_path}/FixedBaseAnchor",
        (float(fixed_base[0]), float(fixed_base[1]), 0.08),
        FIXED_COLOR,
        0.12,
        0.95,
    )


def add_mobile_base_curves(
    stage: Any,
    group_path: str,
    mobile_trajs: torch.Tensor,
    selected_meta: list[dict[str, Any]],
    xy_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    import isaacsim.core.utils.prims as prim_utils

    curve_group = f"{group_path}/BaseTrajectories"
    prim_utils.create_prim(curve_group, "Xform")
    for index, meta in enumerate(selected_meta):
        trajectory = mobile_trajs[int(meta["trajectory_index"])].numpy()
        trajectory = trajectory.copy()
        trajectory[:, 0] += float(xy_offset[0])
        trajectory[:, 1] += float(xy_offset[1])
        color = tuple(meta["color"])
        add_base_path_curve(stage, f"{curve_group}/traj_{index:02d}", trajectory[:, :3], color, width=0.10, z=0.20)
        for marker_id, frame in enumerate(keyframe_indices(trajectory.shape[0], 9)):
            x, y = trajectory[frame, :2]
            add_marker_sphere(
                stage,
                f"{curve_group}/traj_{index:02d}_dot_{marker_id:02d}",
                (float(x), float(y), 0.22),
                color,
                0.085,
                0.92,
        )


def individual_frame_samples(samples: list[GhostSample], mode: str) -> list[GhostSample]:
    if mode == "all":
        return list(samples)
    return [
        sample
        for sample in samples
        if sample.name.startswith("fixed_traj") or sample.name.startswith("mobile_traj")
    ]


def render_with_current_visibility(
    sim: Any,
    camera: Any,
    robots: dict[str, Any],
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    output_path: Path,
) -> None:
    set_camera_view(camera, sim, eye, target)
    dt = sim.get_physics_dt()
    for _ in range(RENDER_SETTLE_STEPS):
        for robot in robots.values():
            robot.write_data_to_sim()
        sim.step(render=True)
        for robot in robots.values():
            robot.update(dt)
        camera.update(dt)
    save_rgb_image(camera, output_path)
    print(f"[render] wrote {output_path}")


def export_individual_frames(
    args: argparse.Namespace,
    sim: Any,
    camera: Any,
    robots: dict[str, Any],
    stage: Any,
    group_path: str,
    samples: list[GhostSample],
    panel_out_dir: Path,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> list[str]:
    selected = individual_frame_samples(samples, args.individual_frame_mode)
    if args.individual_frame_limit > 0:
        selected = selected[: args.individual_frame_limit]
    if not selected:
        return []

    frame_dir = panel_out_dir / "individual_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    footprint_paths = mobile_footprint_paths(group_path, samples)
    base_curve_group = f"{group_path}/BaseTrajectories"

    exported: list[str] = []
    for sample in samples:
        set_prim_visibility(stage, f"{group_path}/{sample.name}", False, recursive=False)
    for footprint_path in footprint_paths.values():
        set_prim_visibility(stage, footprint_path, False, recursive=False)
    # Set inherited visibility only on parent prims where no child needs to be
    # selectively shown. Recursing after physics init can invalidate PhysX.
    set_prim_visibility(stage, base_curve_group, False, recursive=False)

    for index, sample in enumerate(selected):
        robot_path = f"{group_path}/{sample.name}"
        footprint_path = footprint_paths.get(sample.name)
        set_prim_visibility(stage, robot_path, True, recursive=False)
        if footprint_path is not None:
            set_prim_visibility(stage, footprint_path, True, recursive=False)

        output_path = frame_dir / f"{index:03d}_{sample.name}.png"
        render_with_current_visibility(sim, camera, robots, eye, target, output_path)
        overlay_mobile_base_proxies(output_path, [sample], eye, target, args.image_width, args.image_height)
        exported.append(str(output_path))

        set_prim_visibility(stage, robot_path, False, recursive=False)
        if footprint_path is not None:
            set_prim_visibility(stage, footprint_path, False, recursive=False)

    for sample in samples:
        set_prim_visibility(stage, f"{group_path}/{sample.name}", True, recursive=False)
    for footprint_path in footprint_paths.values():
        set_prim_visibility(stage, footprint_path, True, recursive=False)
    set_prim_visibility(stage, base_curve_group, True, recursive=False)
    print(f"[ghost] exported {len(exported)} individual frame renders to {frame_dir}")
    return exported


def episode_name(sample_name: str) -> str:
    for prefix in ("mobile_traj", "fixed_traj"):
        if sample_name.startswith(prefix):
            suffix = sample_name[len(prefix) :].split("_t", 1)[0]
            if suffix.isdigit():
                return f"episode{int(suffix) + 1:02d}"
    if "_t" in sample_name:
        return sample_name.rsplit("_t", 1)[0]
    return sample_name


def add_camera_facing_source_background(
    stage: Any,
    prim_path: str,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    color: tuple[float, float, float],
    size: float = SOURCE_BACKGROUND_SIZE,
    distance: float = SOURCE_BACKGROUND_DISTANCE,
) -> str:
    """Add a pure-color backdrop for robot-only source layers.

    The robots are white, so the default white renderer background makes later
    mask extraction fragile.  A camera-facing green card gives every source
    render a stable chroma-key background without changing the scene layer.
    """

    from pxr import Gf, Sdf, UsdGeom, UsdShade

    eye_np = np.asarray(eye, dtype=np.float64)
    target_np = np.asarray(target, dtype=np.float64)
    forward = target_np - eye_np
    forward /= np.linalg.norm(forward)
    world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1.0e-6:
        right = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    else:
        right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    center = target_np + forward * distance
    half = size * 0.5
    points = [
        Gf.Vec3f(*(center - right * half - up * half)),
        Gf.Vec3f(*(center + right * half - up * half)),
        Gf.Vec3f(*(center + right * half + up * half)),
        Gf.Vec3f(*(center - right * half + up * half)),
    ]

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)

    material_path = f"{prim_path}_Material"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    rgb = Gf.Vec3f(*color)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(rgb)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(rgb)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh).Bind(material)
    set_prim_visibility(stage, prim_path, False, recursive=False)
    return prim_path


def normalize_green_source_background(path: Path, color: tuple[float, float, float]) -> None:
    """Snap the chroma backdrop to exact green after renderer tonemapping."""

    if tuple(round(c, 3) for c in color) != (0.0, 1.0, 0.0):
        return
    try:
        from PIL import Image
    except Exception as exc:
        print(f"[ghost] WARNING: cannot normalize source background without PIL: {exc}")
        return

    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8).copy()
    greenish = (arr[..., 1] > 180) & (arr[..., 0] < 90) & (arr[..., 2] < 90)
    arr[greenish] = np.asarray((0, 255, 0), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def zoom_robot_only_source(
    path: Path,
    color: tuple[float, float, float],
    margin_scale: float = 1.28,
) -> None:
    """Zoom standalone robot-only workspace sources into a detail-style plate."""

    if tuple(round(c, 3) for c in color) != (0.0, 1.0, 0.0):
        return
    try:
        from PIL import Image
    except Exception as exc:
        print(f"[ghost] WARNING: cannot zoom source background without PIL: {exc}")
        return

    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    robot_mask = ~((arr[..., 1] > 220) & (arr[..., 0] < 40) & (arr[..., 2] < 40))
    ys, xs = np.nonzero(robot_mask)
    if xs.size == 0 or ys.size == 0:
        return

    width, height = image.size
    xmin, xmax = int(xs.min()), int(xs.max()) + 1
    ymin, ymax = int(ys.min()), int(ys.max()) + 1
    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    crop_w = max(float(xmax - xmin) * margin_scale, 1.0)
    crop_h = max(float(ymax - ymin) * margin_scale, 1.0)
    aspect = width / height
    if crop_w / crop_h > aspect:
        crop_h = crop_w / aspect
    else:
        crop_w = crop_h * aspect

    left = max(0, int(round(cx - crop_w * 0.5)))
    right = min(width, int(round(cx + crop_w * 0.5)))
    top = max(0, int(round(cy - crop_h * 0.5)))
    bottom = min(height, int(round(cy + crop_h * 0.5)))
    if right <= left or bottom <= top:
        return

    resample = getattr(Image, "Resampling", Image).LANCZOS
    image.crop((left, top, right, bottom)).resize((width, height), resample).save(path)
    normalize_green_source_background(path, color)


def export_layer_sources(
    args: argparse.Namespace,
    sim: Any,
    camera: Any,
    robots: dict[str, Any],
    stage: Any,
    group_path: str,
    samples: list[GhostSample],
    panel_out_dir: Path,
    view_name: str,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    scene_layer_path: Path | None = None,
) -> dict[str, Any]:
    source_dir = panel_out_dir / "sources" / view_name
    source_dir.mkdir(parents=True, exist_ok=True)
    source_outputs: dict[str, Any] = {}

    scene_path = source_dir / "scene_objects.png"
    if scene_layer_path is not None and scene_layer_path.exists():
        shutil.copy2(scene_layer_path, scene_path)
    else:
        render_one(sim, camera, robots, stage, group_path, eye, target, "scene_objects_only", scene_path)
    source_outputs["scene"] = str(scene_path)

    actual_samples = [
        sample
        for sample in samples
        if sample.name.startswith("mobile_traj") or sample.name.startswith("fixed_traj")
    ]
    workspace_samples = [
        sample
        for sample in samples
        if sample.name.startswith("mobile_workspace") or sample.name.startswith("fixed_workspace")
    ]

    source_background_path = add_camera_facing_source_background(
        stage,
        f"{group_path}/SourceBackground_{view_name}",
        eye,
        target,
        tuple(args.source_background_color),
    )

    set_prim_visibility(stage, "/World/Scene", False, recursive=False)
    set_prim_visibility(stage, "/World/Objects", False, recursive=False)
    set_prim_visibility(stage, group_path, True, recursive=False)
    for sample in samples:
        set_prim_visibility(stage, f"{group_path}/{sample.name}", False, recursive=False)
    set_prim_visibility(stage, f"{group_path}/BaseTrajectories", False, recursive=False)
    set_prim_visibility(stage, f"{group_path}/MobileBaseFootprints", False, recursive=False)
    set_prim_visibility(stage, f"{group_path}/FixedBaseAnchor", False, recursive=False)

    episodes: dict[str, list[str]] = {}
    for sample in actual_samples:
        episode_dir = source_dir / episode_name(sample.name)
        episode_dir.mkdir(parents=True, exist_ok=True)
        robot_path = f"{group_path}/{sample.name}"
        set_prim_visibility(stage, robot_path, True, recursive=False)
        if sample.hide_summit_base:
            hide_fixed_base_visuals(stage, robot_path)
        set_prim_visibility(stage, source_background_path, True, recursive=False)
        output_path = episode_dir / f"{sample.name}_robot_only.png"
        render_with_current_visibility(sim, camera, robots, eye, target, output_path)
        normalize_green_source_background(output_path, tuple(args.source_background_color))
        episodes.setdefault(episode_dir.name, []).append(str(output_path))
        set_prim_visibility(stage, robot_path, False, recursive=False)
        set_prim_visibility(stage, source_background_path, False, recursive=False)

    if workspace_samples:
        for sample in workspace_samples:
            robot_path = f"{group_path}/{sample.name}"
            set_prim_visibility(stage, robot_path, True, recursive=False)
            if sample.hide_summit_base:
                hide_fixed_base_visuals(stage, robot_path)
        set_prim_visibility(stage, source_background_path, True, recursive=False)
        workspace_path = source_dir / "workspace_robots_only.png"
        render_with_current_visibility(sim, camera, robots, eye, target, workspace_path)
        normalize_green_source_background(workspace_path, tuple(args.source_background_color))
        if view_name == "close" and all(sample.name.startswith("fixed_workspace") for sample in workspace_samples):
            zoom_robot_only_source(workspace_path, tuple(args.source_background_color))
        source_outputs["workspace"] = str(workspace_path)
        set_prim_visibility(stage, source_background_path, False, recursive=False)
        for sample in workspace_samples:
            set_prim_visibility(stage, f"{group_path}/{sample.name}", False, recursive=False)

    for sample in samples:
        set_prim_visibility(stage, f"{group_path}/{sample.name}", True, recursive=False)
    set_prim_visibility(stage, "/World/Scene", True, recursive=False)
    set_prim_visibility(stage, "/World/Objects", True, recursive=False)
    set_prim_visibility(stage, f"{group_path}/BaseTrajectories", True, recursive=False)
    set_prim_visibility(stage, f"{group_path}/MobileBaseFootprints", True, recursive=False)
    set_prim_visibility(stage, f"{group_path}/FixedBaseAnchor", True, recursive=False)

    source_outputs["episodes"] = episodes
    print(f"[ghost] exported layered sources for {view_name} to {source_dir}")
    return source_outputs


def camera_for_actual_ghost(spec: Any, view: str = "overview") -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    config = SCENE_RENDER_CONFIGS.get(spec.name, {})
    target = tuple(config.get("overview_target", (0.0, 0.25, 0.7)))
    # Poster v3 uses an oblique camera so the scene reads as a 3D room rather
    # than a floor-plan crop.
    if view == "close":
        close_target = tuple(config.get("poster_close_target", (0.95, 1.95, 0.65)))
        return (close_target[0], close_target[1] - 3.15, 5.25), close_target
    return (target[0], target[1] - 5.40, 9.40), (target[0], target[1], 0.55)


def export_standalone_stage(stage: Any, output_path: Path) -> Path | None:
    """Export a flattened USD layer so the poster stage can be opened alone."""

    from pxr import Sdf, Usd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        flattened = stage.Flatten(addSourceFileComment=False)
    except TypeError:
        flattened = stage.Flatten()

    # Keep the file self-contained by removing material texture/MDL asset
    # pointers.  The flattened layer still contains the composed geometry and
    # internal prototype references, but does not need repo-local texture files.
    stripped_assets = 0
    flat_stage = Usd.Stage.Open(flattened)
    if flat_stage is not None:
        for prim in flat_stage.TraverseAll():
            for attr in prim.GetAttributes():
                value = attr.Get()
                if isinstance(value, Sdf.AssetPath) and value.path:
                    attr.Clear()
                    stripped_assets += 1
                elif isinstance(value, (list, tuple)) and any(isinstance(item, Sdf.AssetPath) for item in value):
                    attr.Clear()
                    stripped_assets += 1
        flattened = flat_stage.GetRootLayer()

    try:
        flattened.Export(str(output_path))
    except Exception as exc:
        print(f"[ghost] WARNING: failed to export standalone USD {output_path}: {exc}")
        return None
    print(f"[ghost] exported standalone USD {output_path} stripped_asset_attrs={stripped_assets}")
    return output_path


def _path_is_descendant(path: str, ancestor: str) -> bool:
    return path.startswith(f"{ancestor}/")


def _path_is_ancestor(path: str, descendant: str) -> bool:
    return descendant.startswith(f"{path}/")


def resolve_scene_foreground_paths(stage: Any) -> tuple[list[str], list[str]]:
    """Map original iTHOR USD paths to their referenced paths in the render stage."""

    resolved: list[str] = []
    missing: list[str] = []
    for raw_path in SCENE_FOREGROUND_OCCLUDER_PATHS:
        suffix = raw_path.removeprefix("/World")
        candidates = [
            f"/World/Scene/iTHOR{suffix}",
            f"/World/Scene/iTHOR{raw_path}",
            raw_path,
        ]
        match = next((path for path in candidates if stage.GetPrimAtPath(path).IsValid()), None)
        if match is None:
            missing.append(raw_path)
        else:
            resolved.append(match)
    return resolved, missing


def scene_baseline_visibility(stage: Any) -> dict[str, bool]:
    """Record which prims are actually visible in the loaded USD scene."""

    from pxr import Usd, UsdGeom

    baseline: dict[str, bool] = {}
    for root_path in ("/World/Scene", "/World/Objects"):
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim or not root_prim.IsValid():
            continue
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Imageable):
                imageable = UsdGeom.Imageable(prim)
                baseline[str(prim.GetPath())] = imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
    return baseline


def apply_scene_baseline_visibility(stage: Any, baseline_visibility: dict[str, bool]) -> None:
    from pxr import UsdGeom

    for path, visible in baseline_visibility.items():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
            continue
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible() if visible else imageable.MakeInvisible()


def set_scene_foreground_visibility(
    stage: Any,
    mode: str,
    foreground_paths: list[str],
    baseline_visibility: dict[str, bool],
) -> None:
    """Switch scene visibility between normal, no-foreground, and foreground-only."""

    from pxr import Usd, UsdGeom

    apply_scene_baseline_visibility(stage, baseline_visibility)

    if mode == "normal":
        return

    if mode == "without_foreground":
        for path in foreground_paths:
            set_prim_visibility(stage, path, False, recursive=False)
        return

    if mode != "foreground_only":
        raise ValueError(f"Unknown foreground visibility mode: {mode}")

    for root_path in ("/World/Scene", "/World/Objects"):
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim or not root_prim.IsValid():
            continue
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Imageable):
                continue
            path = str(prim.GetPath())
            keep = any(
                path == foreground
                or _path_is_descendant(path, foreground)
                or _path_is_ancestor(path, foreground)
                for foreground in foreground_paths
            )
            imageable = UsdGeom.Imageable(prim)
            imageable.MakeVisible() if keep and baseline_visibility.get(path, False) else imageable.MakeInvisible()


def visible_foreground_paths(
    stage: Any,
    foreground_paths: list[str],
    baseline_visibility: dict[str, bool],
) -> list[str]:
    from pxr import Usd, UsdGeom

    visible_paths: list[str] = []
    for foreground in foreground_paths:
        prim = stage.GetPrimAtPath(foreground)
        if not prim or not prim.IsValid():
            continue
        if not prim.IsA(UsdGeom.Imageable):
            visible_paths.append(foreground)
            continue
        if not baseline_visibility.get(foreground, False):
            continue
        has_visible_drawable_descendant = False
        for child in Usd.PrimRange(prim):
            path = str(child.GetPath())
            if child.IsA(UsdGeom.Imageable) and baseline_visibility.get(path, False):
                has_visible_drawable_descendant = True
                break
        if has_visible_drawable_descendant:
            visible_paths.append(foreground)
    return visible_paths


def render_scene_layers(args: argparse.Namespace, spec: Any, simulation_app: Any) -> dict[str, Any]:
    """Render canonical scene backgrounds once, before any robot layers."""

    import omni.usd
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext

    root = Path(args.repo_root).resolve()
    out_dir = resolve_repo_path(args.output_root, root) / "scene"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_png in out_dir.glob("*.png"):
        old_png.unlink()
    print("[ghost] rendering canonical scene layers", flush=True)

    cached_foreground_paths: list[str] | None = None
    cached_missing_foreground_paths: list[str] | None = None

    def render_one_scene_image(view_name: str, mode: str, output_path: Path) -> list[str]:
        nonlocal cached_foreground_paths, cached_missing_foreground_paths

        # Scene-only layers must represent the same simulation timestamp across
        # camera views.  Build a fresh stage per image so close/overview and
        # foreground/no-foreground renders all settle for exactly the same
        # number of physics frames from the identical initial USD state.
        SimulationContext.clear_instance()
        omni.usd.get_context().new_stage()
        for _ in range(2):
            simulation_app.update()

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device))
        spawn_static_scene(spec)
        stage = omni.usd.get_context().get_stage()
        camera = create_camera(args.image_width, args.image_height)
        sim.reset()
        baseline_visibility = scene_baseline_visibility(stage)
        foreground_paths, missing_foreground_paths = resolve_scene_foreground_paths(stage)
        foreground_paths = visible_foreground_paths(stage, foreground_paths, baseline_visibility)
        if cached_foreground_paths is None:
            cached_foreground_paths = list(foreground_paths)
            cached_missing_foreground_paths = list(missing_foreground_paths)
            if missing_foreground_paths:
                print(
                    "[ghost] WARNING: missing scene foreground occluders: "
                    + ", ".join(missing_foreground_paths)
                )
            print(f"[ghost] visible scene foreground occluders: {len(foreground_paths)}")

        eye, target = camera_for_actual_ghost(spec, view_name)
        set_scene_foreground_visibility(stage, mode, foreground_paths, baseline_visibility)
        render_with_current_visibility(sim, camera, {}, eye, target, output_path)
        SimulationContext.clear_instance()
        return foreground_paths

    views: dict[str, str] = {}
    foreground_views: dict[str, str] = {}
    for view_name in args.render_views:
        no_foreground_path = out_dir / f"{view_name}_scene_no_foreground.png"
        foreground_path = out_dir / f"{view_name}_scene_foreground_only.png"

        render_one_scene_image(view_name, "without_foreground", no_foreground_path)
        views[view_name] = str(no_foreground_path)

        render_one_scene_image(view_name, "foreground_only", foreground_path)
        foreground_views[view_name] = str(foreground_path)

    result: dict[str, Any] = {
        "views": views,
        "foreground_views": foreground_views,
        "foreground_occluders": cached_foreground_paths or [],
    }
    if cached_missing_foreground_paths:
        result["missing_foreground_occluders"] = cached_missing_foreground_paths
    if args.export_usd:
        print("[ghost] WARNING: --export_usd is ignored for scene-only fresh-stage rendering")
    SimulationContext.clear_instance()
    return result


def render_panel(
    args: argparse.Namespace,
    spec: Any,
    panel_name: str,
    samples: list[GhostSample],
    simulation_app: Any,
    mobile_trajs: torch.Tensor | None = None,
    mobile_meta: list[dict[str, Any]] | None = None,
    fixed_base: tuple[float, float, float] | None = None,
    trajectory_xy_offset: tuple[float, float] = (0.0, 0.0),
    scene_layers: dict[str, str] | None = None,
) -> dict[str, Any]:
    import omni.usd
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext

    root = Path(args.repo_root).resolve()
    out_dir = resolve_repo_path(args.output_root, root) / panel_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ghost] render panel={panel_name} samples={len(samples)}", flush=True)

    SimulationContext.clear_instance()
    omni.usd.get_context().new_stage()
    for _ in range(2):
        simulation_app.update()

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device))
    spawn_static_scene(spec)
    group_path = f"/World/{panel_name.title().replace('_', '')}Ghosts"
    print(f"[ghost] spawning robots for {panel_name}", flush=True)
    robots = spawn_ghost_robots(group_path, samples)
    stage = omni.usd.get_context().get_stage()

    if mobile_trajs is not None and mobile_meta is not None:
        add_mobile_base_curves(stage, group_path, mobile_trajs, mobile_meta, xy_offset=trajectory_xy_offset)
        if args.use_mobile_base_proxy:
            add_mobile_base_footprints(stage, group_path, samples)
    if fixed_base is not None:
        add_fixed_base_marker(stage, group_path, fixed_base)

    print(f"[ghost] styling ghosts for {panel_name}", flush=True)
    style_ghosts(stage, group_path, samples)
    hidden = hide_fixed_base_visuals(stage, group_path) if panel_name.startswith("fixed") else 0
    if hidden:
        print(f"[ghost] hid {hidden} fixed-base Summit visual prims in {panel_name}")

    camera = create_camera(args.image_width, args.image_height)
    print(f"[ghost] resetting sim for {panel_name}", flush=True)
    sim.reset()
    print(f"[ghost] applying samples for {panel_name}", flush=True)
    apply_ghost_samples(robots, samples, sim.get_physics_dt())

    primary_output_path: Path | None = None
    view_outputs: dict[str, Any] = {}
    individual_frames: list[str] = []
    for view_name in args.render_views:
        eye, target = camera_for_actual_ghost(spec, view_name)
        output_path = out_dir / f"{panel_name}_{view_name}_all.png"
        print(f"[ghost] rendering {output_path}", flush=True)
        if panel_name.startswith("fixed"):
            set_prim_visibility(stage, "/World/Scene", True, recursive=False)
            set_prim_visibility(stage, "/World/Objects", True, recursive=False)
            set_prim_visibility(stage, group_path, True, recursive=True)
            hide_fixed_base_visuals(stage, group_path)
            render_with_current_visibility(sim, camera, robots, eye, target, output_path)
        else:
            render_one(sim, camera, robots, stage, group_path, eye, target, "all", output_path)
        if args.use_mobile_base_proxy and panel_name.startswith("mobile"):
            overlay_mobile_base_proxies(output_path, samples, eye, target, args.image_width, args.image_height)
        if primary_output_path is None:
            primary_output_path = output_path
            legacy_path = out_dir / f"{panel_name}_topdown_ghost.png"
            if legacy_path != output_path:
                try:
                    import shutil

                    shutil.copy2(output_path, legacy_path)
                except Exception as exc:
                    print(f"[ghost] WARNING: failed to write legacy topdown image: {exc}")
        view_result: dict[str, Any] = {"image": str(output_path)}
        if args.export_layer_sources:
            view_result["sources"] = export_layer_sources(
                args,
                sim,
                camera,
                robots,
                stage,
                group_path,
                samples,
                out_dir,
                view_name,
                eye,
                target,
                Path(scene_layers[view_name]) if scene_layers and view_name in scene_layers else None,
            )
        if args.export_individual_frames and view_name == args.render_views[0]:
            individual_frames = export_individual_frames(
                args,
                sim,
                camera,
                robots,
                stage,
                group_path,
                samples,
                out_dir,
                eye,
                target,
            )
        view_outputs[view_name] = view_result
    result = {
        "image": str(primary_output_path),
        "views": view_outputs,
        "num_samples": len(samples),
    }
    if args.export_usd:
        stage_path = out_dir / "stage.usd"
        stage.GetRootLayer().Export(str(stage_path))
        standalone_path = export_standalone_stage(stage, out_dir / "stage_standalone.usd")
        result["stage"] = str(stage_path)
        if standalone_path is not None:
            result["standalone_stage"] = str(standalone_path)
    SimulationContext.clear_instance()
    if individual_frames:
        result["individual_frames"] = individual_frames
    return result


def make_side_by_side(output_root: Path, fixed_image: Path, mobile_image: Path) -> Path | None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"[ghost] side-by-side skipped: PIL unavailable ({exc})")
        return None
    fixed = Image.open(fixed_image).convert("RGB")
    mobile = Image.open(mobile_image).convert("RGB")
    label_h = 72
    width = fixed.width + mobile.width
    height = max(fixed.height, mobile.height) + label_h
    sheet = Image.new("RGB", (width, height), (244, 241, 235))
    draw = ImageDraw.Draw(sheet)
    draw.text((28, 22), "Fixed base: arm-only ghost trajectory", fill=(24, 28, 25))
    draw.text((fixed.width + 28, 22), "Mobile base: workspace ghosts + solved trajectories", fill=(24, 28, 25))
    sheet.paste(fixed, (0, label_h))
    sheet.paste(mobile, (fixed.width, label_h))
    path = output_root / "fixed_vs_mobile_actual_ghost.png"
    sheet.save(path)
    return path


def add_common_args(parser: argparse.ArgumentParser) -> None:
    root = repo_root()
    parser.add_argument("--repo_root", default=str(root))
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--object_id", default=DEFAULT_OBJECT_ID)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", default=str(root / DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--summit_robot_cfg", default=str(local_robot_cfg_path(root, "summit_franka.yml")))
    parser.add_argument(
        "--fixed_robot_cfg",
        default=str(local_robot_cfg_path(root, "summit_franka_fixed_base.yml")),
    )
    parser.add_argument(
        "--dataset_config",
        default=str(root / ".idea/cuakr_ithor/configs/dataset_config.yml"),
    )
    parser.add_argument(
        "--object_data",
        default=str(root / ".idea/cuakr_ithor/configs/object_data.json"),
    )
    parser.add_argument("--object_root", default=str(root / "assets/object"))
    parser.add_argument("--image_width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--image_height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--mobile_traj_count", type=int, default=MOBILE_TRAJ_COUNT)
    parser.add_argument("--mobile_keyframes", type=int, default=MOBILE_KEYFRAMES)
    parser.add_argument("--mobile_workspace_ghosts", type=int, default=MOBILE_WORKSPACE_GHOSTS)
    parser.add_argument("--fixed_workspace_ghosts", type=int, default=FIXED_WORKSPACE_GHOSTS)
    parser.add_argument("--fixed_keyframes", type=int, default=FIXED_KEYFRAMES)
    parser.add_argument("--fixed_arm_ghosts", type=int, default=FIXED_ARM_GHOSTS)
    parser.add_argument("--fixed_episode_count", type=int, default=3)
    parser.add_argument("--panels", nargs="+", choices=["fixed", "mobile"], default=["fixed", "mobile"])
    parser.add_argument("--render_views", nargs="+", choices=["overview", "close"], default=["overview"])
    parser.add_argument(
        "--source_background_color",
        nargs=3,
        type=float,
        default=list(SOURCE_BACKGROUND_COLOR),
        metavar=("R", "G", "B"),
        help="RGB color in [0, 1] for robot-only source backdrops; default is pure green.",
    )
    parser.add_argument(
        "--scene_only",
        action="store_true",
        help="Render only canonical scene/object backgrounds and skip robot trajectory panels.",
    )
    parser.add_argument(
        "--export_layer_sources",
        action="store_true",
        help="Export scene-only and robot-only source layers for scriptable poster compositing.",
    )
    parser.add_argument(
        "--export_usd",
        action="store_true",
        help="Export composed USD stages. Off by default for raster-only poster outputs.",
    )
    parser.add_argument(
        "--use_usd_robot",
        action="store_true",
        help="Deprecated compatibility flag; USD is now the default for reliable headless rendering.",
    )
    parser.add_argument(
        "--use_urdf_robot",
        action="store_true",
        help="Experimental: spawn poster-local URDF variants instead of the default IsaacLab-Arena USD.",
    )
    parser.add_argument(
        "--use_mobile_base_proxy",
        action="store_true",
        help="Draw the old simplified mobile-base proxy/PNG overlay. Off by default when URDF base is used.",
    )
    parser.add_argument(
        "--export_individual_frames",
        action="store_true",
        help="Also render one PNG per selected keyframe robot so the ghosts can be composited manually.",
    )
    parser.add_argument(
        "--individual_frame_mode",
        choices=["actual", "all"],
        default="actual",
        help="Which samples to export when --export_individual_frames is set.",
    )
    parser.add_argument(
        "--individual_frame_limit",
        type=int,
        default=0,
        help="Optional cap on per-frame exports per panel; 0 means no cap.",
    )
    parser.add_argument(
        "--trajectory_xy_offset",
        nargs=2,
        type=float,
        default=None,
        metavar=("DX", "DY"),
        help="Poster-only x/y offset applied to actual planned base coordinates before rendering.",
    )
    parser.add_argument("--check_inputs", action="store_true")
    # Keep compatibility with render_reach_comparison.check_inputs().
    parser.add_argument("--scenes", nargs="+", default=None)


def parse_bootstrap_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    add_common_args(parser)
    args, _ = parser.parse_known_args()
    args.scenes = args.scenes or [args.scene]
    return args


def main() -> int:
    bootstrap_args = parse_bootstrap_args()
    bootstrap_args.scenes = [bootstrap_args.scene]
    if bootstrap_args.check_inputs:
        return 0 if check_inputs(bootstrap_args) else 1

    root = Path(bootstrap_args.repo_root).resolve()
    add_repo_import_paths(root)

    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:
        print(
            "[ghost] ERROR: IsaacLab is not importable. Run through "
            "tools/paper/poster/run_actual_ghost_render.sh.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.scenes = [args.scene]

    if not check_inputs(args):
        return 1

    root = Path(args.repo_root).resolve()
    os.environ.setdefault("AUTOMOMA_OBJECT_ROOT", str(resolve_repo_path(args.object_root, root)))
    os.environ.setdefault("AUTOMOMA_ROBOT_ROOT", str(root / "assets" / "robot"))

    if args.scene_only:
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
        try:
            spec = load_scene_specs(args)[0]
            outputs: dict[str, Any] = {
                "scene_name": args.scene,
                "object_id": args.object_id,
                "scene": render_scene_layers(args, spec, simulation_app),
            }
            out_root = resolve_repo_path(args.output_root, root)
            summary_path = out_root / "manifest.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(outputs, f, indent=2)
            print(f"[ghost] summary: {summary_path}")
        finally:
            simulation_app.close()
        return 0

    data_root = Path(args.data_root).expanduser().resolve()
    summit_cfg = Path(args.summit_robot_cfg).expanduser().resolve()
    fixed_cfg = Path(args.fixed_robot_cfg).expanduser().resolve()
    mobile_trajs = load_filtered_trajs(data_root, "summit_franka", args.object_id)
    fixed_planned_trajs = load_filtered_trajs(data_root, "summit_franka_fixed_base", args.object_id)
    fixed_recorded_trajs = load_recorded_fixed_arm_trajs(data_root, "summit_franka_fixed_base", args.object_id)
    fixed_trajs = fixed_recorded_trajs if fixed_recorded_trajs is not None else fixed_planned_trajs
    fixed_traj_source = "recorded_hdf5_obs_joint_arm" if fixed_recorded_trajs is not None else "planned_pt_traj_result"
    fixed_base, fixed_base_source = fixed_base_values(
        fixed_cfg,
        data_root,
        "summit_franka_fixed_base",
        args.object_id,
    )
    trajectory_xy_offset = tuple(
        float(v)
        for v in (
            args.trajectory_xy_offset
            if args.trajectory_xy_offset is not None
            else DEFAULT_TRAJECTORY_DISPLAY_OFFSETS.get((args.scene, args.object_id), (0.0, 0.0))
        )
    )
    fixed_base_display = (
        float(fixed_base[0] + trajectory_xy_offset[0]),
        float(fixed_base[1] + trajectory_xy_offset[1]),
        float(fixed_base[2]),
    )
    print(
        f"[ghost] loaded mobile={tuple(mobile_trajs.shape)} "
        f"fixed_planned={tuple(fixed_planned_trajs.shape)} fixed_render={tuple(fixed_trajs.shape)} "
        f"fixed_traj_source={fixed_traj_source} fixed_base_raw={fixed_base} "
        f"fixed_base_display={fixed_base_display} source={fixed_base_source} "
        f"trajectory_xy_offset={trajectory_xy_offset}"
    )
    print(
        "[ghost] mobile base xy range="
        f"({mobile_trajs[:, :, 0].min().item():.3f}, {mobile_trajs[:, :, 0].max().item():.3f}) x "
        f"({mobile_trajs[:, :, 1].min().item():.3f}, {mobile_trajs[:, :, 1].max().item():.3f})"
    )

    mobile_urdf, fixed_empty_urdf = ensure_local_urdfs(root)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        spec = load_scene_specs(args)[0]
        mobile_samples, mobile_meta = build_mobile_trajectory_samples(
            mobile_trajs,
            args.mobile_traj_count,
            args.mobile_keyframes,
            xy_offset=trajectory_xy_offset,
        )
        mobile_samples = build_mobile_workspace_ghosts(spec, args.mobile_workspace_ghosts, summit_cfg) + mobile_samples
        if args.use_urdf_robot:
            for sample in mobile_samples:
                sample.urdf_path = str(mobile_urdf)

        fixed_samples, fixed_meta = build_fixed_episode_trajectory_samples(
            fixed_trajs,
            fixed_base_display,
            args.fixed_episode_count,
            args.fixed_keyframes,
        )
        fixed_workspace_samples = build_fixed_workspace_ghosts(
            fixed_base_display,
            args.fixed_workspace_ghosts,
            fixed_cfg,
            args.scene,
        )
        fixed_random_source = fixed_trajs[: min(fixed_trajs.shape[0], 128)] if fixed_recorded_trajs is not None else fixed_trajs
        fixed_samples = (
            fixed_workspace_samples
            + build_fixed_random_ghosts(fixed_random_source, fixed_base_display, args.fixed_arm_ghosts)
            + fixed_samples
        )
        if args.use_urdf_robot:
            for sample in fixed_samples:
                sample.urdf_path = str(fixed_empty_urdf)

        outputs: dict[str, Any] = {
            "data_root": str(data_root),
            "scene": args.scene,
            "object_id": args.object_id,
            "summit_robot_cfg": str(summit_cfg),
            "fixed_robot_cfg": str(fixed_cfg),
            "fixed_base": fixed_base,
            "fixed_base_display": fixed_base_display,
            "fixed_base_source": fixed_base_source,
            "trajectory_xy_offset": trajectory_xy_offset,
            "fixed_traj_source": fixed_traj_source,
            "fixed_workspace_ghosts": args.fixed_workspace_ghosts,
            "mobile_trajs_shape": list(mobile_trajs.shape),
            "fixed_planned_trajs_shape": list(fixed_planned_trajs.shape),
            "fixed_render_trajs_shape": list(fixed_trajs.shape),
            "mobile_selected": mobile_meta,
            "fixed_selected": fixed_meta,
            "mobile_urdf": str(mobile_urdf),
            "fixed_empty_base_urdf": str(fixed_empty_urdf),
        }
        if args.export_layer_sources:
            outputs["scene"] = render_scene_layers(args, spec, simulation_app)
            scene_layers = outputs["scene"]["views"]
        else:
            scene_layers = None
        if "fixed" in args.panels:
            outputs["fixed"] = render_panel(
                args,
                spec,
                "fixed_base",
                fixed_samples,
                simulation_app,
                fixed_base=fixed_base_display,
                trajectory_xy_offset=trajectory_xy_offset,
                scene_layers=scene_layers,
            )
        if "mobile" in args.panels:
            outputs["mobile"] = render_panel(
                args,
                spec,
                "mobile_base",
                mobile_samples,
                simulation_app,
                mobile_trajs=mobile_trajs,
                mobile_meta=mobile_meta,
                trajectory_xy_offset=trajectory_xy_offset,
                scene_layers=scene_layers,
            )

        out_root = resolve_repo_path(args.output_root, root)
        if "fixed" in outputs and "mobile" in outputs:
            side_by_side = make_side_by_side(
                out_root,
                Path(outputs["fixed"]["image"]),
                Path(outputs["mobile"]["image"]),
            )
            if side_by_side is not None:
                outputs["side_by_side"] = str(side_by_side)

        summary_path = out_root / "manifest.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2)
        print(f"[ghost] summary: {summary_path}")
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
