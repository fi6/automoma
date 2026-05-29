#!/usr/bin/env python3
"""Render poster figures comparing Summit-Franka and fixed Franka reach.

This is intentionally a hackable standalone script for paper/poster figures.
All tuning knobs live near the top of the file.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml


# =============================================================================
# Hackable poster parameters
# =============================================================================

DEFAULT_SCENES = (
    "ithor_floorplan1_1",
    "ithor_floorplan5_1",
    "ithor_floorplan16_1",
)
DEFAULT_ROBOTS = ("summit_franka", "franka")
DEFAULT_VIEWS = ("overview", "detail")
DEFAULT_MODES = ("all", "robots_only", "scene_objects_only")

SUMMIT_SAMPLE_COUNT = 12
FRANKA_SAMPLE_COUNT = 2
RANDOM_SEED = 20260529

IMAGE_WIDTH = 1800
IMAGE_HEIGHT = 1200
CAMERA_FOCAL_LENGTH = 28.0
RENDER_SETTLE_STEPS = 10
DOME_LIGHT_INTENSITY = 2800.0
DISTANT_LIGHT_INTENSITY = 800.0
SAVE_STAGE_USD = True
MAKE_CONTACT_SHEETS = True

# Keep samples readable in the high-overview view.
SUMMIT_MIN_SEPARATION = 0.85
FRANKA_MIN_SEPARATION = 0.55
SUMMIT_COLLISION_RADIUS = 0.46
FRANKA_BASE_COLLISION_RADIUS = 0.18
FRANKA_OBJECT_CLEARANCE = 0.16
SUMMIT_DETAIL_ANCHOR_RADIUS = 1.15
FRANKA_SURFACE_JITTER_RADIUS = 0.0
FRANKA_SURFACE_Z_OFFSET = 0.005
ELEVATED_SURFACE_MIN_Z = 0.30

# Random arm poses are sampled around natural defaults, then clamped to limits.
SUMMIT_ARM_NOISE_FRACTION = 0.14
FRANKA_ARM_NOISE_FRACTION = 0.10
FINGER_OPEN_RANGE = (0.025, 0.04)

# Approximate scene bounds for floor sampling and overview cameras.
# These are deliberately easy to edit after a first preview render.
SCENE_RENDER_CONFIGS: dict[str, dict[str, Any]] = {
    "ithor_floorplan1_1": {
        "floor_bounds": (-2.7, 2.4, -2.3, 3.0),
        # Reference view verified by the user for correct-looking object scale.
        "overview_eye": (0.0, -6.4, 9.0),
        "overview_target": (0.0, 0.25, 0.7),
        "floor_regions": [
            (-2.12, -1.16, -1.28, 1.35),
            (-1.06, 0.62, -1.65, -1.05),
            (0.62, 1.30, -1.32, 1.30),
            (-1.05, 1.28, 1.12, 1.62),
        ],
        "floor_exclusion_rects": [
            (-0.95, 0.45, -1.05, 0.90),  # kitchen island
            (1.43, 2.22, -0.48, 2.84),  # right counter/sink wall
            (-1.25, 1.45, 1.92, 2.84),  # back counter wall
            (-2.30, -1.24, 1.60, 2.85),  # shelf/corner storage
        ],
        "summit_count": 16,
        "franka_count": 3,
        "summit_fixed_points": [
            (-1.85, -1.05),
            (-1.85, -0.15),
            (-1.85, 0.75),
            (-1.45, -0.75),
            (-1.45, 0.15),
            (-1.45, 1.05),
            (-0.85, -1.53),
            (-0.25, -1.53),
            (0.35, -1.53),
            (0.94, -0.95),
            (0.94, -0.35),
            (0.94, 0.25),
            (0.94, 0.85),
            (-0.35, 1.45),
            (0.25, 1.45),
            (0.92, 1.45),
        ],
        "summit_focus_points": [
            (1.08, 1.20),
            (0.52, 1.28),
            (-1.56, 0.62),
        ],
        "franka_support_points": [
            (0.22, -0.80, 0.93),
            (-0.60, 0.45, 0.93),
            (-0.68, 2.27, 0.93),
        ],
        "franka_avoid_points": [
            (-0.48, -0.62, 0.25),  # book/magazine on the island
            (0.42, -0.12, 0.24),  # apple/food clutter
            (-0.02, 0.48, 0.22),  # bowl/plate clutter
        ],
    },
    "ithor_floorplan5_1": {
        "floor_bounds": (-2.8, 2.1, -2.35, 2.55),
        "overview_eye": (-0.35, -6.0, 8.8),
        "overview_target": (-0.35, 0.05, 0.75),
        "floor_regions": [
            (-1.58, 0.48, -1.70, 0.84),
            (0.50, 1.48, -1.15, 0.28),
            (-2.22, -1.60, -1.95, 0.28),
            (-1.42, 0.10, -2.08, -1.58),
        ],
        "floor_exclusion_rects": [
            (-0.80, 0.85, -0.38, 0.78),  # peninsula/cabinet front
        ],
        "summit_focus_points": [
            (-1.38, 0.48),
            (-0.82, 0.46),
            (-0.38, 0.30),
        ],
        "franka_support_regions": [
            {
                "name": "front_peninsula",
                "rect": (-0.08, 0.70, -1.35, -0.88),
                "z": 0.90,
                "target": (-1.60, 1.30),
            },
            {
                "name": "back_counter",
                "rect": (0.84, 1.28, 1.48, 1.92),
                "z": 0.94,
                "target": (-1.60, 1.30),
            },
        ],
        "franka_avoid_points": [
            (0.20, 1.92, 0.25),  # sink/lettuce area
            (0.95, -0.86, 0.28),  # plant/counter clutter
            (1.25, 2.02, 0.18),  # small tomato/plate clutter
        ],
    },
    "ithor_floorplan16_1": {
        "floor_bounds": (-3.2, 1.6, -4.6, 1.5),
        "overview_eye": (-0.25, -6.9, 9.0),
        "overview_target": (-0.25, -1.0, 0.75),
        "floor_regions": [
            (-1.45, 0.35, -2.18, 0.72),
            (0.35, 0.95, -2.40, -0.25),
            (-1.10, 0.82, -3.10, -2.45),
            (-0.75, 0.35, 0.75, 1.22),
        ],
        "floor_exclusion_rects": [
            (-3.10, -1.58, -2.58, 2.20),  # left counter/cabinet wall
            (0.34, 1.32, -2.72, -0.20),  # right counter/sink/fridge wall
            (-3.20, -1.62, -4.70, -3.15),  # dining table area
            (-0.20, 0.72, -0.72, 0.52),  # center cabinet/fridge block
            (-0.92, 0.72, -3.95, -3.20),  # trash/counter front
        ],
        "summit_focus_points": [
            (-1.18, -0.92),
            (-1.05, -1.75),
            (-0.62, 0.34),
        ],
        "franka_support_regions": [
            {
                "name": "left_counter",
                "rect": (-2.86, -2.45, 0.08, 0.72),
                "z": 0.94,
                "target": (-2.23, -1.00),
            },
            {
                "name": "dining_table",
                "rect": (-2.78, -2.08, -4.30, -3.58),
                "z": 0.78,
                "target": (-2.23, -1.00),
            },
        ],
        "franka_avoid_points": [
            (-2.34, -1.86, 0.24),  # plants on the left counter
            (0.78, -2.32, 0.26),  # toaster/right-counter clutter
        ],
    },
}

# Detail camera is centered on the microwave if one exists in the scene.
DETAIL_CAMERA_EYE_OFFSET = (0.0, -2.55, 2.25)
DETAIL_CAMERA_TARGET_Z_OFFSET = 0.35

ROBOT_OUTPUT_NAMES = {
    "summit_franka": "summit_franka",
    "franka": "fixed_franka",
}

# Object categories in the old cuAKR metadata do not always match the current
# assets/object folder names.
OBJECT_CATEGORY_ALIASES = {
    "StorageFurniture": "Cabinet",
    "Cabinet": "Cabinet",
    "Microwave": "Microwave",
    "Dishwasher": "Dishwasher",
    "Oven": "Oven",
    "Refrigerator": "Refrigerator",
    "TrashCan": "TrashCan",
    "Mug": "Mug",
}
MICROWAVE_CATEGORIES = {"Microwave"}

# Natural defaults and approximate limits for the arm-only part.
PANDA_JOINT_LIMITS = {
    "panda_joint1": (-2.8973, 2.8973),
    "panda_joint2": (-1.7628, 1.7628),
    "panda_joint3": (-2.8973, 2.8973),
    "panda_joint4": (-3.0718, -0.0698),
    "panda_joint5": (-2.8973, 2.8973),
    "panda_joint6": (0.5, 3.7525),
    "panda_joint7": (-2.8973, 2.8973),
    "panda_finger_joint1": (0.0, 0.04),
    "panda_finger_joint2": (0.0, 0.04),
}
SUMMIT_DEFAULT_JOINTS = {
    "base_x": 0.0,
    "base_y": 0.0,
    "base_z": 0.0,
    "panda_joint1": 0.0,
    "panda_joint2": -1.3,
    "panda_joint3": 0.0,
    "panda_joint4": -2.5,
    "panda_joint5": 0.0,
    "panda_joint6": 1.0,
    "panda_joint7": 0.0,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}
FRANKA_DEFAULT_JOINTS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.81,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}


# =============================================================================
# Data model
# =============================================================================


@dataclasses.dataclass
class ObjectSpec:
    asset_id: str
    category: str
    usd_path: Path
    translate: tuple[float, float, float]
    orient_deg: tuple[float, float, float]
    scene_scale: tuple[float, float, float]
    default_scale: float
    scale: tuple[float, float, float]


@dataclasses.dataclass
class SceneSpec:
    name: str
    usd_path: Path
    objects: list[ObjectSpec]


@dataclasses.dataclass
class RobotSample:
    name: str
    position: tuple[float, float, float]
    yaw: float
    joint_pos: dict[str, float]


# =============================================================================
# Path/config helpers that do not require Isaac Sim
# =============================================================================


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path: str | Path, root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return (int(digest[:12], 16) + RANDOM_SEED) % (2**32)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scene_usd_fallbacks(scene_name: str, root: Path) -> list[Path]:
    match = re.fullmatch(r"ithor_floorplan(\d+)_(\d+)", scene_name)
    if match is None:
        return []
    floorplan_id, setup_id = match.groups()
    ithor_root = root / "assets" / "scene" / "ithor"
    return [
        ithor_root
        / f"Collected_FloorPlan{floorplan_id}_physics"
        / f"FloorPlan{floorplan_id}_physics.usd",
        ithor_root / f"FloorPlan{floorplan_id}_physics.usd",
        root
        / "assets"
        / "scene"
        / "ithor"
        / f"setup_{setup_id}"
        / f"Collected_FloorPlan{floorplan_id}_physics"
        / f"FloorPlan{floorplan_id}_physics.usd",
        root
        / "assets"
        / "scene"
        / "ithor"
        / scene_name
        / f"FloorPlan{floorplan_id}_physics.usd",
    ]


def resolve_scene_usd(scene_name: str, scene_cfg: dict[str, Any], root: Path) -> Path:
    configured = resolve_repo_path(scene_cfg["usd_path"], root)
    if configured.exists():
        return configured
    for candidate in scene_usd_fallbacks(scene_name, root):
        if candidate.exists():
            return candidate
    return configured


def normalize_category(category: str) -> str:
    return OBJECT_CATEGORY_ALIASES.get(category, category)


def category_for_object(asset_id: str, object_data: dict[str, Any], object_root: Path) -> str:
    category = object_data.get(str(asset_id), {}).get("category")
    if category is not None:
        normalized = normalize_category(category)
        if (object_root / normalized / str(asset_id)).exists():
            return normalized

    matches = sorted(path.parent.name for path in object_root.glob(f"*/{asset_id}") if path.is_dir())
    if len(matches) == 1:
        return matches[0]
    if matches:
        preferred = ["Microwave", "Dishwasher", "Oven", "Cabinet", "TrashCan", "Refrigerator", "Mug"]
        for name in preferred:
            if name in matches:
                return name
        return matches[0]
    raise FileNotFoundError(f"Could not find object asset {asset_id} under {object_root}")


def object_usd_path(object_root: Path, category: str, asset_id: str) -> Path:
    path = object_root / category / str(asset_id) / "mobility" / "mobility.usd"
    if not path.exists():
        raise FileNotFoundError(f"Object USD not found: {path}")
    return path


def default_object_scale(asset_id: str, object_data: dict[str, Any]) -> float:
    """Return the cuAKR per-asset mesh scale, falling back to identity."""

    instance = object_data.get(str(asset_id), {}).get("instances", {}).get("0", {})
    scaling = instance.get("scaling", 1.0)
    if isinstance(scaling, (list, tuple)):
        if not scaling:
            return 1.0
        return float(scaling[0])
    return float(scaling)


def tuple3(values: Any, *, default: tuple[float, float, float] | None = None) -> tuple[float, float, float]:
    if values is None:
        if default is None:
            raise ValueError("Expected a 3-vector, got None")
        return default
    if len(values) != 3:
        raise ValueError(f"Expected a 3-vector, got {values}")
    return (float(values[0]), float(values[1]), float(values[2]))


def load_scene_specs(args: argparse.Namespace) -> list[SceneSpec]:
    root = Path(args.repo_root).resolve()
    dataset_config = resolve_repo_path(args.dataset_config, root)
    object_data_path = resolve_repo_path(args.object_data, root)
    object_root = resolve_repo_path(args.object_root, root)
    dataset = load_yaml(dataset_config)
    object_data = load_json(object_data_path)
    specs: list[SceneSpec] = []

    for scene_name in args.scenes:
        if scene_name not in dataset["scenes"]:
            raise KeyError(f"Scene {scene_name!r} not found in {dataset_config}")
        scene_cfg = dataset["scenes"][scene_name]
        scene_usd = resolve_scene_usd(scene_name, scene_cfg, root)
        objects: list[ObjectSpec] = []
        for asset_id, object_cfg in scene_cfg.get("objects", {}).items():
            transform = object_cfg.get("transform", {})
            category = category_for_object(str(asset_id), object_data, object_root)
            scene_scale = tuple3(transform.get("scale"), default=(1.0, 1.0, 1.0))
            default_scale = default_object_scale(str(asset_id), object_data)
            objects.append(
                ObjectSpec(
                    asset_id=str(asset_id),
                    category=category,
                    usd_path=object_usd_path(object_root, category, str(asset_id)),
                    translate=tuple3(transform.get("translate"), default=(0.0, 0.0, 0.0)),
                    orient_deg=tuple3(transform.get("orient"), default=(0.0, 0.0, 0.0)),
                    scene_scale=scene_scale,
                    default_scale=default_scale,
                    scale=tuple(default_scale * value for value in scene_scale),
                )
            )
        specs.append(SceneSpec(scene_name, scene_usd, objects))
    return specs


def check_inputs(args: argparse.Namespace) -> bool:
    print("[check] Resolving poster render inputs")
    ok = True
    root = Path(args.repo_root).resolve()
    summit_cfg = resolve_repo_path(args.summit_robot_cfg, root)
    summit_ok = summit_cfg.exists()
    print(f"[check] summit robot cfg: {summit_cfg} {'OK' if summit_ok else 'MISSING'}")
    ok = ok and summit_ok
    try:
        specs = load_scene_specs(args)
    except Exception as exc:
        print(f"[check] ERROR: {exc}", file=sys.stderr)
        return False

    for spec in specs:
        scene_ok = spec.usd_path.exists()
        print(f"[check] scene {spec.name}: {spec.usd_path} {'OK' if scene_ok else 'MISSING'}")
        ok = ok and scene_ok
        for obj in spec.objects:
            obj_ok = obj.usd_path.exists()
            print(
                f"[check]   object {obj.category}/{obj.asset_id}: "
                f"{obj.usd_path} scale={tuple(round(v, 4) for v in obj.scale)} "
                f"{'OK' if obj_ok else 'MISSING'}"
            )
            ok = ok and obj_ok
        if not spec.objects:
            print(f"[check]   WARNING: no objects configured for {spec.name}", file=sys.stderr)
    return ok


# =============================================================================
# Math/sampling helpers that do not require Isaac Sim
# =============================================================================


def euler_deg_to_quat_wxyz(euler_deg: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (math.radians(v) for v in euler_deg)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    half = yaw / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def clamp(value: float, limits: tuple[float, float]) -> float:
    return max(limits[0], min(limits[1], value))


def distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def scene_bounds(scene_name: str) -> tuple[float, float, float, float]:
    return tuple(SCENE_RENDER_CONFIGS.get(scene_name, {}).get("floor_bounds", (-3.0, 3.0, -3.0, 3.0)))


def scene_rects(scene_name: str, key: str) -> list[tuple[float, float, float, float]]:
    rects = SCENE_RENDER_CONFIGS.get(scene_name, {}).get(key, [])
    return [tuple(float(value) for value in rect) for rect in rects]


def point_in_rect_with_margin(x: float, y: float, rect: tuple[float, float, float, float], margin: float) -> bool:
    xmin, xmax, ymin, ymax = rect
    return xmin + margin <= x <= xmax - margin and ymin + margin <= y <= ymax - margin


def point_hits_expanded_rect(x: float, y: float, rect: tuple[float, float, float, float], margin: float) -> bool:
    xmin, xmax, ymin, ymax = rect
    return xmin - margin <= x <= xmax + margin and ymin - margin <= y <= ymax + margin


def object_avoid_radius(obj: ObjectSpec) -> float:
    radius_by_category = {
        "Microwave": 0.34,
        "Dishwasher": 0.42,
        "Oven": 0.44,
        "Cabinet": 0.45,
        "TrashCan": 0.30,
        "Refrigerator": 0.48,
        "Mug": 0.18,
    }
    return radius_by_category.get(obj.category, 0.36) * max(obj.default_scale, 0.35)


def valid_summit_floor_position(spec: SceneSpec, x: float, y: float, radius: float = SUMMIT_COLLISION_RADIUS) -> bool:
    floor_regions = scene_rects(spec.name, "floor_regions")
    if floor_regions:
        if not any(point_in_rect_with_margin(x, y, rect, radius * 0.25) for rect in floor_regions):
            return False
    else:
        xmin, xmax, ymin, ymax = scene_bounds(spec.name)
        if not (xmin + radius <= x <= xmax - radius and ymin + radius <= y <= ymax - radius):
            return False

    for rect in scene_rects(spec.name, "floor_exclusion_rects"):
        if point_hits_expanded_rect(x, y, rect, radius):
            return False

    for obj in spec.objects:
        # Elevated objects are usually on counters, but their XY footprint still
        # marks a region we should avoid with the mobile base in poster views.
        if distance_xy((x, y), obj.translate[:2]) < radius + object_avoid_radius(obj):
            return False
    return True


def configured_summit_focus_points(spec: SceneSpec) -> list[tuple[float, float, float]]:
    points = SCENE_RENDER_CONFIGS.get(spec.name, {}).get("summit_focus_points", [])
    anchors: list[tuple[float, float, float]] = []
    for point in points:
        x, y = float(point[0]), float(point[1])
        if valid_summit_floor_position(spec, x, y, SUMMIT_COLLISION_RADIUS):
            anchors.append((x, y, 0.0))
    return anchors


def configured_summit_fixed_points(spec: SceneSpec) -> list[tuple[float, float, float]]:
    points = SCENE_RENDER_CONFIGS.get(spec.name, {}).get("summit_fixed_points", [])
    fixed: list[tuple[float, float, float]] = []
    for point in points:
        x, y = float(point[0]), float(point[1])
        if not valid_summit_floor_position(spec, x, y, SUMMIT_COLLISION_RADIUS):
            raise RuntimeError(f"Invalid hand-picked Summit point for {spec.name}: {(x, y)}")
        fixed.append((x, y, 0.0))
    return fixed


def find_detail_object(spec: SceneSpec) -> ObjectSpec | None:
    microwaves = [obj for obj in spec.objects if obj.category in MICROWAVE_CATEGORIES]
    if microwaves:
        # Prefer the common microwave_7221 when present.
        return sorted(microwaves, key=lambda obj: (obj.asset_id != "7221", obj.asset_id))[0]
    return spec.objects[0] if spec.objects else None


def random_arm_joints(
    rng: Any,
    defaults: dict[str, float],
    limits: dict[str, tuple[float, float]],
    noise_fraction: float,
) -> dict[str, float]:
    joints: dict[str, float] = {}
    for name, default in defaults.items():
        if name.startswith("base_"):
            joints[name] = default
            continue
        if "finger" in name:
            joints[name] = float(rng.uniform(*FINGER_OPEN_RANGE))
            continue
        low, high = limits.get(name, (-math.pi, math.pi))
        sigma = (high - low) * noise_fraction
        joints[name] = clamp(float(rng.normal(default, sigma)), (low, high))
    return joints


def sample_grid_positions(
    count: int,
    bounds: tuple[float, float, float, float],
    rng: Any,
    min_separation: float,
    anchors: list[tuple[float, float, float]],
    is_valid: Any | None = None,
    candidate_rects: list[tuple[float, float, float, float]] | None = None,
) -> list[tuple[float, float, float]]:
    oversample = 4 if is_valid is not None else 1
    source_rects = candidate_rects or [bounds]
    candidates: list[tuple[float, float, float]] = []
    for rect in source_rects:
        xmin, xmax, ymin, ymax = rect
        rect_count = max(count * oversample // max(len(source_rects), 1), count)
        cols = int(math.ceil(math.sqrt(rect_count * (xmax - xmin) / max(ymax - ymin, 1e-6))))
        rows = int(math.ceil(rect_count / max(cols, 1)))
        xs = [xmin + (i + 0.5) * (xmax - xmin) / cols for i in range(cols)]
        ys = [ymin + (j + 0.5) * (ymax - ymin) / rows for j in range(rows)]
        jitter_x = 0.22 * (xmax - xmin) / cols
        jitter_y = 0.22 * (ymax - ymin) / rows
        for y in ys:
            for x in xs:
                candidate = (
                    float(x + rng.uniform(-jitter_x, jitter_x)),
                    float(y + rng.uniform(-jitter_y, jitter_y)),
                    0.0,
                )
                if is_valid is None or is_valid(candidate[0], candidate[1]):
                    candidates.append(candidate)

    # Extra stochastic candidates help fill narrow walkable strips while still
    # respecting the hand-authored collision/exclusion zones.
    for _ in range(count * 80):
        rect = source_rects[int(rng.integers(0, len(source_rects)))]
        xmin, xmax, ymin, ymax = rect
        candidate = (float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax)), 0.0)
        if is_valid is None or is_valid(candidate[0], candidate[1]):
            candidates.append(candidate)
    rng.shuffle(candidates)

    selected = [anchor for anchor in anchors if is_valid is None or is_valid(anchor[0], anchor[1])]
    used: set[tuple[float, float]] = {(round(anchor[0], 4), round(anchor[1], 4)) for anchor in selected}
    for threshold in (min_separation, min_separation * 0.82, min_separation * 0.68, min_separation * 0.55):
        for candidate in candidates:
            if len(selected) >= count:
                break
            key = (round(candidate[0], 4), round(candidate[1], 4))
            if key in used:
                continue
            if all(distance_xy(candidate[:2], other[:2]) >= threshold for other in selected):
                selected.append(candidate)
                used.add(key)
        if len(selected) >= count:
            break
    return selected[:count]


def sample_summit(spec: SceneSpec, count: int, summit_robot_cfg: Path | None = None) -> list[RobotSample]:
    import numpy as np

    rng = np.random.default_rng(stable_seed(spec.name, "summit_franka"))
    bounds = scene_bounds(spec.name)
    summit_defaults = load_summit_defaults_from_yml(summit_robot_cfg)
    summit_limits = load_summit_joint_limits(summit_robot_cfg)
    detail_obj = find_detail_object(spec)
    fixed_points = configured_summit_fixed_points(spec)
    anchors = fixed_points or configured_summit_focus_points(spec)
    if detail_obj is not None and not anchors:
        obj_x, obj_y, _ = detail_obj.translate
        obj_yaw = math.radians(detail_obj.orient_deg[2])
        for angle in (obj_yaw + math.pi, obj_yaw + math.pi * 0.72, obj_yaw - math.pi * 0.72):
            x = clamp(obj_x + SUMMIT_DETAIL_ANCHOR_RADIUS * math.cos(angle), (bounds[0], bounds[1]))
            y = clamp(obj_y + SUMMIT_DETAIL_ANCHOR_RADIUS * math.sin(angle), (bounds[2], bounds[3]))
            if valid_summit_floor_position(spec, x, y, SUMMIT_COLLISION_RADIUS):
                anchors.append((x, y, 0.0))

    if fixed_points and count <= len(fixed_points):
        positions = fixed_points[:count]
    else:
        positions = sample_grid_positions(
            count,
            bounds,
            rng,
            SUMMIT_MIN_SEPARATION,
            anchors,
            is_valid=lambda x, y: valid_summit_floor_position(spec, x, y, SUMMIT_COLLISION_RADIUS),
            candidate_rects=scene_rects(spec.name, "floor_regions") or None,
        )
    samples: list[RobotSample] = []
    for index, (x, y, z) in enumerate(positions):
        if detail_obj is not None and index < len(anchors):
            yaw = math.atan2(detail_obj.translate[1] - y, detail_obj.translate[0] - x)
        else:
            yaw = float(rng.uniform(-math.pi, math.pi))
        joints = random_arm_joints(rng, summit_defaults, summit_limits, SUMMIT_ARM_NOISE_FRACTION)
        samples.append(
            RobotSample(
                name=f"summit_{index:02d}",
                position=(float(x), float(y), float(z)),
                yaw=float(yaw),
                joint_pos=joints,
            )
        )
    return samples


def elevated_candidates(spec: SceneSpec) -> list[ObjectSpec]:
    candidates = [obj for obj in spec.objects if obj.translate[2] >= ELEVATED_SURFACE_MIN_Z]
    candidates.sort(key=lambda obj: (obj.category not in MICROWAVE_CATEGORIES, obj.asset_id))
    return candidates


def configured_franka_support_points(spec: SceneSpec) -> list[tuple[float, float, float]]:
    points = SCENE_RENDER_CONFIGS.get(spec.name, {}).get("franka_support_points", [])
    return [(float(point[0]), float(point[1]), float(point[2])) for point in points]


def configured_franka_support_regions(spec: SceneSpec) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for index, region in enumerate(SCENE_RENDER_CONFIGS.get(spec.name, {}).get("franka_support_regions", [])):
        rect = tuple(float(value) for value in region["rect"])
        if len(rect) != 4:
            raise ValueError(f"Franka support region for {spec.name} must have a 4-value rect: {region}")
        regions.append(
            {
                "name": str(region.get("name", f"surface_{index:02d}")),
                "rect": rect,
                "z": float(region["z"]),
                "target": tuple(float(value) for value in region.get("target", franka_yaw_target(spec, 0.0, 0.0))),
            }
        )
    return regions


def configured_franka_avoid_points(spec: SceneSpec) -> list[tuple[float, float, float]]:
    points = SCENE_RENDER_CONFIGS.get(spec.name, {}).get("franka_avoid_points", [])
    return [(float(point[0]), float(point[1]), float(point[2])) for point in points]


def valid_franka_surface_position(
    spec: SceneSpec,
    x: float,
    y: float,
    z: float,
    region: dict[str, Any] | None = None,
) -> bool:
    if z < ELEVATED_SURFACE_MIN_Z:
        return False
    if region is not None:
        if not point_in_rect_with_margin(x, y, region["rect"], FRANKA_BASE_COLLISION_RADIUS):
            return False

    for obj in spec.objects:
        # Keep fixed-base samples away from every explicitly imported object.
        # This intentionally over-approximates collision in XY for poster safety.
        clearance = FRANKA_BASE_COLLISION_RADIUS + FRANKA_OBJECT_CLEARANCE + object_avoid_radius(obj)
        if distance_xy((x, y), obj.translate[:2]) < clearance:
            return False

    for px, py, radius in configured_franka_avoid_points(spec):
        if distance_xy((x, y), (px, py)) < FRANKA_BASE_COLLISION_RADIUS + radius:
            return False
    return True


def franka_region_candidates(region: dict[str, Any], rng: Any) -> list[tuple[float, float, float]]:
    xmin, xmax, ymin, ymax = region["rect"]
    z = region["z"]
    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    candidates = [(cx, cy, z)]

    cols = rows = 3
    for iy in range(rows):
        for ix in range(cols):
            x = xmin + (ix + 0.5) * (xmax - xmin) / cols
            y = ymin + (iy + 0.5) * (ymax - ymin) / rows
            candidates.append((x, y, z))

    for _ in range(64):
        candidates.append((float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax)), z))
    return candidates


def franka_yaw_target(spec: SceneSpec, x: float, y: float) -> tuple[float, float]:
    detail_obj = find_detail_object(spec)
    if detail_obj is not None:
        return detail_obj.translate[:2]
    xmin, xmax, ymin, ymax = scene_bounds(spec.name)
    return ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5)


def sample_franka(spec: SceneSpec, count: int) -> list[RobotSample]:
    import numpy as np

    rng = np.random.default_rng(stable_seed(spec.name, "franka"))
    support_regions = configured_franka_support_regions(spec)
    if support_regions:
        samples: list[RobotSample] = []
        used_positions: list[tuple[float, float, float]] = []
        attempts = 0
        while len(samples) < count and attempts < count * max(len(support_regions), 1) * 4:
            region = support_regions[len(samples) % len(support_regions)]
            attempts += 1
            chosen: tuple[float, float, float] | None = None
            for candidate in franka_region_candidates(region, rng):
                x, y, base_z = candidate
                z = max(base_z + FRANKA_SURFACE_Z_OFFSET, ELEVATED_SURFACE_MIN_Z + FRANKA_SURFACE_Z_OFFSET)
                if not valid_franka_surface_position(spec, x, y, z, region):
                    continue
                if any(distance_xy((x, y), other[:2]) < FRANKA_MIN_SEPARATION for other in used_positions):
                    continue
                chosen = (x, y, z)
                break
            if chosen is None:
                print(f"[render] WARNING: no valid fixed-Franka point found on {spec.name}:{region['name']}")
                # Try the next region before failing the scene.
                support_regions = support_regions[1:] + support_regions[:1]
                continue

            x, y, z = chosen
            used_positions.append(chosen)
            target_x, target_y = region["target"]
            yaw = math.atan2(target_y - y, target_x - x)
            joints = random_arm_joints(rng, FRANKA_DEFAULT_JOINTS, PANDA_JOINT_LIMITS, FRANKA_ARM_NOISE_FRACTION)
            samples.append(
                RobotSample(
                    name=f"franka_{len(samples):02d}",
                    position=(float(x), float(y), float(z)),
                    yaw=float(yaw),
                    joint_pos=joints,
                )
            )
        if len(samples) < count:
            raise RuntimeError(f"Only found {len(samples)} valid fixed-Franka tabletop samples for {spec.name}")
        return samples

    configured_supports = configured_franka_support_points(spec)
    if configured_supports:
        samples: list[RobotSample] = []
        for index in range(count):
            base_x, base_y, base_z = configured_supports[index % len(configured_supports)]
            jitter_radius = float(rng.uniform(0.0, FRANKA_SURFACE_JITTER_RADIUS))
            theta = float(rng.uniform(-math.pi, math.pi))
            x = base_x + jitter_radius * math.cos(theta)
            y = base_y + jitter_radius * math.sin(theta)
            z = max(base_z + FRANKA_SURFACE_Z_OFFSET, ELEVATED_SURFACE_MIN_Z + FRANKA_SURFACE_Z_OFFSET)
            if not valid_franka_surface_position(spec, x, y, z):
                raise RuntimeError(f"Configured fixed-Franka point {(base_x, base_y, base_z)} is invalid in {spec.name}")
            target_x, target_y = franka_yaw_target(spec, x, y)
            yaw = math.atan2(target_y - y, target_x - x)
            joints = random_arm_joints(rng, FRANKA_DEFAULT_JOINTS, PANDA_JOINT_LIMITS, FRANKA_ARM_NOISE_FRACTION)
            samples.append(
                RobotSample(
                    name=f"franka_{index:02d}",
                    position=(float(x), float(y), float(z)),
                    yaw=float(yaw),
                    joint_pos=joints,
                )
            )
        return samples

    candidates = elevated_candidates(spec)
    if not candidates:
        target = find_detail_object(spec)
        fallback_z = 0.75 if target is None else max(target.translate[2], 0.75)
        fallback = ObjectSpec(
            asset_id="fallback",
            category="Fallback",
            usd_path=Path(),
            translate=(0.0, 0.0, fallback_z),
            orient_deg=(0.0, 0.0, 0.0),
            scene_scale=(1.0, 1.0, 1.0),
            default_scale=1.0,
            scale=(1.0, 1.0, 1.0),
        )
        candidates = [fallback]

    positions: list[tuple[float, float, float, ObjectSpec]] = []
    attempts = 0
    while len(positions) < count and attempts < count * 40:
        attempts += 1
        base = candidates[len(positions) % len(candidates)]
        radius = float(rng.uniform(0.0, FRANKA_SURFACE_JITTER_RADIUS))
        theta = float(rng.uniform(-math.pi, math.pi))
        pos = (
            base.translate[0] + radius * math.cos(theta),
            base.translate[1] + radius * math.sin(theta),
            max(base.translate[2] + FRANKA_SURFACE_Z_OFFSET, ELEVATED_SURFACE_MIN_Z + FRANKA_SURFACE_Z_OFFSET),
        )
        if all(distance_xy(pos[:2], other[:2]) >= FRANKA_MIN_SEPARATION for (*other, _) in positions):
            positions.append((pos[0], pos[1], pos[2], base))
    while len(positions) < count:
        base = candidates[len(positions) % len(candidates)]
        positions.append((base.translate[0], base.translate[1], base.translate[2] + FRANKA_SURFACE_Z_OFFSET, base))

    samples: list[RobotSample] = []
    for index, (x, y, z, base) in enumerate(positions[:count]):
        yaw = math.atan2(base.translate[1] - y, base.translate[0] - x)
        if abs(base.translate[1] - y) + abs(base.translate[0] - x) < 1e-6:
            yaw = math.radians(base.orient_deg[2]) + math.pi
        joints = random_arm_joints(rng, FRANKA_DEFAULT_JOINTS, PANDA_JOINT_LIMITS, FRANKA_ARM_NOISE_FRACTION)
        samples.append(
            RobotSample(
                name=f"franka_{index:02d}",
                position=(float(x), float(y), float(z)),
                yaw=float(yaw),
                joint_pos=joints,
            )
        )
    return samples


def load_summit_defaults_from_yml(summit_robot_cfg: Path | None) -> dict[str, float]:
    if summit_robot_cfg is None or not summit_robot_cfg.exists():
        return dict(SUMMIT_DEFAULT_JOINTS)
    try:
        cfg = load_yaml(summit_robot_cfg)
        cspace = cfg["robot_cfg"]["kinematics"]["cspace"]
        names = cspace["joint_names"]
        retract = cspace["retract_config"]
        return {str(name): float(value) for name, value in zip(names, retract, strict=True)}
    except Exception as exc:
        print(f"[render] WARNING: failed to read Summit defaults from {summit_robot_cfg}: {exc}")
        return dict(SUMMIT_DEFAULT_JOINTS)


def load_summit_joint_limits(summit_robot_cfg: Path | None) -> dict[str, tuple[float, float]]:
    limits = dict(PANDA_JOINT_LIMITS)
    root = repo_root()
    urdf_path = root / "assets" / "robot" / "summit_franka" / "summit_franka.urdf"
    if summit_robot_cfg is not None and summit_robot_cfg.exists():
        try:
            cfg = load_yaml(summit_robot_cfg)
            configured_urdf = cfg["robot_cfg"]["kinematics"].get("urdf_path")
            if configured_urdf:
                urdf_path = resolve_repo_path(configured_urdf, root)
        except Exception as exc:
            print(f"[render] WARNING: failed to read Summit URDF path from {summit_robot_cfg}: {exc}")
    if not urdf_path.exists():
        return limits
    xml_root = ET.parse(urdf_path).getroot()
    for joint in xml_root.findall("joint"):
        name = joint.attrib.get("name")
        limit = joint.find("limit")
        if not name or limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            continue
        limits[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    return limits


def camera_for_view(spec: SceneSpec, view: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    config = SCENE_RENDER_CONFIGS.get(spec.name, {})
    if view == "overview":
        if "overview_eye" in config and "overview_target" in config:
            return tuple3(config["overview_eye"]), tuple3(config["overview_target"])
        xmin, xmax, ymin, ymax = scene_bounds(spec.name)
        center = ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5, 0.7)
        span = max(xmax - xmin, ymax - ymin)
        return (
            (center[0] - 0.78 * span, center[1] - 0.88 * span, 1.15 * span),
            center,
        )
    if view == "detail":
        target_obj = find_detail_object(spec)
        if target_obj is None:
            target = (0.0, 0.0, 0.7)
        else:
            target = (
                target_obj.translate[0],
                target_obj.translate[1],
                target_obj.translate[2] + DETAIL_CAMERA_TARGET_Z_OFFSET,
            )
        eye = (
            target[0] + DETAIL_CAMERA_EYE_OFFSET[0],
            target[1] + DETAIL_CAMERA_EYE_OFFSET[1],
            target[2] + DETAIL_CAMERA_EYE_OFFSET[2],
        )
        return eye, target
    raise ValueError(f"Unsupported view: {view}")


# =============================================================================
# Isaac Sim implementation
# =============================================================================


def add_repo_import_paths(root: Path) -> None:
    arena_root = root / "third_party" / "IsaacLab-Arena"
    for path in (root, arena_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def set_prim_visibility(stage: Any, prim_path: str, visible: bool) -> None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible() if visible else imageable.MakeInvisible()


def spawn_scaled_usd_reference(
    stage: Any,
    prim_path: str,
    usd_path: Path,
    translation: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    scale: tuple[float, float, float],
) -> None:
    """Spawn USD under an explicit scaled parent Xform.

    IsaacLab's UsdFileCfg also accepts a scale, but using a parent Xform makes
    the poster object default scale visible in the exported stage and avoids any
    ambiguity about whether the file-spawner consumed it.
    """

    from pxr import Gf, UsdGeom

    parent = UsdGeom.Xform.Define(stage, prim_path)
    xformable = UsdGeom.Xformable(parent.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translation))
    w, x, y, z = orientation_wxyz
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))

    asset_prim_path = f"{prim_path}/Asset"
    asset = UsdGeom.Xform.Define(stage, asset_prim_path)
    asset.GetPrim().GetReferences().AddReference(str(usd_path))


def spawn_static_scene(spec: SceneSpec) -> None:
    import isaacsim.core.utils.prims as prim_utils
    import isaaclab.sim as sim_utils
    import omni.usd

    prim_utils.create_prim("/World/Scene", "Xform")
    prim_utils.create_prim("/World/Objects", "Xform")
    prim_utils.create_prim("/World/Lights", "Xform")

    scene_cfg = sim_utils.UsdFileCfg(usd_path=str(spec.usd_path))
    scene_cfg.func("/World/Scene/iTHOR", scene_cfg)

    stage = omni.usd.get_context().get_stage()
    for obj in spec.objects:
        spawn_scaled_usd_reference(
            stage,
            f"/World/Objects/{obj.category}_{obj.asset_id}",
            obj.usd_path,
            obj.translate,
            euler_deg_to_quat_wxyz(obj.orient_deg),
            obj.scale,
        )
        print(
            f"[render] object {obj.category}/{obj.asset_id} "
            f"scene_scale={tuple(round(v, 4) for v in obj.scene_scale)} "
            f"default_scale={obj.default_scale:.6f} import_scale={tuple(round(v, 4) for v in obj.scale)}"
        )

    dome_cfg = sim_utils.DomeLightCfg(intensity=DOME_LIGHT_INTENSITY, color=(0.82, 0.84, 0.86))
    dome_cfg.func("/World/Lights/Dome", dome_cfg)
    distant_cfg = sim_utils.DistantLightCfg(intensity=DISTANT_LIGHT_INTENSITY, color=(1.0, 0.96, 0.9))
    distant_cfg.func("/World/Lights/Key", distant_cfg, orientation=euler_deg_to_quat_wxyz((-45.0, 0.0, -35.0)))


def make_robot_articulation(robot_kind: str, prim_path: str, sample: RobotSample) -> Any:
    from isaaclab.assets import Articulation

    if robot_kind == "summit_franka":
        from isaaclab_arena.embodiments.summit_franka.summit_franka import SummitFrankaSceneCfg

        robot_cfg = SummitFrankaSceneCfg().robot.replace(prim_path=prim_path)
    elif robot_kind == "franka":
        from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

        robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path=prim_path)
    else:
        raise ValueError(f"Unsupported robot kind: {robot_kind}")

    robot_cfg.init_state.pos = sample.position
    robot_cfg.init_state.rot = yaw_to_quat_wxyz(sample.yaw)
    return Articulation(cfg=robot_cfg)


def spawn_robots(robot_kind: str, samples: list[RobotSample]) -> dict[str, Any]:
    import isaacsim.core.utils.prims as prim_utils

    group_path = "/World/SummitSamples" if robot_kind == "summit_franka" else "/World/FrankaSamples"
    prim_utils.create_prim(group_path, "Xform")
    robots: dict[str, Any] = {}
    for sample in samples:
        prim_path = f"{group_path}/{sample.name}"
        robots[sample.name] = make_robot_articulation(robot_kind, prim_path, sample)
    return robots


def apply_robot_samples(robots: dict[str, Any], samples: list[RobotSample], sim_dt: float) -> None:
    import torch

    for sample in samples:
        robot = robots[sample.name]
        root_pose = robot.data.default_root_state[:, :7].clone()
        root_pose[:, :3] = torch.tensor(sample.position, device=robot.device).unsqueeze(0)
        root_pose[:, 3:7] = torch.tensor(yaw_to_quat_wxyz(sample.yaw), device=robot.device).unsqueeze(0)
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


def create_camera(width: int, height: int) -> Any:
    from isaaclab.sensors.camera import Camera, CameraCfg
    import isaaclab.sim as sim_utils

    camera_cfg = CameraCfg(
        prim_path="/World/PosterCamera",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.02, 1.0e5),
        ),
    )
    return Camera(cfg=camera_cfg)


def set_camera_view(camera: Any, sim: Any, eye: tuple[float, float, float], target: tuple[float, float, float]) -> None:
    import torch

    device = sim.device
    camera.set_world_poses_from_view(
        torch.tensor([eye], dtype=torch.float32, device=device),
        torch.tensor([target], dtype=torch.float32, device=device),
    )
    sim.set_camera_view(eye=list(eye), target=list(target))


def save_rgb_image(camera: Any, path: Path) -> None:
    import numpy as np

    rgb = camera.data.output.get("rgb")
    if rgb is None:
        raise RuntimeError("Camera did not produce an rgb output. Did you launch with --enable_cameras?")
    array = rgb[0].detach().cpu().numpy()
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(array).save(path)
    except Exception:
        import imageio.v2 as imageio

        imageio.imwrite(path, array)


def render_one(
    sim: Any,
    camera: Any,
    robots: dict[str, Any],
    stage: Any,
    robot_group_path: str,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    mode: str,
    output_path: Path,
) -> None:
    set_prim_visibility(stage, "/World/Scene", mode != "robots_only")
    set_prim_visibility(stage, "/World/Objects", mode != "robots_only")
    set_prim_visibility(stage, robot_group_path, mode != "scene_objects_only")
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


def sample_for_robot(robot_kind: str, spec: SceneSpec, count_override: int | None, args: argparse.Namespace) -> list[RobotSample]:
    scene_config = SCENE_RENDER_CONFIGS.get(spec.name, {})
    if robot_kind == "summit_franka":
        summit_cfg = resolve_repo_path(args.summit_robot_cfg, Path(args.repo_root).resolve())
        count = count_override if count_override is not None else int(scene_config.get("summit_count", SUMMIT_SAMPLE_COUNT))
        return sample_summit(spec, count, summit_cfg)
    if robot_kind == "franka":
        count = count_override if count_override is not None else int(scene_config.get("franka_count", FRANKA_SAMPLE_COUNT))
        return sample_franka(spec, count)
    raise ValueError(f"Unsupported robot kind: {robot_kind}")


def validate_robot_samples(robot_kind: str, spec: SceneSpec, samples: list[RobotSample]) -> None:
    if robot_kind == "summit_franka":
        for sample in samples:
            x, y, _ = sample.position
            if not valid_summit_floor_position(spec, x, y, SUMMIT_COLLISION_RADIUS):
                raise RuntimeError(f"Invalid Summit floor sample in {spec.name}: {sample.name} at {sample.position}")
        return

    if robot_kind == "franka":
        regions = configured_franka_support_regions(spec)
        for sample in samples:
            x, y, z = sample.position
            region = next(
                (candidate for candidate in regions if point_in_rect_with_margin(x, y, candidate["rect"], 0.0)),
                None,
            )
            if not valid_franka_surface_position(spec, x, y, z, region):
                raise RuntimeError(f"Invalid fixed-Franka surface sample in {spec.name}: {sample.name} at {sample.position}")
        for index, sample in enumerate(samples):
            for other in samples[index + 1 :]:
                if distance_xy(sample.position[:2], other.position[:2]) < FRANKA_MIN_SEPARATION:
                    raise RuntimeError(
                        f"Fixed-Franka samples too close in {spec.name}: {sample.name} and {other.name}"
                    )
        return


def render_scene_robot(
    args: argparse.Namespace,
    spec: SceneSpec,
    robot_kind: str,
    simulation_app: Any,
) -> dict[str, Any]:
    import omni.usd
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext

    root = Path(args.repo_root).resolve()
    output_name = ROBOT_OUTPUT_NAMES[robot_kind]
    out_dir = resolve_repo_path(args.output_root, root) / spec.name / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    SimulationContext.clear_instance()
    omni.usd.get_context().new_stage()
    for _ in range(2):
        simulation_app.update()

    sim_cfg = sim_utils.SimulationCfg(device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    spawn_static_scene(spec)

    count_override = args.summit_count if robot_kind == "summit_franka" else args.franka_count
    samples = sample_for_robot(robot_kind, spec, count_override, args)
    validate_robot_samples(robot_kind, spec, samples)
    robots = spawn_robots(robot_kind, samples)
    camera = create_camera(args.image_width, args.image_height)

    sim.reset()
    print(f"[render] scene={spec.name} robot={robot_kind} samples={len(samples)} objects={len(spec.objects)}")
    apply_robot_samples(robots, samples, sim.get_physics_dt())

    stage = omni.usd.get_context().get_stage()
    robot_group_path = "/World/SummitSamples" if robot_kind == "summit_franka" else "/World/FrankaSamples"
    outputs: dict[str, str] = {}
    for view in args.views:
        eye, target = camera_for_view(spec, view)
        for mode in args.modes:
            if args.skip_existing:
                output_path = out_dir / f"{view}_{mode}.png"
                if output_path.exists():
                    print(f"[render] skip existing {output_path}")
                    outputs[f"{view}_{mode}"] = str(output_path)
                    continue
            output_path = out_dir / f"{view}_{mode}.png"
            render_one(sim, camera, robots, stage, robot_group_path, eye, target, mode, output_path)
            outputs[f"{view}_{mode}"] = str(output_path)

    if SAVE_STAGE_USD or args.save_stage:
        stage_path = out_dir / "stage.usd"
        stage.GetRootLayer().Export(str(stage_path))
        outputs["stage"] = str(stage_path)

    manifest = {
        "scene": spec.name,
        "robot_kind": robot_kind,
        "scene_usd": str(spec.usd_path),
        "objects": [dataclasses.asdict(obj) | {"usd_path": str(obj.usd_path)} for obj in spec.objects],
        "samples": [dataclasses.asdict(sample) for sample in samples],
        "outputs": outputs,
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    outputs["manifest"] = str(manifest_path)

    if MAKE_CONTACT_SHEETS and not args.no_contact_sheets:
        make_contact_sheets(out_dir, args.views, args.modes)

    SimulationContext.clear_instance()
    return outputs


def make_contact_sheets(out_dir: Path, views: list[str], modes: list[str]) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"[render] contact sheets skipped: PIL unavailable ({exc})")
        return

    for view in views:
        image_paths = [out_dir / f"{view}_{mode}.png" for mode in modes]
        if not all(path.exists() for path in image_paths):
            continue
        images = [Image.open(path).convert("RGB") for path in image_paths]
        label_h = 46
        width = sum(img.width for img in images)
        height = max(img.height for img in images) + label_h
        sheet = Image.new("RGB", (width, height), (245, 245, 242))
        draw = ImageDraw.Draw(sheet)
        x = 0
        for mode, image in zip(modes, images, strict=True):
            sheet.paste(image, (x, label_h))
            draw.text((x + 18, 14), mode.replace("_", " "), fill=(25, 25, 25))
            x += image.width
        sheet_path = out_dir / f"{view}_contact_sheet.png"
        sheet.save(sheet_path)
        print(f"[render] wrote {sheet_path}")


# =============================================================================
# CLI
# =============================================================================


def add_common_args(parser: argparse.ArgumentParser) -> None:
    root = repo_root()
    parser.add_argument("--repo_root", default=str(root), help="Repository root.")
    parser.add_argument(
        "--dataset_config",
        default=str(root / ".idea" / "cuakr_ithor" / "configs" / "dataset_config.yml"),
        help="cuAKR iTHOR dataset_config.yml with scene/object placements.",
    )
    parser.add_argument(
        "--object_data",
        default=str(root / ".idea" / "cuakr_ithor" / "configs" / "object_data.json"),
        help="cuAKR object_data.json used to map object ids to categories.",
    )
    parser.add_argument("--object_root", default=str(root / "assets" / "object"), help="AutoMoMa object root.")
    parser.add_argument(
        "--summit_robot_cfg",
        default=str(root / "assets" / "robot" / "summit_franka" / "summit_franka.yml"),
        help="Summit-Franka cuRobo YAML used for joint names/defaults while sampling poster poses.",
    )
    parser.add_argument("--output_root", default=str(root / "outputs" / "paper" / "poster"), help="Output root.")
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES), help="Scene names to render.")
    parser.add_argument("--robots", nargs="+", default=list(DEFAULT_ROBOTS), choices=list(DEFAULT_ROBOTS))
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS), choices=list(DEFAULT_VIEWS))
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=list(DEFAULT_MODES))
    parser.add_argument("--summit_count", type=int, default=None, help="Override Summit-Franka sample count.")
    parser.add_argument("--franka_count", type=int, default=None, help="Override fixed-Franka sample count.")
    parser.add_argument("--image_width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--image_height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--skip_existing", action="store_true", help="Do not re-render existing PNGs.")
    parser.add_argument("--save_stage", action="store_true", help="Force exporting the composed USD stage.")
    parser.add_argument("--no_contact_sheets", action="store_true", help="Skip side-by-side contact sheets.")
    parser.add_argument(
        "--check_inputs",
        action="store_true",
        help="Only validate scene/object paths. Does not import or launch Isaac Sim.",
    )


def parse_bootstrap_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    add_common_args(parser)
    args, _ = parser.parse_known_args()
    return args


def main() -> int:
    bootstrap_args = parse_bootstrap_args()
    if bootstrap_args.check_inputs:
        return 0 if check_inputs(bootstrap_args) else 1

    root = Path(bootstrap_args.repo_root).resolve()
    add_repo_import_paths(root)

    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:  # pragma: no cover - depends on Isaac Sim environment
        print(
            "[render] ERROR: IsaacLab is not importable. Run through "
            "tools/paper/poster/run_poster_render.sh or source scripts/setup_sim_env.sh first.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    add_repo_import_paths(root)

    if not check_inputs(args):
        print("[render] ERROR: input validation failed; fix missing paths before launching Isaac Sim.", file=sys.stderr)
        return 1

    os.environ.setdefault("AUTOMOMA_OBJECT_ROOT", str(resolve_repo_path(args.object_root, root)))
    os.environ.setdefault("AUTOMOMA_ROBOT_ROOT", str(root / "assets" / "robot"))

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        specs = load_scene_specs(args)
        all_outputs: dict[str, Any] = {}
        for spec in specs:
            all_outputs[spec.name] = {}
            for robot_kind in args.robots:
                all_outputs[spec.name][robot_kind] = render_scene_robot(args, spec, robot_kind, simulation_app)
        summary_path = resolve_repo_path(args.output_root, root) / "render_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(all_outputs, f, indent=2)
        print(f"[render] summary: {summary_path}")
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
