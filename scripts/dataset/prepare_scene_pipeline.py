#!/usr/bin/env python3
"""Standalone scene preparation pipeline for AutoMoMa scene assets.

Edit the RUN_STEP_* switches below to enable/disable steps. The script does not
import other local project scripts; all processing logic is contained here.

Usage:
    python scripts/prepare_scene_pipeline.py \
        --scene-root assets/scene/infinigen/kitchen_1130_backup \
        --scene-name scene_54_seed_54

    python scripts/prepare_scene_pipeline.py \
        --scene-root assets/scene/infinigen/kitchen_1130 \
        --scene-name all
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


# =============================================================================
# User-editable configuration
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENE_ROOT = REPO_ROOT / "assets" / "scene" / "infinigen" / "kitchen_1130"

# Set these to True/False instead of passing per-step command-line flags.
RUN_STEP_1_VALIDATE = True
RUN_STEP_2_INTERACTIVE_OBJECT_TRANSFORM = True
RUN_STEP_3_COPY_REQUIREMENT = True
RUN_STEP_4_CHECK_REQUIREMENT = True
RUN_STEP_5_RESTRUCTURE_STAGE = True
RUN_STEP_6_DEACTIVATE_USD = True
RUN_STEP_7_LOWER_KITCHENSPACE_OBJECTS = False
RUN_STEP_8_FINAL_PREPARE_SCENE = True

# Backup/restore behavior.
CREATE_BACKUPS = True
RESTORE_BEFORE_FINAL_PREPARE = False
FORCE_FINAL_PREPARE = False

# Step 2 object order.
INTERACTIVE_OBJECT_IDS = ["46197", "11622", "101773", "10944", "103634", "7221"]

# Step 7 lowering amount.
KITCHENSPACE_Z_LOWERING_AMOUNT = 0.03

# Step 8 final scene offset.
FINAL_SCENE_Z_OFFSET = -0.12

# Embedded requirement.json so this file is standalone.
EMBEDDED_REQUIREMENT: dict[str, Any] = {
    "static_objects": [
        {
            "asset_type": "Microwave",
            "asset_id": "7221",
            "urdf_path": "assets/object/Microwave/7221/mobility.urdf",
            "scale": 0.3563,
        },
        {
            "asset_type": "Dishwasher",
            "asset_id": "11622",
            "urdf_path": "assets/object/Dishwasher/11622/mobility.urdf",
            "scale": 0.6446,
        },
        {
            "asset_type": "TrashCan",
            "asset_id": "103634",
            "urdf_path": "assets/object/TrashCan/103634/mobility.urdf",
            "scale": 0.48385408192053975,
        },
        {
            "asset_type": "Cabinet",
            "asset_id": "46197",
            "urdf_path": "assets/object/Cabinet/46197/mobility.urdf",
            "scale": 0.5113198146209817,
        },
        {
            "asset_type": "Refrigerator",
            "asset_id": "10944",
            "urdf_path": "assets/object/Refrigerator/10944/mobility.urdf",
            "scale": 0.9,
        },
        {
            "asset_type": "Oven",
            "asset_id": "101773",
            "urdf_path": "assets/object/Oven/101773/mobility.urdf",
            "scale": 0.7231779244463762,
        },
    ]
}


# =============================================================================
# Common helpers
# =============================================================================

def natural_sort_key(path: Path) -> list[Any]:
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", path.name)]


def iter_scenes(scene_root: Path, scene_name: str) -> list[Path]:
    if scene_name != "all":
        scene_dir = scene_root / scene_name
        return [scene_dir] if scene_dir.is_dir() else []
    return sorted(
        [p for p in scene_root.iterdir() if p.is_dir() and p.name.startswith("scene_")],
        key=natural_sort_key,
    )


def usd_path_for(scene_dir: Path) -> Path:
    return scene_dir / "export" / "export_scene.blend" / "export_scene.usdc"


def metadata_path_for(scene_dir: Path) -> Path:
    return scene_dir / "info" / "metadata.json"


def pipeline_state_path_for(scene_dir: Path) -> Path:
    return scene_dir / "info" / "prepare_scene_pipeline_state.json"


def final_state_path_for(scene_dir: Path) -> Path:
    return scene_dir / "info" / "prepare_scene_state.json"


def print_step(index: int, title: str) -> None:
    print("\n" + "=" * 80)
    print(f"STEP {index}: {title}")
    print("=" * 80)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = load_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    save_json(path, state)


def ensure_backup(src: Path, backup: Path) -> bool:
    if not src.exists():
        return False
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, backup)
        print(f"  Backed up {src.name} -> {backup.name}")
    return True


def restore_from_backup(target: Path, backup: Path) -> bool:
    if not backup.exists():
        return False
    shutil.copy2(backup, target)
    print(f"  Restored {target.name} <- {backup.name}")
    return True


# =============================================================================
# Step 1: validate scene files
# =============================================================================

def validate_scenes(scene_dirs: list[Path]) -> bool:
    ok = True
    for scene_dir in scene_dirs:
        checks = [
            metadata_path_for(scene_dir),
            scene_dir / "export" / "solve_state.json",
            usd_path_for(scene_dir),
        ]
        missing = [p for p in checks if not p.exists()]
        if missing:
            ok = False
            print(f"  [missing] {scene_dir.name}")
            for path in missing:
                print(f"    - {path}")
        else:
            print(f"  [ok] {scene_dir.name}")
    return ok


# =============================================================================
# Step 2: interactive absolute object transform updates
# =============================================================================

def parse_vector3(text: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    text = text.strip()
    if not text:
        return default
    try:
        value = ast.literal_eval(text)
    except Exception:
        value = [part.strip() for part in text.split(",")]
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("expected 3 values, e.g. (1.0, 2.0, 3.0)")
    return tuple(float(v) for v in value)


def object_translation_default(obj: dict[str, Any]) -> tuple[float, float, float]:
    position = obj.get("position")
    if isinstance(position, (tuple, list)) and len(position) == 3:
        return tuple(float(v) for v in position)

    matrix = obj.get("matrix")
    if isinstance(matrix, list) and len(matrix) >= 3 and all(len(row) >= 4 for row in matrix[:3]):
        return tuple(float(matrix[i][3]) for i in range(3))

    return (0.0, 0.0, 0.0)


def object_rotation_degrees_default(obj: dict[str, Any]) -> tuple[float, float, float]:
    rotation = obj.get("rotation")
    if isinstance(rotation, (tuple, list)) and len(rotation) == 3:
        return tuple(math.degrees(float(v)) for v in rotation)

    matrix = obj.get("matrix")
    if isinstance(matrix, list):
        matrix_np = np.array(matrix, dtype=float)
        if matrix_np.shape == (4, 4):
            upper = matrix_np[:3, :3]
            scale = np.linalg.norm(upper, axis=0)
            if np.all(scale > 0):
                pure_rot = upper / scale
                return tuple(float(v) for v in R.from_matrix(pure_rot).as_euler("xyz", degrees=True))

    return (0.0, 0.0, 0.0)


def find_object_name_by_asset_id(metadata: dict[str, Any], object_id: str) -> str | None:
    for object_name, obj_data in metadata.get("static_objects", {}).items():
        if str(obj_data.get("asset_id")) == str(object_id):
            return object_name
        if str(object_id) in object_name:
            return object_name
    return None


def euler_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    rx_mat = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
    )
    ry_mat = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )
    rz_mat = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )
    return rz_mat @ ry_mat @ rx_mat


def create_transform_matrix(
    translation: tuple[float, float, float],
    rotation_radians: tuple[float, float, float],
    scale: tuple[float, float, float],
) -> list[list[float]]:
    rotation = euler_to_rotation_matrix(*rotation_radians)
    scaled_rotation = rotation * np.array(scale)[np.newaxis, :]
    matrix = np.eye(4)
    matrix[:3, :3] = scaled_rotation
    matrix[:3, 3] = translation
    return matrix.tolist()


def transform_bbox_corners(
    original_corners: list[list[float]],
    old_matrix: np.ndarray,
    new_matrix: np.ndarray,
) -> list[list[float]]:
    corners = np.array(original_corners)
    corners_h = np.hstack([corners, np.ones((len(corners), 1))])
    local_corners = (np.linalg.inv(old_matrix) @ corners_h.T).T
    new_corners = (new_matrix @ local_corners.T).T
    return new_corners[:, :3].tolist()


def update_object_transform(
    metadata_path: Path,
    object_name: str,
    new_translation: tuple[float, float, float],
    new_rotation_radians: tuple[float, float, float],
) -> None:
    metadata = load_json(metadata_path)
    obj = metadata["static_objects"][object_name]
    old_matrix = np.array(obj["matrix"])
    scale = tuple(obj.get("scale", [1.0, 1.0, 1.0]))
    new_matrix = create_transform_matrix(new_translation, new_rotation_radians, scale)

    obj["matrix"] = new_matrix
    obj["position"] = list(new_translation)
    obj["rotation"] = list(new_rotation_radians)
    if "bbox_corners" in obj:
        obj["bbox_corners"] = transform_bbox_corners(
            obj["bbox_corners"], old_matrix, np.array(new_matrix)
        )
    save_json(metadata_path, metadata)

    print(f"    Updated object: {object_name}")
    print(f"      position: {new_translation}")
    print(f"      rotation radians: {new_rotation_radians}")


def run_interactive_object_transform(scene_dirs: list[Path]) -> bool:
    ok = True
    for scene_dir in scene_dirs:
        metadata_path = metadata_path_for(scene_dir)
        if not metadata_path.exists():
            ok = False
            print(f"  [missing metadata] {metadata_path}")
            continue

        if CREATE_BACKUPS:
            ensure_backup(metadata_path, metadata_path.with_name("metadata_backup.json"))

        metadata = load_json(metadata_path)
        print(f"\n  Scene: {scene_dir.name}")
        print(f"  Metadata: {metadata_path}")

        for object_id in INTERACTIVE_OBJECT_IDS:
            object_name = find_object_name_by_asset_id(metadata, object_id)
            if object_name is None:
                print(f"    [{object_id}] not found, skipping")
                continue

            obj = metadata["static_objects"][object_name]
            default_translation = object_translation_default(obj)
            default_rotation_deg = object_rotation_degrees_default(obj)
            print("\n" + "-" * 80)
            print(f"    Object ID: {object_id}")
            print(f"    Object name: {object_name}")
            print(f"    Current position: {obj.get('position')}")
            print(f"    Current rotation: {obj.get('rotation')}")
            answer = input("    Update this object? [y/N]: ").strip().lower()
            if answer not in {"y", "yes"}:
                print("    skipped")
                continue

            try:
                translation = parse_vector3(
                    input(f"    New absolute translation {default_translation}: "),
                    default_translation,
                )
                rotation_deg = parse_vector3(
                    input(f"    New absolute rotation degrees {default_rotation_deg}: "),
                    default_rotation_deg,
                )
            except ValueError as exc:
                ok = False
                print(f"    invalid input: {exc}")
                continue

            rotation_rad = tuple(math.radians(v) for v in rotation_deg)
            try:
                update_object_transform(metadata_path, object_name, translation, rotation_rad)
                metadata = load_json(metadata_path)
            except Exception as exc:
                ok = False
                print(f"    failed to update {object_id}: {exc}")

    print("\n" + "!" * 80)
    print("  WARNING: Save the USD file manually before continuing.")
    print("  Later steps assume the scene USD has the matching manual edits.")
    print("!" * 80)
    answer = input("  Have you saved the USD file manually? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        print("  Stopping before later pipeline steps. Save the USD file, then run again.")
        return False
    return ok


# =============================================================================
# Step 3/4: requirement copy/check
# =============================================================================

def copy_requirement(scene_dirs: list[Path]) -> bool:
    ok = True
    for scene_dir in scene_dirs:
        dst = scene_dir / "info" / "requirement.json"
        try:
            save_json(dst, EMBEDDED_REQUIREMENT)
            print(f"  [wrote] {dst}")
        except Exception as exc:
            ok = False
            print(f"  [failed] {scene_dir.name}: {exc}")
    return ok


def check_requirement(scene_dirs: list[Path]) -> bool:
    ok = True
    for scene_dir in scene_dirs:
        metadata_path = metadata_path_for(scene_dir)
        requirement_path = scene_dir / "info" / "requirement.json"
        if not metadata_path.exists() or not requirement_path.exists():
            ok = False
            print(f"  [missing] {scene_dir.name}: metadata.json or requirement.json")
            continue

        metadata = load_json(metadata_path)
        requirement = load_json(requirement_path)
        static_objects = metadata.get("static_objects", {})
        required = requirement.get("static_objects", [])

        missing = []
        for obj in required:
            asset_type = obj.get("asset_type")
            asset_id = str(obj.get("asset_id"))
            found = any(asset_type in name and asset_id in name for name in static_objects)
            if not found:
                missing.append(f"{asset_type}:{asset_id}")

        if missing:
            ok = False
            print(f"  [invalid] {scene_dir.name}: missing {missing}")
        else:
            print(f"  [valid] {scene_dir.name}: {len(required)}/{len(required)} required objects")
    return ok


# =============================================================================
# Step 5: USD restructure
# =============================================================================

def step5_restructure_stage(scene_dirs: list[Path]) -> bool:
    from pxr import Sdf, Usd

    ok = True
    for scene_dir in scene_dirs:
        usd_path = usd_path_for(scene_dir)
        print(f"\n  Scene: {scene_dir.name}")
        try:
            stage = Usd.Stage.Open(str(usd_path))
            if not stage:
                raise RuntimeError(f"failed to open USD: {usd_path}")

            new_parent_path = Sdf.Path("/World/scene")
            if stage.GetPrimAtPath(new_parent_path):
                print("    /World/scene already exists, skipping restructure")
                continue

            stage.DefinePrim(new_parent_path, "Xform")
            world_prim = stage.GetPrimAtPath("/World")
            if not world_prim:
                raise RuntimeError("no /World prim found")

            root_layer = stage.GetRootLayer()
            children_paths = [child.GetPath() for child in world_prim.GetChildren()]
            prims_to_move = [
                path
                for path in children_paths
                if path != new_parent_path
                and not path.HasPrefix(new_parent_path)
                and "_materials" not in path.name
                and "material" not in path.name.lower()
            ]

            moved_count = 0
            for old_path in prims_to_move:
                new_path = new_parent_path.AppendChild(old_path.name)
                print(f"    Moving {old_path} -> {new_path}")
                if Sdf.CopySpec(root_layer, old_path, root_layer, new_path):
                    stage.RemovePrim(old_path)
                    moved_count += 1
                else:
                    ok = False
                    print(f"    failed to move {old_path}")

            stage.Save()
            print(f"    moved {moved_count} prims under /World/scene")
        except Exception as exc:
            ok = False
            print(f"    error: {exc}")
    return ok


# =============================================================================
# Step 6: USD deactivate
# =============================================================================

def step6_deactivate_usd(scene_dirs: list[Path]) -> bool:
    from pxr import Usd

    keywords = ["exterior", "ceiling", "Ceiling", "camera_"]
    ok = True
    for scene_dir in scene_dirs:
        usd_path = usd_path_for(scene_dir)
        print(f"\n  Scene: {scene_dir.name}")
        try:
            stage = Usd.Stage.Open(str(usd_path))
            if not stage:
                raise RuntimeError(f"failed to open USD: {usd_path}")

            count = 0
            for prim in stage.Traverse():
                prim_path = str(prim.GetPath())
                if any(keyword in prim_path for keyword in keywords) and prim.IsActive():
                    prim.SetActive(False)
                    print(f"    Deactivated: {prim_path}")
                    count += 1
            stage.Save()
            print(f"    deactivated {count} prims")
        except Exception as exc:
            ok = False
            print(f"    error: {exc}")
    return ok


# =============================================================================
# Step 7: lower KitchenSpaceFactory tabletop objects
# =============================================================================

def kitchen_space_objects_from_solve_state(scene_dir: Path) -> list[tuple[str, str, dict[str, Any]]]:
    solve_state_path = scene_dir / "export" / "solve_state.json"
    if not solve_state_path.exists():
        return []

    solve_state = load_json(solve_state_path)
    objects = []
    seen = set()
    for obj_id, obj_data in solve_state.get("objs", {}).items():
        if not isinstance(obj_data, dict):
            continue
        for relation in obj_data.get("relations", []):
            if "KitchenSpaceFactory" in relation.get("target_name", "") and obj_id not in seen:
                obj_str = obj_data.get("obj", "")
                if obj_str:
                    objects.append((obj_id, obj_str, obj_data))
                    seen.add(obj_id)
    return objects


def search_patterns_for_solve_object(obj_str: str) -> list[str]:
    patterns = []
    if "StaticCategoryFactory" in obj_str:
        match = re.search(r"StaticCategoryFactory\((\w+)_(\d+)", obj_str)
        if match:
            patterns.append(f"StaticCategoryFactory_{match.group(1)}_{match.group(2)}")
    else:
        match = re.search(r"(\w+)\((\d+)\)\.spawn_asset\((\d+)\)", obj_str)
        if match:
            _factory_name, factory_num, spawn_num = match.groups()
            patterns.extend([spawn_num, factory_num])
        else:
            match = re.search(r"(\w+)\((\d+)", obj_str)
            if match:
                patterns.append(match.group(2))
    return patterns


def lower_matching_usd_prims(scene_dir: Path, objects_to_lower: list[tuple[str, str, dict[str, Any]]]) -> bool:
    from pxr import Usd

    usd_path = usd_path_for(scene_dir)
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"failed to open USD: {usd_path}")

    lowered_prims = set()
    lowered_count = 0
    for _obj_id, obj_str, _obj_data in objects_to_lower:
        patterns = search_patterns_for_solve_object(obj_str)
        found = False
        print(f"    Searching USD for {patterns}")
        for pattern in patterns:
            for prim in stage.Traverse():
                prim_path = str(prim.GetPath())
                prim_id = prim.GetPath()
                if prim_id in lowered_prims:
                    continue
                if pattern in prim_path and prim.GetTypeName() == "Xform":
                    attr = prim.GetAttribute("xformOp:translate")
                    current = attr.Get() if attr else None
                    if current and len(current) >= 3:
                        new = (current[0], current[1], current[2] - KITCHENSPACE_Z_LOWERING_AMOUNT)
                        attr.Set(new)
                        lowered_prims.add(prim_id)
                        lowered_count += 1
                        found = True
                        print(f"      Lowered {prim_path}: z {current[2]:.5f} -> {new[2]:.5f}")
                        break
            if found:
                break
        if not found:
            print(f"      no USD prim found for {patterns}: {obj_str}")

    stage.Save()
    print(f"    lowered {lowered_count} USD prims")
    return True


def lower_matching_metadata_objects(scene_dir: Path, objects_to_lower: list[tuple[str, str, dict[str, Any]]]) -> bool:
    metadata_path = metadata_path_for(scene_dir)
    if not metadata_path.exists():
        print(f"    metadata not found: {metadata_path}")
        return False

    metadata = load_json(metadata_path)
    updated = set()
    updated_count = 0
    for _obj_id, obj_str, _obj_data in objects_to_lower:
        matched_name = None
        if "StaticCategoryFactory" in obj_str:
            match = re.search(r"StaticCategoryFactory\((\w+)_(\d+)", obj_str)
            if match:
                obj_type, asset_id = match.groups()
                for name in metadata.get("static_objects", {}):
                    if name not in updated and obj_type in name and asset_id in name:
                        matched_name = name
                        break
        else:
            spawn_match = re.search(r"\.spawn_asset\((\d+)\)", obj_str)
            spawn_num = spawn_match.group(1) if spawn_match else None
            if spawn_num:
                for name, data in metadata.get("static_objects", {}).items():
                    if name not in updated and (spawn_num in name or spawn_num in str(data)):
                        matched_name = name
                        break

        if not matched_name:
            print(f"      no metadata object found for {obj_str}")
            continue

        obj = metadata["static_objects"][matched_name]
        if "position" in obj and len(obj["position"]) >= 3:
            old_z = obj["position"][2]
            obj["position"][2] -= KITCHENSPACE_Z_LOWERING_AMOUNT
            if "matrix" in obj and len(obj["matrix"]) >= 3 and len(obj["matrix"][2]) >= 4:
                obj["matrix"][2][3] -= KITCHENSPACE_Z_LOWERING_AMOUNT
            if "bbox_corners" in obj:
                for corner in obj["bbox_corners"]:
                    if len(corner) >= 3:
                        corner[2] -= KITCHENSPACE_Z_LOWERING_AMOUNT
            updated.add(matched_name)
            updated_count += 1
            print(f"      Updated metadata {matched_name}: z {old_z:.5f} -> {obj['position'][2]:.5f}")

    save_json(metadata_path, metadata)
    print(f"    updated {updated_count} metadata objects")
    return True


def step7_lower_kitchenspace_objects(scene_dirs: list[Path]) -> bool:
    ok = True
    for scene_dir in scene_dirs:
        print(f"\n  Scene: {scene_dir.name}")
        try:
            objects_to_lower = kitchen_space_objects_from_solve_state(scene_dir)
            if not objects_to_lower:
                print("    no objects on KitchenSpaceFactory")
                continue
            for obj_id, obj_str, _obj_data in objects_to_lower:
                print(f"    Found KitchenSpaceFactory object: {obj_id} -> {obj_str}")
            lower_matching_usd_prims(scene_dir, objects_to_lower)
            lower_matching_metadata_objects(scene_dir, objects_to_lower)
        except Exception as exc:
            ok = False
            print(f"    error: {exc}")
    return ok


# =============================================================================
# Step 8: final prepare_scene fixes
# =============================================================================

def self_rotate_z_180(matrix: np.ndarray) -> np.ndarray:
    result = matrix.copy()
    result[:3, :3] = matrix[:3, :3] @ R.from_euler("z", np.pi).as_matrix()
    return result


def transform_corners(corners: list[list[float]], old_matrix: np.ndarray, new_matrix: np.ndarray) -> list[list[float]]:
    if not corners:
        return []
    corners_np = np.array(corners)
    homo = np.hstack([corners_np, np.ones((corners_np.shape[0], 1))])
    try:
        old_inv = np.linalg.inv(old_matrix)
    except np.linalg.LinAlgError:
        return corners
    local = (old_inv @ homo.T).T
    new = (new_matrix @ local.T).T
    return new[:, :3].tolist()


def fix_metadata_rotation(metadata_path: Path) -> bool:
    data = load_json(metadata_path)
    for _key, obj in data.get("static_objects", {}).items():
        old_matrix = np.array(obj["matrix"])
        new_matrix = self_rotate_z_180(old_matrix)
        upper = new_matrix[:3, :3]
        scale = np.linalg.norm(upper, axis=0)
        pure_rot = upper / scale
        obj["rotation"] = R.from_matrix(pure_rot).as_euler("xyz").tolist()
        if "bbox_corners" in obj:
            obj["bbox_corners"] = transform_corners(obj["bbox_corners"], old_matrix, new_matrix)
        obj["matrix"] = new_matrix.tolist()
    save_json(metadata_path, data)
    print(f"    [rotation] fixed {metadata_path}")
    return True


def fix_metadata_z_offset(metadata_path: Path) -> bool:
    data = load_json(metadata_path)
    for _key, obj in data.get("static_objects", {}).items():
        if "position" in obj and len(obj["position"]) == 3:
            obj["position"][2] += FINAL_SCENE_Z_OFFSET
        if "matrix" in obj:
            matrix = np.array(obj["matrix"])
            if matrix.shape == (4, 4):
                matrix[2, 3] += FINAL_SCENE_Z_OFFSET
                obj["matrix"] = matrix.tolist()
        if "bbox_corners" in obj:
            obj["bbox_corners"] = [
                [corner[0], corner[1], corner[2] + FINAL_SCENE_Z_OFFSET]
                if len(corner) == 3 else corner
                for corner in obj["bbox_corners"]
            ]
    save_json(metadata_path, data)
    print(f"    [metadata z-offset] fixed {metadata_path}")
    return True


def fix_usd_scene_z_offset(scene_dir: Path) -> bool:
    from pxr import Gf, Usd, UsdGeom

    usd_path = usd_path_for(scene_dir)
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print(f"    [USD z-offset] failed to open USD: {usd_path}")
        return False

    prim = stage.GetPrimAtPath("/World/scene")
    if not prim:
        print("    [USD z-offset] /World/scene prim not found")
        return False

    xform = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xform.AddTranslateOp()

    current = translate_op.Get() or Gf.Vec3d(0, 0, 0)
    new = Gf.Vec3d(current[0], current[1], current[2] + FINAL_SCENE_Z_OFFSET)
    translate_op.Set(new)
    stage.Save()
    print(f"    [USD z-offset] /World/scene: {current} -> {new}")
    return True


def step8_final_prepare_scene(scene_dirs: list[Path]) -> bool:
    ok = True
    for scene_dir in scene_dirs:
        print(f"\n  Scene: {scene_dir.name}")
        metadata_path = metadata_path_for(scene_dir)
        metadata_backup = metadata_path.with_name("metadata_backup.json")
        usd_path = usd_path_for(scene_dir)
        usd_backup = usd_path.with_name("export_scene_backup.usdc")
        state_path = final_state_path_for(scene_dir)
        state = load_state(state_path)

        if not metadata_path.exists():
            ok = False
            print(f"    missing metadata: {metadata_path}")
            continue

        if RESTORE_BEFORE_FINAL_PREPARE:
            restored = False
            restored |= restore_from_backup(metadata_path, metadata_backup)
            restored |= restore_from_backup(usd_path, usd_backup)
            if restored:
                state = {}

        if CREATE_BACKUPS:
            ensure_backup(metadata_path, metadata_backup)
            # ensure_backup(usd_path, usd_backup)

        if state.get("metadata_rotation_fixed") and not FORCE_FINAL_PREPARE:
            print("    [rotation] skipped")
        else:
            ok = fix_metadata_rotation(metadata_path) and ok
            state["metadata_rotation_fixed"] = True

        if state.get("metadata_z_offset_fixed") and not FORCE_FINAL_PREPARE:
            print("    [metadata z-offset] skipped")
        else:
            ok = fix_metadata_z_offset(metadata_path) and ok
            state["metadata_z_offset_fixed"] = True

        if state.get("usd_z_offset_fixed") and not FORCE_FINAL_PREPARE:
            print("    [USD z-offset] skipped")
        else:
            if fix_usd_scene_z_offset(scene_dir):
                state["usd_z_offset_fixed"] = True
            else:
                ok = False

        save_state(state_path, state)
        print(f"    [state] {state_path}")
    return ok


# =============================================================================
# Main
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone AutoMoMa scene preparation pipeline.")
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--scene-name", default="all", help="Scene name, e.g. scene_0_seed_0, or all")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scene_root = args.scene_root.resolve()
    if not scene_root.exists():
        print(f"Scene root not found: {scene_root}")
        return 1

    scene_dirs = iter_scenes(scene_root, args.scene_name)
    if not scene_dirs:
        print(f"No scenes found for scene_name={args.scene_name!r} under {scene_root}")
        return 1

    print(f"Scene root: {scene_root}")
    print(f"Scenes: {len(scene_dirs)}")
    success = True

    if RUN_STEP_1_VALIDATE:
        print_step(1, "Validate raw scene files")
        success = validate_scenes(scene_dirs) and success

    if RUN_STEP_2_INTERACTIVE_OBJECT_TRANSFORM:
        print_step(2, "Interactive absolute object transform updates")
        step_2_success = run_interactive_object_transform(scene_dirs)
        success = step_2_success and success
        if not step_2_success:
            return 1
    else:
        print_step(2, "Interactive absolute object transform updates")
        print("  [skipped] Set RUN_STEP_2_INTERACTIVE_OBJECT_TRANSFORM=True to enable")

    if RUN_STEP_3_COPY_REQUIREMENT:
        print_step(3, "Write embedded requirement.json")
        success = copy_requirement(scene_dirs) and success

    if RUN_STEP_4_CHECK_REQUIREMENT:
        print_step(4, "Check metadata against requirement.json")
        success = check_requirement(scene_dirs) and success

    if RUN_STEP_5_RESTRUCTURE_STAGE:
        print_step(5, "Restructure USD under /World/scene")
        success = step5_restructure_stage(scene_dirs) and success

    if RUN_STEP_6_DEACTIVATE_USD:
        print_step(6, "Deactivate ceiling/exterior USD prims")
        success = step6_deactivate_usd(scene_dirs) and success

    if RUN_STEP_7_LOWER_KITCHENSPACE_OBJECTS:
        print_step(7, "Lower KitchenSpaceFactory tabletop objects by 0.03m")
        success = step7_lower_kitchenspace_objects(scene_dirs) and success

    if RUN_STEP_8_FINAL_PREPARE_SCENE:
        print_step(8, "Final prepare_scene fixes")
        success = step8_final_prepare_scene(scene_dirs) and success

    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Processed scenes: {len(scene_dirs)}")
    print(f"Status: {'success' if success else 'completed with errors'}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
