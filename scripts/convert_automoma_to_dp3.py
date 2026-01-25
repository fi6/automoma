#!/usr/bin/env python3
"""
Convert Automoma camera_data episodes to DP3 zarr format.

Modes:
  collect: scan output directories and update convert_statistic.json with state
  convert: convert collected episodes into one zarr per scene-object
  check:   verify zarr files and mark as converted
  clean:   delete hdf5 files for converted entries (keep folders/json)
"""

import argparse
import gc
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime

import h5py
import numpy as np
import zarr

# Add RoboTwin policy config to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POLICY_DIR = os.path.join(PROJECT_ROOT, "baseline", "RoboTwin", "policy")
if POLICY_DIR not in sys.path:
    sys.path.append(POLICY_DIR)

from config import CollectConfig


def load_hdf5(dataset_path, mobile_base_mode="relative"):
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        robot_name = root["env_info"]["robot_name"][()].decode("utf-8")
        joint_group = root["/obs/joint"]

        robot_config = CollectConfig.JOINT_CONFIG.get(robot_name, {})
        output_joints = robot_config.get("output_joints", {})
        if not output_joints:
            raise ValueError(f"No output_joints config for robot: {robot_name}")

        joint_data = {}
        for joint_name in output_joints.keys():
            if joint_name in joint_group:
                joint_data[joint_name] = joint_group[joint_name][()]

        pointcloud = root["/obs/point_cloud"][()]
        eef_data = root["/obs/eef"][()] if "eef" in root["/obs"] else None

        timestamps = pointcloud.shape[0]
        joint_states = {joint_name: [] for joint_name in output_joints.keys()}

        for t in range(timestamps):
            for joint_name, dim in output_joints.items():
                if joint_name in joint_data:
                    if len(joint_data[joint_name].shape) == 1:
                        joint_states[joint_name].append([joint_data[joint_name][t]])
                    else:
                        joint_states[joint_name].append(joint_data[joint_name][t][:dim])

        for joint_name in joint_states:
            joint_states[joint_name] = np.array(joint_states[joint_name])
            if len(joint_states[joint_name].shape) == 1:
                joint_states[joint_name] = joint_states[joint_name].reshape(-1, 1)

        state_arrays = np.concatenate(
            [joint_states[joint_name] for joint_name in output_joints.keys()], axis=1
        )

        if mobile_base_mode == "absolute":
            actions = state_arrays[1:]
        elif mobile_base_mode == "relative":
            action_components = {}
            for joint_name in output_joints.keys():
                if joint_name in joint_states:
                    if joint_name == "mobile_base":
                        action_components[joint_name] = (
                            joint_states[joint_name][1:] - joint_states[joint_name][:-1]
                        )
                    else:
                        action_components[joint_name] = joint_states[joint_name][1:]
            actions = np.concatenate(
                [action_components[joint_name] for joint_name in output_joints.keys()], axis=1
            )
        else:
            raise ValueError(
                f"Invalid mobile_base_mode: {mobile_base_mode}. Choose 'absolute' or 'relative'."
            )

        return pointcloud[:-1], state_arrays[:-1], actions, eef_data, robot_name


def find_traj_roots(output_dir):
    traj_roots = []
    for root, dirs, _ in os.walk(output_dir):
        if os.path.basename(root) == "traj":
            traj_roots.append(root)
    return sorted(set(traj_roots))


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def ensure_stat_structure(stat):
    if not stat:
        stat = {"task_name": "automoma_camera_collection", "statics": {}}
    if "statics" not in stat:
        stat["statics"] = {}
    return stat


def collect_mode(output_dir, stat_path):
    stat = ensure_stat_structure(load_json(stat_path))

    traj_roots = find_traj_roots(output_dir)
    if not traj_roots:
        print(f"No traj directories found under {output_dir}")
        save_json(stat_path, stat)
        return

    for traj_root in traj_roots:
        for robot_name in os.listdir(traj_root):
            robot_path = os.path.join(traj_root, robot_name)
            if not os.path.isdir(robot_path):
                continue
            if robot_name.endswith(".json"):
                continue

            robot_stat = stat["statics"].setdefault(robot_name, {})

            for scene_name in os.listdir(robot_path):
                scene_path = os.path.join(robot_path, scene_name)
                if not os.path.isdir(scene_path):
                    continue
                if not scene_name.startswith("scene_"):
                    continue

                scene_stat = robot_stat.setdefault(scene_name, {})

                for object_id in os.listdir(scene_path):
                    object_path = os.path.join(scene_path, object_id)
                    if not os.path.isdir(object_path):
                        continue
                    if object_id.startswith("grasp_") or object_id.endswith(".json"):
                        continue

                    camera_data_path = os.path.join(object_path, "camera_data")
                    if not os.path.exists(camera_data_path):
                        continue

                    metadata_path = os.path.join(object_path, "collection_metadata.json")
                    has_metadata = os.path.exists(metadata_path)

                    hdf5_files = [
                        f for f in os.listdir(camera_data_path) if f.endswith(".hdf5")
                    ]
                    hdf5_count = len(hdf5_files)

                    num = hdf5_count
                    if has_metadata:
                        try:
                            meta = load_json(metadata_path)
                            num = int(meta.get("episodes_recorded", hdf5_count))
                        except Exception:
                            num = hdf5_count

                    entry = scene_stat.get(object_id, {})
                    prev_state = entry.get("state")
                    state = prev_state
                    if prev_state not in {"converted", "cleaned"}:
                        state = "collected" if has_metadata else "collecting"

                    source_paths = entry.get("source_paths", [])
                    if camera_data_path not in source_paths:
                        source_paths.append(camera_data_path)

                    scene_stat[object_id] = {
                        "num": num,
                        "state": state,
                        "source_paths": sorted(source_paths),
                        "updated_at": datetime.now().isoformat(),
                    }

    save_json(stat_path, stat)
    print(f"Updated convert statistics at {stat_path}")


def build_zarr_name(robot_name, scene_name, object_id, num):
    return f"automoma_manip_{robot_name}-task_{object_id}_{scene_name}-{num}.zarr"


def get_all_hdf5_files(source_paths):
    all_files = []
    for path in source_paths:
        if not os.path.exists(path):
            continue
        for f in os.listdir(path):
            if f.endswith(".hdf5"):
                all_files.append(os.path.join(path, f))
    return sorted(all_files)


def convert_dataset(hdf5_files, save_path, mobile_base_mode="relative", batch_size=100):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    # Find first valid episode for shapes
    pointcloud_sample = None
    state_sample = None
    action_sample = None
    robot_name = None
    for file_path in hdf5_files[:10]:
        try:
            pointcloud_sample, state_sample, action_sample, _, robot_name = load_hdf5(
                file_path, mobile_base_mode
            )
            if pointcloud_sample is not None:
                break
        except Exception:
            continue

    if pointcloud_sample is None:
        raise RuntimeError("No valid episodes found for conversion.")

    zarr_root = zarr.group(save_path)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    zarr_point_cloud = zarr_data.create_dataset(
        "point_cloud",
        shape=(0,) + pointcloud_sample.shape[1:],
        chunks=(100,) + pointcloud_sample.shape[1:],
        dtype=pointcloud_sample.dtype,
        compressor=compressor,
    )
    zarr_state = zarr_data.create_dataset(
        "state",
        shape=(0,) + state_sample.shape[1:],
        chunks=(100,) + state_sample.shape[1:],
        dtype="float32",
        compressor=compressor,
    )
    zarr_action = zarr_data.create_dataset(
        "action",
        shape=(0,) + action_sample.shape[1:],
        chunks=(100,) + action_sample.shape[1:],
        dtype="float32",
        compressor=compressor,
    )

    current_idx = 0
    episode_ends = []

    for i in range(0, len(hdf5_files), batch_size):
        batch_files = hdf5_files[i : i + batch_size]
        batch_point_clouds = []
        batch_states = []
        batch_actions = []

        for file_path in batch_files:
            try:
                pointcloud_all, state_all, action_all, _, _ = load_hdf5(
                    file_path, mobile_base_mode
                )
                batch_point_clouds.extend(pointcloud_all)
                batch_states.extend(state_all)
                batch_actions.extend(action_all)
                current_idx += len(pointcloud_all)
                episode_ends.append(current_idx)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        if batch_point_clouds:
            batch_point_clouds = np.array(batch_point_clouds)
            batch_states = np.array(batch_states)
            batch_actions = np.array(batch_actions)

            old_size = zarr_point_cloud.shape[0]
            new_size = old_size + len(batch_point_clouds)

            zarr_point_cloud.resize(new_size, *zarr_point_cloud.shape[1:])
            zarr_state.resize(new_size, *zarr_state.shape[1:])
            zarr_action.resize(new_size, *zarr_action.shape[1:])

            zarr_point_cloud[old_size:new_size] = batch_point_clouds
            zarr_state[old_size:new_size] = batch_states
            zarr_action[old_size:new_size] = batch_actions

            del batch_point_clouds, batch_states, batch_actions
            gc.collect()

    zarr_meta.create_dataset(
        "episode_ends",
        data=np.array(episode_ends),
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )
    if robot_name:
        zarr_meta.attrs["robot_name"] = robot_name


def convert_mode(stat_path, zarr_dir, mobile_base_mode="relative", batch_size=100):
    stat = ensure_stat_structure(load_json(stat_path))
    os.makedirs(zarr_dir, exist_ok=True)

    for robot_name, robot_data in stat["statics"].items():
        for scene_name, scene_data in robot_data.items():
            for object_id, entry in scene_data.items():
                if entry.get("state") != "collected":
                    continue

                source_paths = entry.get("source_paths", [])
                hdf5_files = get_all_hdf5_files(source_paths)
                if not hdf5_files:
                    print(f"No hdf5 files found for {robot_name}/{scene_name}/{object_id}")
                    continue

                num = len(hdf5_files)
                zarr_name = build_zarr_name(robot_name, scene_name, object_id, num)
                zarr_path = os.path.join(zarr_dir, zarr_name)

                print(f"Converting {robot_name}/{scene_name}/{object_id} -> {zarr_name}")
                convert_dataset(
                    hdf5_files,
                    zarr_path,
                    mobile_base_mode=mobile_base_mode,
                    batch_size=batch_size,
                )

                entry["num"] = num
                entry["zarr_path"] = zarr_path
                entry["state"] = "converted"
                entry["updated_at"] = datetime.now().isoformat()

    save_json(stat_path, stat)
    print(f"Conversion done. Updated {stat_path}")


def check_mode(stat_path, zarr_dir=None):
    stat = ensure_stat_structure(load_json(stat_path))

    for robot_name, robot_data in stat["statics"].items():
        for scene_name, scene_data in robot_data.items():
            for object_id, entry in scene_data.items():
                if entry.get("state") not in {"collected", "converted"}:
                    continue

                zarr_path = entry.get("zarr_path")
                if not zarr_path and zarr_dir:
                    num = entry.get("num")
                    if num is None:
                        hdf5_files = get_all_hdf5_files(entry.get("source_paths", []))
                        num = len(hdf5_files)
                        entry["num"] = num
                    zarr_name = build_zarr_name(robot_name, scene_name, object_id, num)
                    zarr_path = os.path.join(zarr_dir, zarr_name)
                    entry["zarr_path"] = zarr_path

                if not zarr_path or not os.path.exists(zarr_path):
                    print(f"Zarr file missing for entry: {entry}")
                    raise FileNotFoundError(f"Zarr file not found: {zarr_path}")

                try:
                    zarr_root = zarr.open(zarr_path, mode="r")
                    data_group = zarr_root.get("data")
                    meta_group = zarr_root.get("meta")
                    if data_group is None or meta_group is None:
                        raise ValueError(f"Invalid zarr structure in: {zarr_path}")
                    if data_group["point_cloud"].shape[0] == 0:
                        raise ValueError(f"No data in zarr file: {zarr_path}")
                    if "episode_ends" not in meta_group:
                        raise ValueError(f"No episode_ends in zarr meta: {zarr_path}")

                    entry["state"] = "converted"
                    entry["updated_at"] = datetime.now().isoformat()
                except Exception:
                    raise RuntimeError(f"Failed to validate zarr file: {zarr_path}")

    save_json(stat_path, stat)
    print(f"Check done. Updated {stat_path}")


def clean_mode(stat_path):
    stat = ensure_stat_structure(load_json(stat_path))

    for robot_data in stat["statics"].values():
        for scene_data in robot_data.values():
            for entry in scene_data.values():
                if entry.get("state") != "converted":
                    continue

                zarr_path = entry.get("zarr_path")
                if not zarr_path or not os.path.exists(zarr_path):
                    print(f"Skip clean: zarr missing for entry: {entry}")
                    continue

                try:
                    zarr_root = zarr.open(zarr_path, mode="r")
                    data_group = zarr_root.get("data")
                    meta_group = zarr_root.get("meta")
                    if data_group is None or meta_group is None:
                        raise ValueError("Invalid zarr structure")
                    if data_group["point_cloud"].shape[0] == 0:
                        raise ValueError("No data in zarr")
                    if "episode_ends" not in meta_group:
                        raise ValueError("No episode_ends in zarr meta")
                except Exception as e:
                    print(f"Skip clean: zarr check failed for {zarr_path}: {e}")
                    continue

                source_paths = entry.get("source_paths", [])
                removed_count = 0
                for path in source_paths:
                    if not os.path.exists(path):
                        continue
                    removed_in_path = 0
                    for f in os.listdir(path):
                        if f.endswith(".hdf5"):
                            file_path = os.path.join(path, f)
                            try:
                                os.remove(file_path)
                                removed_count += 1
                                removed_in_path += 1
                            except Exception as e:
                                print(f"Failed to remove {file_path}: {e}")
                    if removed_in_path > 0:
                        print(f"Removed {removed_in_path} hdf5 files from {path}")

                if removed_count > 0:
                    entry["state"] = "cleaned"
                    entry["updated_at"] = datetime.now().isoformat()

    save_json(stat_path, stat)
    print(f"Clean done. Updated {stat_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert automoma camera_data to DP3 zarr format"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["collect", "convert", "check", "clean"],
        help="Operation mode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "output"),
        help="Root output directory containing automoma data",
    )
    parser.add_argument(
        "--stat_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "scripts", "config", "convert_statistic.json"),
        help="Path to convert_statistic.json",
    )
    parser.add_argument(
        "--zarr_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "baseline/RoboTwin/policy/DP3", "data"),
        help="Directory to save zarr files",
    )
    parser.add_argument(
        "--mobile_base_mode",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
        help="Mobile base action mode",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Episodes to process in each batch",
    )

    args = parser.parse_args()

    if args.mode == "collect":
        collect_mode(args.output_dir, args.stat_path)
    elif args.mode == "convert":
        convert_mode(
            args.stat_path,
            args.zarr_dir,
            mobile_base_mode=args.mobile_base_mode,
            batch_size=args.batch_size,
        )
    elif args.mode == "check":
        check_mode(args.stat_path, zarr_dir=args.zarr_dir)
    elif args.mode == "clean":
        clean_mode(args.stat_path)


if __name__ == "__main__":
    main()