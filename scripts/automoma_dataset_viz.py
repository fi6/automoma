#!/usr/bin/env python
"""Visualize AutoMoMa HDF5 datasets with an interactive Rerun viewer.

Examples:
    python scripts/automoma_dataset_viz.py \
        data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5

    python scripts/automoma_dataset_viz.py \
        data/automoma/summit_franka_open-microwave_7221-scene_0_seed_0-10.hdf5 \
        --demo-index 0 --save --output-dir outputs/viz
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import rerun as rr
import rerun.blueprint as rrb


def _demo_sort_key(name: str) -> tuple[int, str]:
    try:
        return (0, int(name.rsplit("_", 1)[-1]))
    except ValueError:
        return (1, name)


def _find_demo_root(root: h5py.File | h5py.Group) -> h5py.Group:
    if "data" in root and isinstance(root["data"], h5py.Group):
        data_group = root["data"]
        if any(key.startswith("demo_") for key in data_group.keys()):
            return data_group

    if any(key.startswith("demo_") for key in root.keys()):
        return root

    raise ValueError("Could not find demo groups. Expected `data/demo_*` or `demo_*`.")


def _sorted_demo_keys(demo_root: h5py.Group) -> list[str]:
    return sorted([key for key in demo_root.keys() if key.startswith("demo_")], key=_demo_sort_key)


def _demo_number(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def _parse_demo_indices(spec: str) -> list[int]:
    demo_indices: list[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid demo range `{token}`: end must be >= start.")
            demo_indices.extend(range(start, end + 1))
        else:
            demo_indices.append(int(token))
    return sorted(set(demo_indices))


def _normalize_rgb_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError(f"Expected RGB frame with 3 dims, got {arr.shape}")

    if arr.shape[-1] == 3:
        rgb = arr
    elif arr.shape[0] == 3:
        rgb = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"Expected RGB frame with 3 channels, got {arr.shape}")

    if rgb.dtype == np.uint8:
        return rgb

    if np.issubdtype(rgb.dtype, np.floating):
        scale = 255.0 if np.nanmax(rgb) <= 1.0 else 1.0
        return np.clip(rgb * scale, 0, 255).astype(np.uint8)

    return np.clip(rgb, 0, 255).astype(np.uint8)


def _normalize_depth_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim != 2:
        raise ValueError(f"Expected depth frame with 2 dims or singleton channel, got {arr.shape}")

    return arr.astype(np.float32, copy=False)


def _log_scalar_series(entity_path: str, values: np.ndarray) -> None:
    flat = np.asarray(values).reshape(-1)
    if flat.size == 1:
        rr.log(entity_path, rr.Scalars(float(flat[0])))
        return

    for dim_idx, value in enumerate(flat):
        rr.log(f"{entity_path}/{dim_idx}", rr.Scalars(float(value)))


def _log_pose_if_possible(entity_path: str, values: np.ndarray) -> None:
    flat = np.asarray(values).reshape(-1)
    if flat.size < 7:
        return

    position = flat[:3].astype(np.float32, copy=False)
    quaternion = flat[3:7].astype(np.float32, copy=False)
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=position,
            rotation=rr.Quaternion(xyzw=quaternion),
            axis_length=0.1,
        ),
    )


def _log_initial_state(demo_group: h5py.Group, root_path: str) -> None:
    if "initial_state" not in demo_group:
        return

    articulation = demo_group["initial_state"].get("articulation")
    if articulation is None:
        return

    for entity_name in articulation.keys():
        entity_group = articulation[entity_name]
        for dataset_name in entity_group.keys():
            values = np.asarray(entity_group[dataset_name][0])
            base_path = f"{root_path}/initial_state/articulation/{entity_name}/{dataset_name}"
            _log_scalar_series(base_path, values)
            if dataset_name.endswith("root_pose"):
                _log_pose_if_possible(f"{base_path}_tf", values)


def _camera_dataset_kind(name: str) -> str | None:
    if name.endswith("_rgb"):
        return "rgb"
    if name.endswith("_depth"):
        return "depth"
    return None


def _camera_base_name(name: str) -> str:
    if name.endswith("_rgb"):
        return name[: -len("_rgb")]
    if name.endswith("_depth"):
        return name[: -len("_depth")]
    return name


def _infer_num_frames(demo_group: h5py.Group) -> int:
    if "actions" in demo_group:
        return int(demo_group["actions"].shape[0])
    if "processed_actions" in demo_group:
        return int(demo_group["processed_actions"].shape[0])
    if "obs" in demo_group:
        for key in demo_group["obs"].keys():
            return int(demo_group["obs"][key].shape[0])
    raise ValueError("Could not infer number of frames from demo group.")


def _collect_summary(path: Path, demo_root: h5py.Group, demo_keys: list[str]) -> str:
    lines = [
        f"# AutoMoMa Dataset",
        "",
        f"- file: `{path}`",
        f"- demos: `{len(demo_keys)}`",
    ]

    if demo_keys:
        first_demo = demo_root[demo_keys[0]]
        num_frames = _infer_num_frames(first_demo)
        lines.append(f"- frames per first demo: `{num_frames}`")

        camera_obs = first_demo.get("camera_obs")
        if camera_obs is not None:
            rgb_cams = sorted(
                [_camera_base_name(key) for key in camera_obs.keys() if _camera_dataset_kind(key) == "rgb"]
            )
            depth_cams = sorted(
                [_camera_base_name(key) for key in camera_obs.keys() if _camera_dataset_kind(key) == "depth"]
            )
            lines.append(f"- rgb cameras: `{', '.join(rgb_cams) if rgb_cams else 'none'}`")
            lines.append(f"- depth cameras: `{', '.join(depth_cams) if depth_cams else 'none'}`")

        obs = first_demo.get("obs")
        if obs is not None:
            lines.append(f"- obs keys: `{', '.join(sorted(obs.keys()))}`")

        states = first_demo.get("states")
        if states is not None and "articulation" in states:
            lines.append(
                f"- articulation entities: `{', '.join(sorted(states['articulation'].keys()))}`"
            )

    return "\n".join(lines)


def _select_demo_keys(
    demo_root: h5py.Group,
    all_demo_keys: list[str],
    demo_indices_spec: str | None,
    demo_start: int | None,
    demo_count: int | None,
    auto_tab_limit: int,
) -> list[str]:
    if demo_indices_spec is not None and demo_start is not None:
        raise ValueError("Use either `--demo-indices` or `--demo-start/--demo-count`, not both.")

    if demo_count is not None and demo_count <= 0:
        raise ValueError("`--demo-count` must be > 0.")

    if demo_indices_spec is not None:
        demo_indices = _parse_demo_indices(demo_indices_spec)
        selected_keys = []
        missing = []
        for idx in demo_indices:
            key = f"demo_{idx}"
            if key in demo_root:
                selected_keys.append(key)
            else:
                missing.append(key)
        if missing:
            raise ValueError(f"Requested demos not found: {missing[:10]}")
        return selected_keys

    if demo_start is not None or demo_count is not None:
        start = demo_start or 0
        count = demo_count or 1
        selected = [f"demo_{idx}" for idx in range(start, start + count) if f"demo_{idx}" in demo_root]
        if not selected:
            raise ValueError(f"No demos found in requested window start={start}, count={count}.")
        return selected

    if len(all_demo_keys) > auto_tab_limit:
        logging.info(
            "Dataset has %d demos. Auto-limiting preview to `demo_0`. "
            "Use `--demo-index`, `--demo-indices`, or `--demo-start/--demo-count` for targeted loading.",
            len(all_demo_keys),
        )
        return [all_demo_keys[0]]

    return all_demo_keys


def _compute_depth_range(dataset: h5py.Dataset) -> tuple[float, float] | None:
    depth_values = _normalize_depth_frame(np.asarray(dataset[0]))
    if dataset.shape[0] > 1:
        stacked = np.asarray(dataset[:], dtype=np.float32)
        if stacked.ndim == 4 and stacked.shape[-1] == 1:
            stacked = stacked[..., 0]
        elif stacked.ndim == 4 and stacked.shape[1] == 1:
            stacked = stacked[:, 0, ...]
        depth_values = stacked

    valid = np.isfinite(depth_values) & (depth_values > 0)
    if not np.any(valid):
        return None

    valid_values = depth_values[valid]
    depth_min = float(np.percentile(valid_values, 2.0))
    depth_max = float(np.percentile(valid_values, 98.0))
    if not np.isfinite(depth_min) or not np.isfinite(depth_max) or depth_max <= depth_min:
        depth_min = float(valid_values.min())
        depth_max = float(valid_values.max())
        if depth_max <= depth_min:
            depth_max = depth_min + 1e-6
    return depth_min, depth_max


def _make_camera_view(entity_path: str, name: str) -> rrb.Spatial2DView:
    return rrb.Spatial2DView(origin="/", contents=[entity_path], name=name)


def _make_timeseries_view(name: str, contents: list[str]) -> rrb.TimeSeriesView:
    return rrb.TimeSeriesView(
        origin="/",
        contents=contents,
        name=name,
        plot_legend=rrb.PlotLegend(visible=True),
    )


def _build_demo_layout(demo_key: str, camera_obs: h5py.Group | None) -> rrb.Vertical:
    root = f"/{demo_key}"
    camera_views = []
    if camera_obs is not None:
        rgb_keys = sorted([key for key in camera_obs.keys() if key.endswith("_rgb")])
        depth_keys = sorted([key for key in camera_obs.keys() if key.endswith("_depth")])

        camera_views.extend([_make_camera_view(f"{root}/camera_obs/{key}", key) for key in rgb_keys])
        camera_views.extend([_make_camera_view(f"{root}/camera_obs/{key}", key) for key in depth_keys])

    scalar_views = [
        _make_timeseries_view(
            "action.base+processed",
            [f"{root}/actions/{i}" for i in range(3)] + [f"{root}/processed_actions/{i}" for i in range(3)],
        ),
        _make_timeseries_view(
            "action.arm+gripper",
            [f"{root}/actions/{i}" for i in range(3, 12)] + [f"{root}/processed_actions/{i}" for i in range(3, 12)],
        ),
        _make_timeseries_view("obs.joint_pos", [f"{root}/obs/joint_pos/**"]),
        _make_timeseries_view("obs.joint_vel", [f"{root}/obs/joint_vel/**"]),
        _make_timeseries_view(
            "obs.gripper+eef",
            [f"{root}/obs/gripper_pos/**", f"{root}/obs/eef_pos/**", f"{root}/obs/eef_quat/**"],
        ),
        _make_timeseries_view("states.articulation", [f"{root}/states/articulation/**"]),
    ]

    layout_parts = []
    row_shares = []
    if camera_views:
        layout_parts.append(rrb.Grid(contents=camera_views, grid_columns=3, name=f"{demo_key}_camera_grid"))
        row_shares.append(2.0)
    layout_parts.append(rrb.Grid(contents=scalar_views, grid_columns=3, name=f"{demo_key}_scalar_grid"))
    row_shares.append(1.6)
    return rrb.Vertical(contents=layout_parts, row_shares=row_shares, name=demo_key)


def _build_blueprint(
    demo_layouts: list[rrb.Vertical],
    fps: float,
) -> rrb.Blueprint:
    viewport: rrb.Container | rrb.View
    if len(demo_layouts) == 1:
        viewport = demo_layouts[0]
    else:
        viewport = rrb.Tabs(contents=demo_layouts, active_tab=0, name="demos")

    return rrb.Blueprint(
        viewport,
        rrb.SelectionPanel(state="expanded"),
        rrb.TimePanel(state="expanded", timeline="frame_index", fps=fps),
        rrb.BlueprintPanel(state="expanded"),
        rrb.TopPanel(state="expanded"),
        auto_views=False,
        auto_layout=False,
    )


def visualize_dataset(
    hdf5_path: Path,
    demo_index: str | None = None,
    demo_start: int | None = None,
    demo_count: int | None = None,
    auto_tab_limit: int = 32,
    fps: float = 30.0,
    mode: str = "local",
    web_port: int = 9090,
    save: bool = False,
    output_dir: Path | None = None,
    display_compressed_images: bool = False,
) -> Path | None:
    if save and output_dir is None:
        raise ValueError("Set `--output-dir` when using `--save`.")

    if mode not in {"local", "distant"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"automoma_dataset/{hdf5_path.stem}", spawn=spawn_local_viewer)

    if mode == "distant":
        rr.serve_web_viewer(open_browser=False, web_port=web_port)

    with h5py.File(hdf5_path, "r") as root:
        demo_root = _find_demo_root(root)
        demo_keys = _sorted_demo_keys(demo_root)

        if not demo_keys:
            raise ValueError("No `demo_*` groups found in dataset.")

        demo_keys = _select_demo_keys(
            demo_root=demo_root,
            all_demo_keys=demo_keys,
            demo_indices_spec=demo_index,
            demo_start=demo_start,
            demo_count=demo_count,
            auto_tab_limit=auto_tab_limit,
        )

        logging.info("Visualizing %d demo(s): %s", len(demo_keys), ", ".join(demo_keys[:10]))
        if len(demo_keys) > 10:
            logging.info("Demo list truncated in log; total selected demos: %d", len(demo_keys))

        demo_layouts = [_build_demo_layout(demo_key, demo_root[demo_key].get("camera_obs")) for demo_key in demo_keys]
        blueprint = _build_blueprint(demo_layouts, fps=fps)
        rr.send_blueprint(blueprint)

        summary = _collect_summary(hdf5_path, demo_root, demo_keys)
        logging.info("Dataset summary:\n%s", summary)
        rr.log("dataset/summary", rr.TextDocument(summary, media_type="markdown"), static=True)

        for demo_key in demo_keys:
            demo_group = demo_root[demo_key]
            num_frames = _infer_num_frames(demo_group)
            demo_number = _demo_number(demo_key)
            root_path = demo_key

            rr.set_time("frame_index", sequence=0)
            if fps > 0:
                rr.set_time("time", duration=0.0)
            rr.log(f"{root_path}/meta/demo_name", rr.TextDocument(demo_key), static=True)
            rr.log(f"{root_path}/meta/demo_index", rr.Scalars(float(demo_number)), static=True)
            _log_initial_state(demo_group, root_path=root_path)

            camera_obs = demo_group.get("camera_obs")
            obs_group = demo_group.get("obs")
            states_group = demo_group.get("states")
            depth_ranges = {}

            if camera_obs is not None:
                for dataset_name in camera_obs.keys():
                    if dataset_name.endswith("_depth"):
                        depth_range = _compute_depth_range(camera_obs[dataset_name])
                        if depth_range is not None:
                            depth_ranges[dataset_name] = depth_range

            for frame_idx in range(num_frames):
                rr.set_time("frame_index", sequence=frame_idx)
                if fps > 0:
                    rr.set_time("time", duration=frame_idx / fps)

                if "actions" in demo_group:
                    _log_scalar_series(f"{root_path}/actions", np.asarray(demo_group["actions"][frame_idx]))

                if "processed_actions" in demo_group:
                    _log_scalar_series(
                        f"{root_path}/processed_actions",
                        np.asarray(demo_group["processed_actions"][frame_idx]),
                    )

                if camera_obs is not None:
                    for dataset_name in camera_obs.keys():
                        dataset = camera_obs[dataset_name]
                        kind = _camera_dataset_kind(dataset_name)
                        if kind == "rgb":
                            frame = _normalize_rgb_frame(dataset[frame_idx])
                            image = rr.Image(frame).compress() if display_compressed_images else rr.Image(frame)
                            rr.log(f"{root_path}/camera_obs/{dataset_name}", image)
                        elif kind == "depth":
                            depth = _normalize_depth_frame(dataset[frame_idx])
                            rr.log(
                                f"{root_path}/camera_obs/{dataset_name}",
                                rr.DepthImage(
                                    depth,
                                    meter=1.0,
                                    colormap="Turbo",
                                    depth_range=depth_ranges.get(dataset_name),
                                ),
                            )

                if obs_group is not None:
                    for dataset_name in obs_group.keys():
                        values = np.asarray(obs_group[dataset_name][frame_idx])
                        base_path = f"{root_path}/obs/{dataset_name}"
                        _log_scalar_series(base_path, values)
                        if dataset_name == "eef_quat" and "eef_pos" in obs_group:
                            pose = np.concatenate(
                                [
                                    np.asarray(obs_group["eef_pos"][frame_idx]).reshape(-1),
                                    values.reshape(-1),
                                ]
                            )
                            _log_pose_if_possible(f"{root_path}/obs/eef_pose", pose)

                if states_group is not None and "articulation" in states_group:
                    articulation = states_group["articulation"]
                    for entity_name in articulation.keys():
                        entity_group = articulation[entity_name]
                        for dataset_name in entity_group.keys():
                            values = np.asarray(entity_group[dataset_name][frame_idx])
                            base_path = f"{root_path}/states/articulation/{entity_name}/{dataset_name}"
                            _log_scalar_series(base_path, values)
                            if dataset_name.endswith("root_pose"):
                                _log_pose_if_possible(f"{base_path}_tf", values)

    if mode == "local" and save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rrd_path = output_dir / f"{hdf5_path.stem}.rrd"
        rr.save(rrd_path)
        return rrd_path

    if mode == "distant":
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Ctrl-C received. Exiting.")

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize AutoMoMa HDF5 datasets with Rerun.")
    parser.add_argument("hdf5_path", type=Path, help="Path to an AutoMoMa HDF5 file.")
    parser.add_argument(
        "--demo-index",
        type=str,
        default=None,
        help="Visualize one or more demos, e.g. `123` or `0,5,10-15`.",
    )
    parser.add_argument(
        "--demo-start",
        type=int,
        default=None,
        help="Start demo index for windowed preview, used with `--demo-count`.",
    )
    parser.add_argument(
        "--demo-count",
        type=int,
        default=None,
        help="Number of demos to preview from `--demo-start`. If omitted with `--demo-start`, defaults to 1.",
    )
    parser.add_argument(
        "--auto-tab-limit",
        type=int,
        default=32,
        help="If total demos exceed this limit and no demo selector is given, only `demo_0` is loaded.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS used to derive the timestamp timeline. Set <= 0 to disable timestamp logging.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["local", "distant"],
        help="Viewer mode. `local` spawns a viewer, `distant` serves a web viewer.",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port used when `--mode distant` is selected.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save a `.rrd` file instead of spawning a local viewer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used together with `--save`.",
    )
    parser.add_argument(
        "--display-compressed-images",
        action="store_true",
        help="Log compressed RGB images to reduce viewer memory usage.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    visualize_dataset(
        hdf5_path=args.hdf5_path,
        demo_index=args.demo_index,
        demo_start=args.demo_start,
        demo_count=args.demo_count,
        auto_tab_limit=args.auto_tab_limit,
        fps=args.fps,
        mode=args.mode,
        web_port=args.web_port,
        save=args.save,
        output_dir=args.output_dir,
        display_compressed_images=args.display_compressed_images,
    )


if __name__ == "__main__":
    main()
