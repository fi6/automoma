#!/usr/bin/env python3
"""Build attached-object AKR URDF and cuRobo configurations."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from yourdfpy import URDF

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from automoma.assets.akr import (  # noqa: E402
    AKR_7221_SELF_COLLISION_IGNORE,
    build_akr_urdf,
    build_robot_config,
    dump_yaml,
    extract_sphere_map,
    generate_collision_spheres,
    load_grasp_pose,
    load_object_init,
    load_yaml,
    parse_grasp_ids,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-urdf", type=Path, required=True)
    parser.add_argument("--object-init-state", type=Path, required=True)
    parser.add_argument("--grasp-dir", type=Path, required=True)
    parser.add_argument("--grasp-ids", default="0-20")
    parser.add_argument("--object-base", default="link_1")
    parser.add_argument("--handle-link", default="link_0")
    parser.add_argument("--object-joint", default="joint_0")
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--robot-yaml", type=Path, required=True)
    parser.add_argument("--robot-fixed-yaml", type=Path, required=True)
    spheres = parser.add_mutually_exclusive_group(required=True)
    spheres.add_argument("--sphere-yaml", type=Path)
    spheres.add_argument("--generate-spheres", action="store_true")
    parser.add_argument("--asset-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_inputs(args: argparse.Namespace) -> None:
    paths = [
        args.object_urdf,
        args.object_init_state,
        args.grasp_dir,
        args.robot_urdf,
        args.robot_yaml,
        args.robot_fixed_yaml,
    ]
    if args.sphere_yaml:
        paths.append(args.sphere_yaml)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing inputs: " + ", ".join(map(str, missing)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.output_dir} is not empty; pass --force to replace generated files")


def _fixed_locks(fixed_yaml: dict, mobile_yaml: dict) -> dict[str, float]:
    fixed = fixed_yaml["robot_cfg"]["kinematics"]["lock_joints"]
    mobile = mobile_yaml["robot_cfg"]["kinematics"]["lock_joints"]
    return {key: value for key, value in fixed.items() if key not in mobile}


def _grasp_profile(path: Path | None, grasp_id: int) -> dict | None:
    if path is None:
        return None
    match = re.search(r"(grasp_)(\d{4})(\.ya?ml)$", path.name)
    if match:
        sibling = path.with_name(path.name[: match.start(2)] + f"{grasp_id:04d}" + path.name[match.end(2) :])
        if sibling.is_file():
            return load_yaml(sibling)
    return load_yaml(path)


def build(args: argparse.Namespace) -> list[Path]:
    _validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scale, qpos = load_object_init(args.object_init_state)
    grasp_ids = parse_grasp_ids(args.grasp_ids)
    robot_yaml = load_yaml(args.robot_yaml)
    fixed_yaml = load_yaml(args.robot_fixed_yaml)
    profile = _grasp_profile(args.sphere_yaml, grasp_ids[0])
    if profile:
        all_spheres = extract_sphere_map(profile)
        sphere_map = {
            args.handle_link: all_spheres[args.handle_link],
            args.object_base: all_spheres[args.object_base],
        }
    else:
        sphere_map = generate_collision_spheres(
            args.object_urdf,
            handle_link=args.handle_link,
            body_link=args.object_base,
        )
    standalone_spheres = args.output_dir / "collision_spheres.yml"
    dump_yaml({"collision_spheres": sphere_map}, standalone_spheres)

    robot_name = args.robot_urdf.stem
    object_id = Path(args.asset_prefix.rstrip("/")).name
    instance_id = args.object_init_state.parent.name
    base_name = f"{robot_name}_{object_id}_{instance_id}_grasp"
    fixed_name = f"{robot_name}_fixed_base_{object_id}_{instance_id}_grasp"
    try:
        asset_root = str(args.robot_urdf.absolute().parent.relative_to(REPO_ROOT.absolute()))
    except ValueError:
        asset_root = str(args.robot_urdf.parent)
    object_model = URDF.load(str(args.object_urdf))
    outputs = [standalone_spheres]
    fixed_locks = _fixed_locks(fixed_yaml, robot_yaml)

    for grasp_id in grasp_ids:
        grasp_profile = _grasp_profile(args.sphere_yaml, grasp_id)
        grasp_path = args.grasp_dir / f"{grasp_id:04d}.npy"
        if not grasp_path.is_file():
            raise FileNotFoundError(f"missing grasp: {grasp_path}")
        grasp_pose = load_grasp_pose(grasp_path, scale)
        stem = f"{base_name}_{grasp_id:04d}"
        urdf_name = f"{stem}.urdf"
        logical_urdf_path = f"{args.asset_prefix.rstrip('/')}/{urdf_name}"
        akr = build_akr_urdf(
            args.object_urdf,
            args.robot_urdf,
            grasp_pose,
            object_base=args.object_base,
            handle_link=args.handle_link,
            object_joint_positions=qpos,
            asset_prefix=args.asset_prefix,
            golden_compatibility=object_id == "7221",
        )
        urdf_output = args.output_dir / urdf_name
        akr.write_xml_file(str(urdf_output))
        mobile = build_robot_config(
            robot_yaml,
            urdf_path=logical_urdf_path,
            object_urdf=object_model,
            object_base=args.object_base,
            object_joint=args.object_joint,
            collision_spheres=sphere_map,
            asset_root_path=asset_root,
            profile=grasp_profile,
        )
        if grasp_profile is None and object_id == "7221":
            mobile["robot_cfg"]["kinematics"]["self_collision_ignore"] = AKR_7221_SELF_COLLISION_IGNORE
        mobile_output = args.output_dir / f"{stem}.yml"
        dump_yaml(mobile, mobile_output)
        fixed = build_robot_config(
            robot_yaml,
            urdf_path=logical_urdf_path,
            object_urdf=object_model,
            object_base=args.object_base,
            object_joint=args.object_joint,
            collision_spheres=sphere_map,
            asset_root_path=asset_root,
            profile=grasp_profile,
            fixed_lock_joints=fixed_locks,
        )
        if grasp_profile is None and object_id == "7221":
            fixed["robot_cfg"]["kinematics"]["self_collision_ignore"] = AKR_7221_SELF_COLLISION_IGNORE
        fixed_output = args.output_dir / f"{fixed_name}_{grasp_id:04d}.yml"
        dump_yaml(fixed, fixed_output)
        outputs.extend([urdf_output, mobile_output, fixed_output])
    return outputs


def main(argv: list[str] | None = None) -> int:
    try:
        outputs = build(parse_args(argv))
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"generated {len(outputs) - 1} AKR assets and {outputs[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
