#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
import trimesh.transformations as tra
from yourdfpy import URDF
from scene_synthesizer.procedural_assets import URDFAsset

from automoma.utils.transform import pose_to_matrix
from automoma.utils.urdf.construct import attach_object_to_robot
from automoma.utils.urdf.collision import generate_urdf_collision_spheres
from automoma.utils.urdf.object import BaseObject


DEFAULT_GRASP_POSE = np.array(
    [
        [6.123234e-17, 0.000000e00, 1.000000e00, -3.150000e-01],
        [0.000000e00, 1.000000e00, 0.000000e00, -0.4],
        [-1.000000e00, 0.000000e00, 6.123234e-17, 9.500000e-01],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _parse_object_cfg(items: Optional[List[str]]) -> Dict[str, float]:
    cfg: Dict[str, float] = {}
    if not items:
        return cfg
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid object cfg entry '{item}'. Use joint=value.")
        name, value = item.split("=", 1)
        cfg[name.strip()] = float(value)
    return cfg


def _parse_grasp_pose(args: argparse.Namespace) -> np.ndarray:
    if args.grasp_pose_file:
        grasp_raw = np.load(args.grasp_pose_file)
        if grasp_raw.shape == (4, 4):
            grasp_pose = grasp_raw
        elif grasp_raw.shape == (7,):
            grasp_pose = pose_to_matrix(grasp_raw)
        else:
            raise ValueError("grasp_pose_file must be shape (4,4) or (7,)")
    elif args.grasp_pose:
        values = [float(v) for v in args.grasp_pose]
        if len(values) == 16:
            grasp_pose = np.array(values, dtype=np.float64).reshape(4, 4)
        elif len(values) == 7:
            grasp_pose = pose_to_matrix(np.array(values, dtype=np.float64))
        else:
            raise ValueError("grasp_pose must have 16 (matrix) or 7 (pose) values")
    else:
        grasp_pose = DEFAULT_GRASP_POSE.copy()

    if args.grasp_adjust_rpy is not None:
        roll, pitch, yaw = args.grasp_adjust_rpy
        adjust = tra.euler_matrix(roll, pitch, yaw, "rxyz")
        grasp_pose = grasp_pose @ adjust

    if args.grasp_adjust_xyz is not None:
        dx, dy, dz = args.grasp_adjust_xyz
        grasp_pose = grasp_pose.copy()
        grasp_pose[:3, 3] += np.array([dx, dy, dz], dtype=np.float64)

    return grasp_pose


def _get_object_base_link(urdf: URDF) -> Optional[str]:
    for joint in urdf.robot.joints:
        if joint.parent == "world":
            return joint.child
    return None


def _load_collision_spheres(path: Path) -> Dict[str, List[dict]]:
    data = yaml.safe_load(path.read_text())
    spheres = data.get("collision_spheres", {})
    if isinstance(spheres, list):
        merged: Dict[str, List[dict]] = {}
        for item in spheres:
            if isinstance(item, dict):
                merged.update(item)
        spheres = merged
    if not isinstance(spheres, dict):
        raise ValueError("collision_spheres must be a dict or list of dicts")
    return spheres


def _normalize_mesh_paths(urdf: URDF, urdf_path: Path) -> URDF:
    """Ensure mesh filenames are resolvable by prefixing the URDF directory when needed."""
    base_dir = urdf_path.parent

    def _fix_mesh(mesh):
        if mesh is None or not getattr(mesh, "filename", None):
            return
        filename = mesh.filename
        if filename.startswith("/"):
            return
        if "://" in filename:
            return
        if "/" in filename:
            return
        mesh.filename = str((base_dir / filename).as_posix())

    for link in urdf.robot.links:
        for visual in getattr(link, "visuals", []) or []:
            geo = getattr(visual, "geometry", None)
            if geo is not None and getattr(geo, "mesh", None) is not None:
                _fix_mesh(geo.mesh)
        for collision in getattr(link, "collisions", []) or []:
            geo = getattr(collision, "geometry", None)
            if geo is not None and getattr(geo, "mesh", None) is not None:
                _fix_mesh(geo.mesh)

    return urdf


def _merge_object_into_robot_yaml(
    robot_yaml: dict,
    object_urdf_path: Path,
    object_links: List[str],
    object_joints: List[str],
    object_cfg: Dict[str, float],
    object_collision_spheres: Dict[str, List[dict]],
    output_urdf_path: str,
    ee_link: str,
    robot_tcp_links: List[str],
):
    kin = robot_yaml["robot_cfg"]["kinematics"]

    # Update URDF path and ee link
    kin["urdf_path"] = output_urdf_path
    kin["ee_link"] = ee_link

    # Collision links and mesh links
    collision_links = kin.get("collision_link_names", [])
    mesh_links = kin.get("mesh_link_names", [])
    for link in object_links:
        if link not in collision_links:
            collision_links.append(link)
        if link not in mesh_links:
            mesh_links.append(link)
    kin["collision_link_names"] = collision_links
    kin["mesh_link_names"] = mesh_links

    # Collision spheres
    if "collision_spheres" not in kin:
        kin["collision_spheres"] = {}
    for link, spheres in object_collision_spheres.items():
        if link in object_links:
            kin["collision_spheres"][link] = spheres

    # Self collision buffer
    scb = kin.get("self_collision_buffer", {})
    for link in object_links:
        scb.setdefault(link, 0.0)
    kin["self_collision_buffer"] = scb

    # Self collision ignore - add object links vs robot TCP links
    sci = kin.get("self_collision_ignore", {})
    for obj_link in object_links:
        sci.setdefault(obj_link, [])
        for tcp in robot_tcp_links:
            if tcp not in sci[obj_link]:
                sci[obj_link].append(tcp)
    kin["self_collision_ignore"] = sci

    # C-space joints
    cspace = kin["cspace"]
    joint_names = cspace.get("joint_names", [])
    retract = cspace.get("retract_config", [])
    null_w = cspace.get("null_space_weight", [])
    dist_w = cspace.get("cspace_distance_weight", [])

    for joint_name in object_joints:
        if joint_name not in joint_names:
            joint_names.append(joint_name)
            retract.append(float(object_cfg.get(joint_name, 0.0)))
            null_w.append(1.0)
            dist_w.append(1.0)

    cspace["joint_names"] = joint_names
    cspace["retract_config"] = retract
    cspace["null_space_weight"] = null_w
    cspace["cspace_distance_weight"] = dist_w


def _attach_object_base_to_robot(
    object_urdf: URDF,
    robot_urdf: URDF,
    robot_ee_link: str,
    grasp_pose: np.ndarray,
    object_base_link: str,
) -> URDF:
    """Attach object base to robot EE using grasp pose (object frame -> EE pose)."""
    # Find the joint that attaches the object base to world
    world_joint_name = None
    for joint in object_urdf.robot.joints:
        if joint.parent == "world" and joint.child == object_base_link:
            world_joint_name = joint.name
            break
    if world_joint_name is None:
        raise ValueError(
            f"No world joint found for base link '{object_base_link}'."
        )

    attached_origin = np.linalg.inv(grasp_pose)

    vkc_robot = BaseObject("scene")
    obj_base = BaseObject.from_robot(object_urdf.robot)
    vkc_robot._merge_robot(obj_base, robot_urdf.robot)

    joint = vkc_robot.joint_map[world_joint_name]
    joint.parent = robot_ee_link
    joint.origin = attached_origin

    return URDF(vkc_robot)


def main():
    parser = argparse.ArgumentParser(
        description="Generate AKR URDF/YAML by attaching a Fetch robot to an object URDF."
    )
    parser.add_argument(
        "--robot-urdf",
        default="assets/robot/fetch/fetch.urdf",
        help="Path to Fetch URDF",
    )
    parser.add_argument(
        "--robot-yaml",
        default="assets/robot/fetch/fetch.yml",
        help="Path to Fetch YAML config",
    )
    parser.add_argument(
        "--object-urdf",
        default="assets/object/Refrigerator/10000/10000_0_scaling.urdf",
        help="Path to object URDF",
    )
    parser.add_argument(
        "--object-collision-yml",
        default="assets/object/Refrigerator/10000/fridge_dynamic_sphere.yml",
        help="Optional object collision spheres YAML",
    )
    parser.add_argument(
        "--object-base-link",
        default=None,
        help="Object base link (defaults to link attached to world)",
    )
    parser.add_argument(
        "--object-tip-link",
        default="top_door",
        help="Object tip/handle link to attach",
    )
    parser.add_argument(
        "--object-ee-link",
        default=None,
        help="Object root link for inversion (defaults to object base link)",
    )
    parser.add_argument(
        "--ee-link",
        default=None,
        help="EE link in output YAML (defaults to object base link)",
    )
    parser.add_argument(
        "--robot-ee-link",
        default=None,
        help="Robot EE link to attach to (defaults to ee_link in robot yaml)",
    )
    parser.add_argument(
        "--output-urdf",
        default="assets/object/Refrigerator/10000/fetch_10000_0_grasp_0000.urdf",
        help="Output combined URDF path",
    )
    parser.add_argument(
        "--output-yml",
        default="assets/object/Refrigerator/10000/fetch_10000_0_grasp_0000.yml",
        help="Output combined YAML path",
    )
    parser.add_argument(
        "--grasp-pose",
        nargs="+",
        help="Grasp pose as 16 matrix values or 7 pose values",
    )
    parser.add_argument(
        "--grasp-pose-file",
        help="Path to .npy file with 4x4 matrix or 7D pose",
    )
    parser.add_argument(
        "--grasp-adjust-rpy",
        nargs=3,
        type=float,
        help="Additional roll pitch yaw (rad) adjustment",
    )
    parser.add_argument(
        "--grasp-adjust-xyz",
        nargs=3,
        type=float,
        help="Additional xyz translation adjustment",
    )
    parser.add_argument(
        "--object-cfg",
        nargs="*",
        help="Object joint config in joint=value format",
    )
    parser.add_argument(
        "--robot-tcp-links",
        nargs="+",
        default=["gripper_link", "l_gripper_finger_link", "r_gripper_finger_link"],
        help="Robot TCP links for collision ignore",
    )
    parser.add_argument(
        "--attach-via-base",
        action="store_true",
        help="Attach object base (body) to robot EE using grasp pose",
    )
    parser.add_argument(
        "--attach-via-tip",
        action="store_true",
        help="Attach object tip (top_door) to robot EE (inverse-root method)",
    )

    args = parser.parse_args()

    robot_urdf_path = Path(args.robot_urdf)
    robot_yaml_path = Path(args.robot_yaml)
    object_urdf_path = Path(args.object_urdf)
    output_urdf_path = Path(args.output_urdf)
    output_yml_path = Path(args.output_yml)

    output_urdf_path.parent.mkdir(parents=True, exist_ok=True)
    output_yml_path.parent.mkdir(parents=True, exist_ok=True)

    grasp_pose = _parse_grasp_pose(args)
    object_cfg = _parse_object_cfg(args.object_cfg)

    robot_yaml = yaml.safe_load(robot_yaml_path.read_text())
    kin = robot_yaml["robot_cfg"]["kinematics"]
    robot_ee_link = args.robot_ee_link or kin.get("ee_link", "ee_link")

    object_urdf = URDF.load(str(object_urdf_path))
    object_urdf = _normalize_mesh_paths(object_urdf, object_urdf_path)
    object_links = [link.name for link in object_urdf.robot.links if link.name != "world"]
    object_joints = [j.name for j in object_urdf.robot.joints if j.type != "fixed"]

    object_base_link = args.object_base_link or _get_object_base_link(object_urdf)
    if object_base_link is None:
        raise ValueError("Could not infer object base link. Please set --object-base-link.")

    object_ee_link = args.object_ee_link or object_base_link
    output_ee_link = args.ee_link or object_base_link

    object_collision_spheres: Dict[str, List[dict]] = {}
    if args.object_collision_yml and Path(args.object_collision_yml).exists():
        object_collision_spheres = _load_collision_spheres(Path(args.object_collision_yml))
    else:
        object_collision_spheres = generate_urdf_collision_spheres(
            str(object_urdf_path), [args.object_tip_link]
        )["collision_spheres"]

    # Build combined URDF
    robot_urdf = URDF.load(str(robot_urdf_path))

    if args.attach_via_tip and args.attach_via_base:
        raise ValueError("Choose only one of --attach-via-base or --attach-via-tip.")

    attach_via_base = False
    if args.attach_via_base:
        attach_via_base = True

    if not attach_via_base and args.ee_link is None:
        output_ee_link = args.object_tip_link

    if attach_via_base:
        combined_urdf = _attach_object_base_to_robot(
            object_urdf=object_urdf,
            robot_urdf=robot_urdf,
            robot_ee_link=robot_ee_link,
            grasp_pose=grasp_pose,
            object_base_link=object_base_link,
        )
    else:
        # Write normalized object URDF to a temp file for URDFAsset ingestion
        temp_object_urdf = output_urdf_path.parent / f"{object_urdf_path.stem}_normalized.urdf"
        object_urdf.write_xml_file(str(temp_object_urdf))
        object_asset = URDFAsset(str(temp_object_urdf))
        combined_urdf = attach_object_to_robot(
            object_asset,
            robot_urdf,
            grasp_pose,
            args.object_tip_link,
            object_ee_link,
            robot_ee_link,
            object_cfg=object_cfg if object_cfg else None,
        )
    combined_urdf.write_xml_file(str(output_urdf_path))

    # Build combined YAML
    _merge_object_into_robot_yaml(
        robot_yaml=robot_yaml,
        object_urdf_path=object_urdf_path,
        object_links=object_links,
        object_joints=object_joints,
        object_cfg=object_cfg,
        object_collision_spheres=object_collision_spheres,
        output_urdf_path=str(output_urdf_path.as_posix()),
        ee_link=output_ee_link,
        robot_tcp_links=args.robot_tcp_links,
    )

    output_yml_path.write_text(yaml.safe_dump(robot_yaml, sort_keys=False))


if __name__ == "__main__":
    main()
