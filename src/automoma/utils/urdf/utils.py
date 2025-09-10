from __future__ import annotations
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation as R
import sys
from yourdfpy.urdf import URDF, Collision, Geometry, Visual
import numpy as np
import yaml
from automoma.utils.urdf.object import BaseObject
import itertools
from collections import defaultdict
import torch
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.wrap.model.robot_world import RobotWorld
import time


def sum_link_mesh(geo_list):
    link_mesh = trimesh.Trimesh()
    print(f"geo_list, {geo_list}")
    for i in geo_list:
        print(i.geometry.mesh.filename)
        m: trimesh.Trimesh = trimesh.load(i.geometry.mesh.filename)
        print(f"Mesh: {m.vertices.shape}")
        link_mesh = trimesh.util.concatenate(m, link_mesh)
    return link_mesh


def quaternion2rot(quaternion):
    r = R.from_quat(quaternion)
    rot = r.as_matrix()
    return rot


def scale_geometry(mesh, scaling_factor, file_name):
    mesh.apply_scale(scaling_factor)
    mesh.export(file_name)
    m = yourdfpy.Mesh(filename=file_name)
    geo = Geometry(mesh=m)
    return geo


def init_scale_urdf(
    urdf, scaling_factor, extracted_file: str | None = None, mesh_folder=""
):
    # if not mesh_folder:
    #     extracted_folder = extracted_file.rsplit("/", 1)[0]
    #     mesh_folder = f"{extracted_folder}/new_mesh/"
    
    print(f"Scaling URDF by factor {scaling_factor}")

    for link in urdf.robot.links:
        print(f"Processing link: {link.name}")
        # Scale link visuals
        if len(link.visuals) != 0:
            print(f"  Scale")
            link_mesh_sum = sum_link_mesh(link.visuals)
            print(f"  Merged visuals into one mesh")
            link_visual_file = f"{mesh_folder}{link.name}_scale.obj"
            link_geo = scale_geometry(link_mesh_sum, scaling_factor, link_visual_file)
            link_vis = Visual(
                name=link.name,
                origin=link.visuals[0].origin * scaling_factor,
                geometry=link_geo,
            )
            print(f" scale_geometry")
            

            link.visuals.clear()
            link.visuals.append(link_vis)
            print(f"  Scaled visuals and saved to {link_visual_file}")

        # Scale link collisions
        if len(link.collisions) != 0:
            print(f"  Scale")
            link_col_sum = sum_link_mesh(link.collisions)
            print(f"  Merged collisions into one mesh")
            link_col_file = f"{mesh_folder}{link.name}_col_scale.obj"
            link_geo = scale_geometry(link_col_sum, scaling_factor, link_col_file)
            link_vis = Collision(
                name=link.name,
                origin=link.collisions[0].origin * scaling_factor,
                geometry=link_geo,
            )
            print(f" scale_geometry")

            link.collisions.clear()
            link.collisions.append(link_vis)
            print(f"  Scaled collisions and saved to {link_col_file}")
            
    print("Finished scaling all link geometries.")

    # Re-calculate joint origin after scaling link geometry
    for joint in urdf.robot.joints:
        t_parent_child = urdf.get_transform(joint.child, joint.parent)
        joint.origin[:3, :3] = t_parent_child[:3, :3]
        joint.origin[:3, 3] = t_parent_child[:3, 3] * scaling_factor
        
    print("Re-calculated joint origins after scaling.")

    if extracted_file is not None:
        urdf.write_xml_file(extracted_file)
    
    return urdf

# TODO: Clean up
def init_scale_urdf_old(
    urdf_file, scaling_factor, init_cfg, extracted_file: str | None = None, mesh_folder=""
):
    urdf = URDF(robot=URDF.load(urdf_file + "mobility.urdf").robot)
    if not mesh_folder:
        mesh_folder = f"{urdf_file}new_mesh/"
    # Apply init joint state
    for joint_id, joint in enumerate(urdf.robot.joints):
        urdf.update_cfg({joint.name: init_cfg[joint_id]})

    for link in urdf.robot.links:
        # Scale link visuals
        if len(link.visuals) != 0:
            link_mesh_sum = sum_link_mesh(link.visuals)
            link_visual_file = f"{mesh_folder}{link.name}_scale.obj"
            link_geo = scale_geometry(link_mesh_sum, scaling_factor, link_visual_file)
            link_vis = Visual(
                name=link.name,
                origin=link.visuals[0].origin * scaling_factor,
                geometry=link_geo,
            )

            link.visuals.clear()
            link.visuals.append(link_vis)

        # Scale link collisions
        if len(link.collisions) != 0:
            link_col_sum = sum_link_mesh(link.collisions)
            link_col_file = f"{mesh_folder}{link.name}_col_scale.obj"
            link_geo = scale_geometry(link_col_sum, scaling_factor, link_col_file)
            link_vis = Collision(
                name=link.name,
                origin=link.collisions[0].origin * scaling_factor,
                geometry=link_geo,
            )

            link.collisions.clear()
            link.collisions.append(link_vis)

    # Re-calculate joint origin after scaling link geometry
    for joint in urdf.robot.joints:
        t_parent_child = urdf.get_transform(joint.child, joint.parent)
        joint.origin[:3, :3] = t_parent_child[:3, :3]
        joint.origin[:3, 3] = t_parent_child[:3, 3] * scaling_factor

    if extracted_file is not None:
        urdf.write_xml_file(extracted_file)
    
    return urdf


def collision_checking(robot: CudaRobotModel, q, link_pairs=None):
    if link_pairs is None:
        links = robot.kinematics_config.link_name_to_idx_map.keys()
        link_pairs = itertools.combinations(links, 2)
    link_pairs = list(link_pairs)  # fix link pairs iterate end error
    robot_state = robot.get_state(q)
    sphere_states = robot_state.get_link_spheres()
    sphere_idx_pairs = []
    valid_link_pairs = []
    for link1, link2 in link_pairs:
        idx1 = robot.kinematics_config.get_sphere_index_from_link_name(link1)
        idx2 = robot.kinematics_config.get_sphere_index_from_link_name(link2)
        if len(idx1) == 0 or len(idx2) == 0:
            continue
        valid_link_pairs.append((link1, link2))
        sphere_idx_pairs.append((idx1, idx2))
    collision_count = defaultdict.fromkeys(valid_link_pairs, 0)
    for i, sphere_state in enumerate(sphere_states):
        print(f"\033[92m    Checking collision for sample {i}\033[90m", end="\r")
        for (link1, link2), (idx1, idx2) in zip(valid_link_pairs, sphere_idx_pairs):
            spheres1 = sphere_state[idx1]
            spheres2 = sphere_state[idx2]

            # Create tensors from sphere centers and radii
            centers1 = spheres1[:, :3]  # Extract xyz
            radii1 = spheres1[:, 3]  # Extract radii

            centers2 = spheres2[:, :3]
            radii2 = spheres2[:, 3]

            # Broadcasting to calculate all combinations of distances between centers
            dist_matrix = torch.cdist(
                centers1, centers2, p=2
            )  # Euclidean distance matrix

            # Broadcasting radii sum for all combinations
            radii_sum = radii1[:, None] + radii2  # Shape [num_spheres1, num_spheres2]

            # Find collisions
            collision_matrix = dist_matrix < radii_sum

            # Check if there is any collision
            if collision_matrix.any():
                collision_count[(link1, link2)] += 1
    return collision_count


def extract_col_link_set(yml_file):

    with open(yml_file, "r") as read_file:
        yml_data = yaml.safe_load(read_file)

    link_list = yml_data["robot_cfg"]["kinematics"]["collision_link_names"]
    return set(link_list)


def cal_self_collision_ignore(yml_data, samples_n, robot_col_links):
    start_time = time.time()
    robot_world = RobotWorld(RobotWorld.load_from_config(yml_data["robot_cfg"]))
    robot = robot_world.kinematics
    col_links = yml_data["robot_cfg"]["kinematics"]["collision_link_names"]
    link_pairs = itertools.combinations(col_links, 2)

    filtered_pairs = (
        pair
        for pair in link_pairs
        if not (pair[0] in robot_col_links and pair[1] in robot_col_links)
    )
    q_sample = robot_world.sample(samples_n, mask_valid=False)
    collision_count = collision_checking(robot, q_sample, filtered_pairs)

    original_link_pairs = len(col_links) * (len(col_links) - 1) / 2
    filtered_link_pairs = len(collision_count.items())
    print(
        f"\033[92m    {int(original_link_pairs)} original link pairs, filtered as {filtered_link_pairs}\033[90m"
    )

    self_collision_ignore = defaultdict(
        list, yml_data["robot_cfg"]["kinematics"]["self_collision_ignore"]
    )
    for key, value in collision_count.items():
        if value >= samples_n * 0.8 or value == 0:
            self_collision_ignore[key[0]].append(key[1])
        if value > samples_n * 0.2:
            print(
                f"Collision>{int(samples_n * 0.2)} between {key[0]} and {key[1]}: {value}"
            )

    new_self_collision_ignore = {"self_collision_ignore": {}}
    new_self_collision_ignore["self_collision_ignore"].update(self_collision_ignore)
    yml_data["robot_cfg"]["kinematics"].update(new_self_collision_ignore)

    run_time = time.time() - start_time
    print(
        f"\033[92m   Collision Checking with {samples_n} samples finished in {(run_time/60): .1f}min\033[0m"
    )


def cal_base_tip(urdf_file, preferenced_tip_link=""):
    obj = BaseObject.load(urdf_file)
    """ base_link: link attached to 'world' """
    base_link, tip_link, tip_joint = "", "", ""
    tip_num = 0
    tips = list()
    for j in obj.joints:

        if j.parent == "world":
            base_link = j.child

        if preferenced_tip_link and j.child == preferenced_tip_link:
            tip_link = preferenced_tip_link
            tip_joint = j.name

        if not preferenced_tip_link and j.child not in [k.parent for k in obj.joints]:
            tip_link = j.child
            tip_joint = j.name
            tips.append(j.child)
            tip_num += 1

        if preferenced_tip_link and base_link and tip_joint:
            break

    if tip_num > 1:
        sys.exit(
            f"Error: Multiple tip_links detected ({tips}). Please specify one as prior_tip_link."
        )

    return base_link, tip_link, tip_joint


def cal_base_tip_from_urdf(urdf_obj: URDF, preferenced_tip_link=""):
    """
    Calculate base and tip links from a URDF object directly.

    Args:
        urdf_obj: URDF object
        preferenced_tip_link: Optional preferred tip link name

    Returns:
        tuple: (base_link, tip_link, tip_joint)
    """
    base_link, tip_link, tip_joint = "", "", ""
    tip_num = 0
    tips = list()

    for j in urdf_obj.robot.joints:
        if j.parent == "world":
            base_link = j.child

        if preferenced_tip_link and j.child == preferenced_tip_link:
            tip_link = preferenced_tip_link
            tip_joint = j.name

        if not preferenced_tip_link and j.child not in [
            k.parent for k in urdf_obj.robot.joints
        ]:
            tip_link = j.child
            tip_joint = j.name
            tips.append(j.child)
            tip_num += 1

        if preferenced_tip_link and base_link and tip_joint:
            break

    if tip_num > 1:
        sys.exit(
            f"Error: Multiple tip_links detected ({tips}). Please specify one as prior_tip_link."
        )

    return base_link, tip_link, tip_joint


def inverse_urdf(urdf_file, tip_link, extracted_file, root_link="world"):
    obj = BaseObject.load(urdf_file)
    obj.inverse_root_tip(root_link, tip_link)
    inv_urdf = URDF(robot=obj)
    inv_urdf.write_xml_file(extracted_file)


def inverse_urdf_from_object(urdf_obj: URDF, tip_link: str, root_link="world") -> URDF:
    """
    Create an inverse URDF from a URDF object directly.

    Args:
        urdf_obj: Input URDF object
        tip_link: Link to become the new root
        root_link: Current root link (default: "world")

    Returns:
        URDF: Inversed URDF object
    """
    # Convert URDF to BaseObject
    obj = BaseObject.from_robot(urdf_obj.robot)
    obj.inverse_root_tip(root_link, tip_link)
    inv_urdf = URDF(robot=obj)
    return inv_urdf


def attach2robot_from_urdf(
    robot_urdf: URDF,
    inversed_obj_urdf: URDF,
    attached_link: str,
    attached_trans: np.matrix,
) -> URDF:
    """
    Attach the object URDF to the robot using URDF objects directly.

    Args:
        robot_urdf_file: Path to robot URDF file
        inversed_obj_urdf: Inversed object URDF object
        attached_link: Link of robot which the object is attached to, usually ee_link
        attached_trans: Object handle to robot ee_link transform

    Returns:
        URDF: Spliced URDF object
    """
    vkc_robot = BaseObject("scene")
    obj_inv = BaseObject(inversed_obj_urdf.robot)
    robot = robot_urdf.robot
    vkc_robot._merge_robot(obj_inv, robot)

    world_joint_name = ""
    for j in obj_inv.joints:
        if j.parent == "world":
            world_joint_name = j.name

    joint = vkc_robot.joint_map[world_joint_name]
    joint.parent = attached_link
    joint.origin = attached_trans

    vkc_robot_urdf = URDF(vkc_robot)
    return vkc_robot_urdf


def cal_attached_origin(
    obj_file, tip_link, tip_joint, grasp_pose, init_data_dict=None
) -> np.matrix:
    """
    Calculate the trans from object tip_link to robot tip link
    Args:
        obj_file: regular urdf (not inversed)
        tip_link:
        tip_joint: joint between object's tip_link and its parent
        grasp_pose: relative to the [object] base (usually obtained from grasp dataset)

    Returns: attached_origin
    """

    t_handle = np.eye(4, dtype=np.float64)
    obj_r = URDF.load(obj_file)
    tip_joint_angle = 0.0

    if init_data_dict:
        for joint_id, joint in enumerate(obj_r.robot.joints):
            if joint.name == tip_joint:
                tip_joint_angle = init_data_dict["object"]["qpos"][joint_id]

    # Update joint configuration and get transform
    obj_r.update_cfg({tip_joint: tip_joint_angle})
    t_handle = obj_r.get_transform(tip_link)

    t_grasp = np.matrix(grasp_pose)
    return t_grasp.I * t_handle


def cal_attached_origin_from_urdf(
    urdf_obj: URDF,
    tip_link: str,
    tip_joint: str,
    grasp_pose: np.ndarray,
    init_data_dict=None,
) -> np.matrix:
    """
    Calculate the trans from object tip_link to robot tip link using URDF object directly.

    Args:
        urdf_obj: URDF object (not inversed)
        tip_link: Tip link name
        tip_joint: Joint between object's tip_link and its parent
        grasp_pose: Relative to the object base (usually obtained from grasp dataset)
        init_data_dict: Optional initialization data

    Returns:
        attached_origin matrix
    """
    t_handle = np.eye(4, dtype=np.float64)
    tip_joint_angle = 0.0

    if init_data_dict:
        for joint_id, joint in enumerate(urdf_obj.robot.joints):
            if joint.name == tip_joint:
                tip_joint_angle = init_data_dict["object"]["qpos"][joint_id]

    # Update joint configuration and get transform
    urdf_obj.update_cfg({tip_joint: tip_joint_angle})
    t_handle = urdf_obj.get_transform(tip_link)

    t_grasp = np.matrix(grasp_pose)
    return t_grasp.I * t_handle


# def cal_attached_origin(obj_file, tip_link, tip_joint, grasp_pose, init_data_dict=None) -> np.matrix:
#     """
#     Calculate the trans from object tip_link to robot tip link
#     Args:
#         obj_file: regular urdf (not inversed)
#         tip_link:
#         tip_joint: joint between object's tip_link and its parent
#         grasp_pose: relative to the [object] base (usually obtained from grasp dataset)

#     Returns: attached_origin
#     """

#     t_handle = np.eye(4, dtype=np.float64)
#     obj_r = Urdf.load(obj_file)
#     tip_joint_angle = 0.0

#     if init_data_dict:
#         for joint_id, joint in enumerate(obj_r.joints):
#             if joint.name == tip_joint:
#                 tip_joint_angle = init_data_dict["object"]["qpos"][joint_id]

#     fk = obj_r.link_fk(cfg={tip_joint: tip_joint_angle})
#     for link in obj_r.links:
#         if link.name == tip_link:
#             t_handle = fk[link]

#     t_grasp = np.matrix(grasp_pose)
#     return t_grasp.I * t_handle


def get_pairs_from_dict(pairs_dict):
    pairs = set()

    for key, values in pairs_dict.items():
        for value in values:
            pairs.add(frozenset([key, value]))

    return [pair for pair in pairs]


def pair_list_to_dict(pair_list):
    pair_dict = {}

    for pair in pair_list:
        a, b = sorted(pair, reverse=True)

        if a not in pair_dict:
            pair_dict[a] = []
        if b not in pair_dict[a]:
            pair_dict[a].append(b)

    return pair_dict


def add_collision_pairs(yaml_data, robot_name="summit_franka", obj_tip_link="link_0"):

    self_collision_ignore = yaml_data["robot_cfg"]["kinematics"][
        "self_collision_ignore"
    ]
    ee_link = yaml_data["robot_cfg"]["kinematics"]["ee_link"]
    pairs = get_pairs_from_dict(self_collision_ignore)
    if robot_name == "summit_franka":
        robot_tcp_links = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]

    elif robot_name == "tiago_single":
        robot_tcp_links = [
            "gripper_link",
            "gripper_left_finger_link",
            "gripper_right_finger_link",
        ]

    elif robot_name == "r1_left":
        robot_tcp_links = ["left_arm_link6", "left_gripper_link1", "left_gripper_link2"]

    else:
        raise Exception("Please specify robot tcp links")

    add_pairs = {obj_tip_link: robot_tcp_links, ee_link: robot_tcp_links}

    for key, values in add_pairs.items():
        for value in values:
            pair = frozenset([key, value])
            if pair not in pairs:
                pairs.append(pair)
                print("Add pair ", pair)

    added_dict = pair_list_to_dict(pairs)
    yaml_data["robot_cfg"]["kinematics"].update({"self_collision_ignore": added_dict})
