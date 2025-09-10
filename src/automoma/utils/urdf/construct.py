import tempfile
from scene_synthesizer.procedural_assets import URDFAsset
from automoma.utils.urdf.object import BaseObject
from automoma.utils.urdf.utils import init_scale_urdf
from yourdfpy import URDF
import numpy as np
from typing import Dict


def attach_object_to_robot(
    object: URDFAsset,
    robot: URDF,
    grasp_pose: np.ndarray,
    object_tip: str,
    object_ee_link: str,
    robot_ee_link: str,
    object_cfg: Dict[str, float] | None = None,
):
    """
    Attach an object to a robot at a specified grasp pose.

    :param object: The URDFAsset representing the object to attach.
    :param robot: The URDF of the robot to which the object will be attached.
    :param grasp_pose: A 4x4 numpy array representing the grasp pose in the robot's frame.
    :param object_tip: The link of the object that will be attached to the robot.
    :param object_tip_joint: The joint name of the object tip.
    :param robot_ee_link: The end-effector link of the robot.
    :return: A new URDFAsset with the object attached to the robot.
    """
    
    t_handle = np.eye(4, dtype=np.float64)
    
    # Update joint configuration and get transform
    if object_cfg is not None:
        object._model.update_cfg(object_cfg)
    t_handle = object._model.get_transform(object_tip)
    
    # calculate attached origin from grasp pose
    inversed_object_urdf = inverse_object(object, object_ee_link, object_tip)


    t_grasp = np.matrix(grasp_pose)
    attached_origin = t_grasp.I * t_handle

    vkc_robot = BaseObject("scene")

    obj_inv = BaseObject.from_robot(inversed_object_urdf._model.robot)

    vkc_robot._merge_robot(obj_inv, robot.robot)

    world_joint_name = ""
    for j in obj_inv.joints:
        if j.parent == "world":
            world_joint_name = j.name

    joint = vkc_robot.joint_map[world_joint_name]
    joint.parent = robot_ee_link
    joint.origin = attached_origin

    vkc_robot_urdf = URDF(vkc_robot)
    # vkc_robot_urdf.show()
    return vkc_robot_urdf


def inverse_object(urdf: URDFAsset, root_link: str, tip_link: str):

    base_object = BaseObject.from_robot(urdf._model.robot, create_world_joint=True)
    base_object.inverse_root_tip(root_link, tip_link)
    # create temp file
    new_urdf_asset = None
    return _robot_to_urdf_asset(base_object)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".urdf") as temp_file:
        URDF(base_object).write_xml_file(temp_file.name)
        new_urdf_asset = URDFAsset(temp_file.name)
        new_urdf_asset._attributes = {}

    # new_urdf_asset.show()
    return new_urdf_asset

def _robot_to_urdf_asset(robot) -> URDFAsset:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".urdf") as temp_file:
        URDF(robot).write_xml_file(temp_file.name)
        new_urdf_asset = URDFAsset(temp_file.name)
        if new_urdf_asset._attributes is None:
            new_urdf_asset._attributes = {}
        return new_urdf_asset


def load_urdf_asset(urdf_path: str, scale: float | None = None) -> URDFAsset:
    urdf_asset = URDFAsset(urdf_path)
    if scale is not None:
        from automoma.utils.scale_urdf import scale_urdf
        output_path = urdf_path.replace(".urdf", "output/test/scaled.urdf")
        scale_urdf(urdf_path, output_path, scale)
        urdf_asset = URDFAsset(output_path)
    return urdf_asset


if __name__ == "__main__":
    import trimesh.transformations as tra
    import os
    import shutil
    
    output_dir = "output/test/kitchen_7221"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    
    # urdf = URDFAsset("/home/xinhai/Documents/cuakr-docker/scene/infinigen/assets/partnet_mobility/processed_data/Microwave/7221/mobility.urdf")
    urdf = URDFAsset("/home/xinhai/Documents/automoma/assets/object/Microwave/7221/mobility.urdf")
    
    scene = urdf.scene()
    
    from scene_synthesizer.exchange.urdf import scene_as_urdf
    plain_urdf = scene_as_urdf(scene)
    urdf = _robot_to_urdf_asset(plain_urdf.robot)

    # scene.show()
    
    # urdf._model = init_scale_urdf(urdf._model, 0.4, None)
    
    urdf._model.write_xml_file(os.path.join(output_dir, "mobility.urdf"))
    
    # new_urdf_asset = inverse_object(
    #     urdf, "base", "link_0"
    # )  # tip link is the link that will be attached to the robot
    # new_urdf_asset._model.write_xml_file("/home/xinhai/Documents/automoma/assets/object/Microwave/7221/mobility_inversed.urdf")
    robot_urdf = URDF.load(
        # "/home/yida/projects/cuakr-docker/assets/robot/robot/summit_franka/summit_franka.urdf"
        "/home/xinhai/Documents/automoma/assets/robot/summit_franka/summit_franka.urdf"
    )
    grasp_pose = np.load("/home/xinhai/Documents/automoma/assets/object/Microwave/7221/grasp/0000.npy")
    print(grasp_pose)
    from automoma.utils.transform import pose_to_matrix
    grasp_pose = pose_to_matrix(grasp_pose)
    
    grasp_pose_adjust = tra.euler_matrix(-np.pi / 2, 0, 0, "rxyz") # TODO: what is this?
    grasp_pose = grasp_pose @ grasp_pose_adjust
    vkc_robot = attach_object_to_robot(
        urdf, robot_urdf, grasp_pose, "object_link_0", "object_base", "ee_link", object_cfg={"object_joint_0": 0.0}
    )
    # vkc_robot.write_xml_file("output/test/attached_object_scene.urdf")
    vkc_robot_asset = _robot_to_urdf_asset(vkc_robot.robot)
    vkc_robot_asset.get_bounds()
    scene = vkc_robot_asset.scene()
    scene.subscene(["object"]).export(os.path.join(output_dir, "attached_object_scene.urdf"))


if __name__ == "__main2__":
    from scene_synthesizer.procedural_assets import RefrigeratorAsset
    import trimesh.transformations as tra
    from scene_synthesizer.procedural_scenes import Scene

    import os
    import shutil
    
    output_dir = "output/test/kitchen_syn"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    urdf = RefrigeratorAsset()
    urdf._model.write_xml_file(os.path.join(output_dir, "refrigerator.urdf"))
    # urdf = URDFAsset("/home/yida/repos/scene_synthesizer/examples/kitchen/refrigerator.urdf")
    # new_urdf_asset = inverse_object(
    #     urdf, "world", "door"
    # )  # tip link is the link that will be attached to the robot
    # new_urdf_asset._model.write_xml_file("output/test/inversed_refrigerator.urdf")
    
    robot_urdf = URDF.load(
        # "/home/yida/projects/cuakr-docker/assets/robot/robot/summit_franka/summit_franka.urdf"
        "/home/xinhai/Documents/automoma/assets/robot/summit_franka/summit_franka.urdf"
    )
    grasp_pose = np.array(
        [
            [6.123234e-17, 0.000000e00, 1.000000e00, -3.150000e-01],
            [0.000000e00, 1.000000e00, 0.000000e00, -0.4],
            [-1.000000e00, 0.000000e00, 6.123234e-17, 9.500000e-01],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    grasp_pose_adjust = tra.euler_matrix(-np.pi / 2, 0, 0, "rxyz")
    grasp_pose = grasp_pose @ grasp_pose_adjust
    vkc_robot = attach_object_to_robot(
        urdf, robot_urdf, grasp_pose, "door", "world","ee_link", object_cfg={"door_joint": 0.0}
    )
    # vkc_robot.write_xml_file("output/test/attached_object_scene.urdf")
    vkc_robot_asset = _robot_to_urdf_asset(vkc_robot.robot)
    vkc_robot_asset.get_bounds()
    scene = vkc_robot_asset.scene()
    scene.subscene(["object"]).export(os.path.join(output_dir, "attached_object_scene.urdf"))

