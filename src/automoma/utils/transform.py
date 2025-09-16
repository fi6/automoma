import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_to_euler(quat: np.ndarray, order: str = 'xyz') -> np.ndarray:
    """Convert a quaternion to Euler angles.

    Args:
        quat: np.ndarray, shape (4,), [w, x, y, z] (quaternion WXYZ order)
        order: str, order of Euler angles, e.g., 'xyz', 'zyx'

    Returns:
        euler_angles: np.ndarray, shape (3,), Euler angles in radians
    """
    assert quat.shape == (4,)
    # Convert quaternion from WXYZ to XYZW for scipy
    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
    rot = R.from_quat(quat_xyzw)  # XYZW
    euler_angles = rot.as_euler(order)
    return euler_angles

def euler_to_quat(euler_angles: np.ndarray, order: str = 'xyz') -> np.ndarray:
    """Convert Euler angles to a quaternion.

    Args:
        euler_angles: np.ndarray, shape (3,), Euler angles in radians
        order: str, order of Euler angles, e.g., 'xyz', 'zyx'

    Returns:
        quat: np.ndarray, shape (4,), [w, x, y, z] (quaternion WXYZ order)
    """
    assert euler_angles.shape == (3,)
    rot = R.from_euler(order, euler_angles)
    quat_xyzw = rot.as_quat()  # XYZW
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return quat_wxyz

def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert a pose (position + quaternion(WXYZ)) to a 4x4 transformation matrix.

    Args:
        pose: np.ndarray, shape (7,), [x, y, z, w, x, y, z] (quaternion WXYZ order)

    Returns:
        matrix: np.ndarray, shape (4, 4)
    """
    assert pose.shape == (7,)
    position = pose[:3]
    quat = pose[3:]  # [w, x, y, z]

    # Convert quaternion from WXYZ to XYZW for scipy
    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
    rot = R.from_quat(quat_xyzw)  # XYZW
    matrix = np.eye(4)
    matrix[:3, :3] = rot.as_matrix()
    matrix[:3, 3] = position
    return matrix

def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 transformation matrix to a pose (position + quaternion(WXYZ)).

    Args:
        matrix: np.ndarray, shape (4, 4)

    Returns:
        pose: np.ndarray, shape (7,), [x, y, z, w, x, y, z] (quaternion WXYZ order)
    """
    assert matrix.shape == (4, 4)
    position = matrix[:3, 3]
    rot = R.from_matrix(matrix[:3, :3])
    quat_xyzw = rot.as_quat()  # XYZW
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    pose = np.concatenate([position, quat_wxyz])
    return pose


def single_axis_self_rotation(matrix: np.ndarray, axis: str, angle: float) -> np.ndarray:
    """Apply a single-axis rotation to a 4x4 transformation matrix.

    Args:
        matrix: np.ndarray, shape (4, 4)
        axis: 'x', 'y', or 'z' indicating the axis of rotation
        angle: float, rotation angle in radians

    Returns:
        rotated_matrix: np.ndarray, shape (4, 4)
    """
    assert matrix.shape == (4, 4)
    
    rot = R.from_euler(axis, angle)
    R_local = rot.as_matrix() # 3x3 rotation matrix
    
    # Current matrix
    result = matrix.copy()
    result[:3, :3] = matrix[:3, :3] @ R_local
    
    return result


from typing import Optional
from curobo.types.math import Pose
from yourdfpy import URDF
def get_open_ee_pose(object_pose: Pose, grasp_pose: Pose, object_urdf: URDF, handle: str, joint_cfg: dict, default_joint_cfg: Optional[dict] = None) -> Pose:
    '''
    Compute the open end-effector pose given the object pose, grasp pose, and robot joint configuration.
    Args:
        object_pose: Pose of the object in world frame
        grasp_pose: Pose of the grasp in object frame
        object_urdf: URDF object representing the robot
        handle: Name of the end-effector link in the URDF
        joint_cfg: Dictionary of joint names to angles for the open configuration
        default_joint_cfg: Optional dictionary of joint names to angles for the default configuration
    Returns:
        open_grasp_pose: Pose of the grasp in world frame when in the open
    '''
    
    # 1. Fix default angle is not zero
    if default_joint_cfg is not None:
        object_urdf.update_cfg(default_joint_cfg)
        
        
    # 2: Compute relative pose of the grasp and handle in object frame
    T_object_handle_init = Pose.from_matrix(object_urdf.get_transform(handle, "base"))
    T_object_grasp_init = grasp_pose
    T_handle_grasp = T_object_handle_init.inverse().multiply(T_object_grasp_init)
    
    # 3. Compute the open grasp pose in object frame
    object_urdf.update_cfg(joint_cfg)
    T_object_handle_open = Pose.from_matrix(object_urdf.get_transform(handle, "base"))
    T_object_grasp_open = T_object_handle_open.multiply(T_handle_grasp)

    # 4. Compute the open grasp pose in world frame
    T_world_object = object_pose
    T_world_grasp_open = T_world_object.multiply(T_object_grasp_open)

    return T_world_grasp_open


if __name__ == "__main__":
    pose = np.array([1, 2, 3, 0.7071, 0, 0.7071, 0])  # 90 degrees around Y axis
    print("Pose:\n", pose)
    matrix = pose_to_matrix(pose)
    print("Transformation Matrix:\n", matrix)

    recovered_pose = matrix_to_pose(matrix)
    print("Recovered Pose:\n", recovered_pose)
    assert np.allclose(pose, recovered_pose, atol=1e-4), "Pose conversion failed!"