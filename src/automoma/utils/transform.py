import numpy as np
from scipy.spatial.transform import Rotation as R

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


if __name__ == "__main__":
    pose = np.array([1, 2, 3, 0.7071, 0, 0.7071, 0])  # 90 degrees around Y axis
    print("Pose:\n", pose)
    matrix = pose_to_matrix(pose)
    print("Transformation Matrix:\n", matrix)

    recovered_pose = matrix_to_pose(matrix)
    print("Recovered Pose:\n", recovered_pose)
    assert np.allclose(pose, recovered_pose, atol=1e-4), "Pose conversion failed!"