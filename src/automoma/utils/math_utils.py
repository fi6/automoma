from typing import List, Union, Tuple, Optional
import numpy as np
import torch
import math
from yourdfpy import URDF
from curobo.types.math import Pose
from curobo.geom.types import WorldConfig, Cuboid
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

def expand_to_pairs(start_iks: torch.Tensor, goal_iks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expand start and goal IKs into pairs.
    If they have different counts, create a Cartesian product (N x M).
    """
    # Create Cartesian product
    # goal_iks: [M, D] -> repeat(N, 1) -> [N*M, D]
    # start_iks: [N, D] -> repeat_interleave(M) -> [N*M, D]
    goal_iks_expanded = goal_iks.repeat(start_iks.shape[0], 1)
    start_iks_expanded = torch.repeat_interleave(start_iks, goal_iks.shape[0], dim=0)
    
    return start_iks_expanded, goal_iks_expanded

def quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Calculate the angular distance between two quaternions.
    Assumes [w, x, y, z] format.
    """
    # Ensure they are unit quaternions
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    q2 = q2 / (np.linalg.norm(q2) + 1e-8)
    
    # Inner product
    dot = np.abs(np.sum(q1 * q2))
    dot = np.clip(dot, -1.0, 1.0)
    
    # Angular distance in radians
    angle = 2.0 * np.arccos(dot)
    
    # Ensure the smallest angle (if the angle is greater than pi, subtract from 2*pi)
    if angle > np.pi:
        angle = 2 * np.pi - angle
        
    return angle

def _convert_to_list(x: Union[List[float], np.ndarray, torch.Tensor]) -> List[float]:
    """Helper function to reliably convert data into a Python list."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def get_transform(transform):
    translate = transform.get("translate", [0, 0, 0])
    orient = transform.get("orient", [0, 0, 0])
    scale = transform.get("scale", [1, 1, 1])

    # Convert Euler angles to quaternion
    # TODO: According to the experiment and the doc from https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_camera.html, add extrinsic=False
    import omni.isaac.core.utils.torch.rotations as rot_utils

    if len(orient) == 3:
        rot_quat = rot_utils.euler_angles_to_quats(
            torch.tensor([orient], dtype=torch.float32), degrees=True, extrinsic=False
        )[0].tolist()
    elif len(orient) == 4:
        rot_quat = orient

    return translate + rot_quat, scale


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiplies two quaternions using NumPy vector operations.

    This function assumes the quaternion format is [w, x, y, z].

    Args:
        q1: A NumPy array of one or more quaternions.
        q2: A NumPy array of one or more quaternions.

    Returns:
        The resulting quaternion(s) as a new NumPy array.
    """
    # Ensure inputs are NumPy arrays
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)

    # Handle both single quaternions and batches of quaternions
    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    w2 = q2[..., 0]
    x2 = q2[..., 1]
    y2 = q2[..., 2]
    z2 = q2[..., 3]

    # Calculate the components of the resulting quaternion(s)
    w_res = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x_res = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_res = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_res = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    # Stack the results into a new array
    return np.stack([w_res, x_res, y_res, z_res], axis=-1)


def pose_multiply(p1, p2):
    """
    Multiplies two poses, preserving the data type of the first input.

    This function converts two poses into a standard format, performs the
    multiplication using the curobo library, and then casts the result
    back to the original type of the first pose (including its device
    and dtype for tensors).

    Args:
        p1: The first pose in [x, y, z, qw, qx, qy, qz] format.
               Can be a list, NumPy array, or PyTorch tensor.
        p2: The second pose, in a compatible format.

    Returns:
        The resulting pose, in the same data type as `p1`.
    """
    # Store the original type and properties to reconstruct the output later
    original_type = type(p1)
    device = getattr(p1, "device", None)
    dtype = getattr(p1, "dtype", None)

    # Use the curobo Pose object for the core calculation
    curobo_pose1 = Pose.from_list(_convert_to_list(p1))
    curobo_pose2 = Pose.from_list(_convert_to_list(p2))

    # Perform the multiplication
    result_pose = curobo_pose1.multiply(curobo_pose2)
    result_list = result_pose.to_list()

    # Convert the result back to the original input type
    if original_type is torch.Tensor:
        return torch.as_tensor(result_list, device=device, dtype=dtype)
    if original_type is np.ndarray:
        return np.array(result_list, dtype=dtype)

    return result_list


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
    T_object_handle_init = Pose.from_matrix(object_urdf.get_transform(handle))
    T_object_grasp_init = grasp_pose
    T_handle_grasp = T_object_handle_init.inverse().multiply(T_object_grasp_init)
    
    # 3. Compute the open grasp pose in object frame
    object_urdf.update_cfg(joint_cfg)
    T_object_handle_open = Pose.from_matrix(object_urdf.get_transform(handle))
    T_object_grasp_open = T_object_handle_open.multiply(T_handle_grasp)

    # 4. Compute the open grasp pose in world frame
    T_world_object = object_pose
    T_world_grasp_open = T_world_object.multiply(T_object_grasp_open)

    return T_world_grasp_open


def mark_cuboid_as_empty(esdf,
                         cuboid: Cuboid,
                         empty_value: float | None = None):
    """
    Mark points in a specific cuboid region as empty by updating the feature_tensor.
    Args:
        cuboid (Cuboid): The cuboid region to mark as empty.
        empty_value (float): Value to set for empty voxels in the feature_tensor.
    """
    if esdf.feature_tensor is None or esdf.xyzr_tensor is None:
        raise ValueError("feature_tensor and xyzr_tensor must be initialized.")

    import numpy as np

    # Get voxel positions as numpy array
    points_xyz = esdf.xyzr_tensor[:, :3].cpu().numpy()
    center = np.array(cuboid.pose[:3])
    dims = np.array(cuboid.dims)
    half_dims = dims / 2.0

    # Default: no rotation (identity)
    rot_matrix = np.eye(3)
    if len(cuboid.pose) == 7:
        from scipy.spatial.transform import Rotation as R
        quat = cuboid.pose[3:]  # [qw, qx, qy, qz]
        rot_matrix = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

    # Transform points to cuboid local frame
    points_local = points_xyz - center
    points_local = points_local @ rot_matrix  # Apply rotation

    # Mask for points inside the cuboid (OBB)
    mask = np.all((points_local >= -half_dims) & (points_local <= half_dims), axis=1)

    # Update the feature_tensor for the selected voxels
    if empty_value is None:
        empty_value = max(esdf.feature_tensor.min(), -1.0)
    esdf.feature_tensor[mask] = empty_value
    return esdf


def ik_clustering_kmeans_ap_fallback(
    all_iks, kmeans_clusters=500, return_stats=False, **kwargs
):
    # kmeans to 300 iks and then run ap
    # Perform KMeans clustering to reduce the number of clusters to 300
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min

    emb = all_iks.detach().cpu()
    
    initial_count = emb.size()[0]
    stats = {
        'initial_ik_count': initial_count,
        'kmeans_applied': False,
        'kmeans_clusters': 0,
    }

    # TODO: This is not a good idea, because we need to run AP on the kmeans centers
    # if emb.size()[0] <= kmeans_clusters:
    #     return emb
    if emb.size()[0] <= kmeans_clusters:
        kmeans_clusters = n_clusters = emb.size()[0]

    kmeans = KMeans(n_clusters=kmeans_clusters, random_state=0).fit(emb)

    # Collect the cluster centers from KMeans
    kmeans_centers = (
        emb[pairwise_distances_argmin_min(kmeans.cluster_centers_, emb)[0]]
        .clone()
        .detach()
    )
    
    stats['kmeans_applied'] = True
    stats['kmeans_clusters'] = kmeans_centers.size()[0]

    result = ik_clustering_ap_fallback(kmeans_centers, return_stats=return_stats, **kwargs)
    
    if return_stats:
        clustered_iks, ap_stats = result
        stats.update(ap_stats)
        return clustered_iks, stats
    else:
        return result

def ik_clustering_ap_fallback(all_iks, ap_fallback_clusters=30, ap_clusters_upperbound=50, ap_clusters_lowerbound=10, return_stats=False):
    from sklearn.cluster import AffinityPropagation, KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import torch

    emb = all_iks.detach().cpu().numpy()
    
    initial_count = emb.shape[0]
    
    stats = {
        'ap_input_count': initial_count,
        'ap_unique_labels': 0,
        'clustering_method': 'none',
        'final_ik_count': 0,
    }

    # TODO: Debug for those origial numbers small than fallback_clusters, causing NO iks
    if emb.shape[0] <= ap_fallback_clusters:
        result = torch.tensor(emb, device=all_iks.device)
        stats['clustering_method'] = 'none'
        stats['final_ik_count'] = result.shape[0]
        if return_stats:
            return result, stats
        return result

    # First try with AffinityPropagation
    af = AffinityPropagation(affinity="precomputed", damping=0.9)
    similarity_matrix = cosine_similarity(emb)
    af.fit(similarity_matrix)
    labels = af.labels_
    unique_label_count = len(np.unique(labels))
    
    stats['ap_unique_labels'] = unique_label_count

    print(
        "Embedding shape:",
        emb.shape,
        "AP labels:",
        len(labels),
        "unique labels:",
        unique_label_count
    )

    if (
        unique_label_count > ap_clusters_upperbound or unique_label_count < ap_clusters_lowerbound
    ):  # TODO: 25 is a magic number, get from data
        # Fall back to KMeans with target number of clusters
        print(
            f"AP produced {unique_label_count} clusters, not in range [{ap_clusters_lowerbound}, {ap_clusters_upperbound}]. Falling back to KMeans with {ap_fallback_clusters} clusters."
        )
        kmeans = KMeans(n_clusters=ap_fallback_clusters).fit(emb)
        labels = kmeans.labels_
        stats['clustering_method'] = 'kmeans_fallback'
        stats['kmeans_fallback_clusters'] = ap_fallback_clusters
    else:
        stats['clustering_method'] = 'affinity_propagation'

    # Extract median from each cluster
    unique_iks = []
    for i in np.unique(labels):
        cluster_set = all_iks[(labels == i).tolist()]
        set_index = cluster_set[:, 0].argsort()
        cluster_median = cluster_set[set_index[len(set_index) // 2]]
        unique_iks.append(cluster_median)

    unique_iks = torch.stack(unique_iks)
    stats['final_ik_count'] = unique_iks.shape[0]
    print("Unique IKs shape:", unique_iks.shape)
    
    if return_stats:
        return unique_iks, stats
    return unique_iks
