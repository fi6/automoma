from typing import List, Union, Tuple, Optional
import numpy as np
import torch
import math
from yourdfpy import URDF
from curobo.types.math import Pose
from curobo.geom.types import WorldConfig, Cuboid
from scipy.spatial.transform import Rotation as R


def unpack_ik(ik: Union[torch.Tensor, np.ndarray, List[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpack IK solution into robot state and environment state.

    Args:
        ik: Union[torch.Tensor, np.ndarray, List[float]], IK solution with last element as env state
    Returns:
        Returns:
        robot_state: torch.Tensor, robot joint positions
        env_state: torch.Tensor, environment state (e.g., object position)
    """
    if isinstance(ik, list):
        ik = torch.tensor(ik, dtype=torch.float32)
    elif isinstance(ik, np.ndarray):
        ik = torch.from_numpy(ik).float()

    robot_state = ik[:-1]
    env_state = ik[-1].unsqueeze(0)

    return robot_state, env_state
        
        
def quat_to_euler(quat: np.ndarray, order: str = "xyz") -> np.ndarray:
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


def euler_to_quat(euler_angles: np.ndarray, order: str = "xyz") -> np.ndarray:
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
    R_local = rot.as_matrix()  # 3x3 rotation matrix

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
    goal_iks_expanded = goal_iks.repeat(start_iks.shape[0], 1).clone()
    start_iks_expanded = torch.repeat_interleave(start_iks, goal_iks.shape[0], dim=0).clone()

    return start_iks_expanded, goal_iks_expanded

def stack_iks_angle(iks: torch.Tensor, angle: float) -> torch.Tensor:
        """Add joint angle information to IK solutions"""
        joint_angle_expand = (
            torch.tensor([angle], device=iks.device)
            .unsqueeze(0)
            .expand(iks.shape[0], -1)
        )
        return torch.cat((iks, joint_angle_expand), dim=1)
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

def euler_to_quat_omni(orient):
    # Convert Euler angles to quaternion
    # TODO: According to the experiment and the doc from https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_camera.html, add extrinsic=False
    import omni.isaac.core.utils.torch.rotations as rot_utils

    if len(orient) == 3:
        rot_quat = rot_utils.euler_angles_to_quats(
            torch.tensor([orient], dtype=torch.float32), degrees=True, extrinsic=False
        )[0].tolist()
    elif len(orient) == 4:
        rot_quat = orient

    return rot_quat


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
    original_type = type(p2)
    device = getattr(p2, "device", None)
    dtype = getattr(p2, "dtype", None)

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


def get_open_ee_pose(
    object_pose: Pose,
    grasp_pose: Pose,
    object_urdf: URDF,
    handle: str,
    joint_cfg: dict,
    default_joint_cfg: Optional[dict] = None,
) -> Pose:
    """
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
    """

    # 1. Fix default angle is not zero
    if default_joint_cfg is not None:
        # Update joint angle to tenser
        default_joint_cfg = {k: torch.tensor(v) for k, v in default_joint_cfg.items()}
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


def mark_cuboid_as_empty(esdf, cuboid: Cuboid, empty_value: float | None = None):
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
    all_iks,
    kmeans_clusters=500,
    ap_fallback_clusters=30,
    ap_clusters_upperbound=50,
    ap_clusters_lowerbound=10,
    **kwargs,
):
    """
    Cluster IK solutions using KMeans followed by Affinity Propagation.
    Returns a list of boolean masks [kmeans_mask, ap_mask, final_mask].
    """
    from sklearn.cluster import KMeans, AffinityPropagation
    from sklearn.metrics import pairwise_distances_argmin_min
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    emb = all_iks.detach().cpu().numpy()
    n_samples = emb.shape[0]
    masks = []

    # 1. KMeans Filter
    if n_samples <= kmeans_clusters:
        kmeans_mask = np.ones(n_samples, dtype=bool)
    else:
        kmeans = KMeans(n_clusters=kmeans_clusters, random_state=0).fit(emb)
        kmeans_indices = pairwise_distances_argmin_min(kmeans.cluster_centers_, emb)[0]
        kmeans_mask = np.zeros(n_samples, dtype=bool)
        kmeans_mask[kmeans_indices] = True
    masks.append(kmeans_mask)

    # 2. AP Filter (on KMeans results)
    kmeans_selected_indices = np.where(kmeans_mask)[0]
    emb_reduced = emb[kmeans_selected_indices]

    if emb_reduced.shape[0] <= ap_fallback_clusters:
        ap_mask_reduced = np.ones(emb_reduced.shape[0], dtype=bool)
    else:
        af = AffinityPropagation(affinity="precomputed", damping=0.9, random_state=0)
        similarity_matrix = cosine_similarity(emb_reduced)
        af.fit(similarity_matrix)
        labels = af.labels_
        unique_labels = np.unique(labels)

        if len(unique_labels) > ap_clusters_upperbound or len(unique_labels) < ap_clusters_lowerbound:
            kmeans_ap = KMeans(n_clusters=min(ap_fallback_clusters, emb_reduced.shape[0]), random_state=0).fit(
                emb_reduced
            )
            labels = kmeans_ap.labels_
            unique_labels = np.unique(labels)

        ap_indices_reduced = []
        for i in unique_labels:
            cluster_indices = np.where(labels == i)[0]
            cluster_set = emb_reduced[cluster_indices]
            set_index = cluster_set[:, 0].argsort()
            median_idx = cluster_indices[set_index[len(set_index) // 2]]
            ap_indices_reduced.append(median_idx)

        ap_mask_reduced = np.zeros(emb_reduced.shape[0], dtype=bool)
        ap_mask_reduced[ap_indices_reduced] = True

    # Map back to original indices
    ap_mask = np.zeros(n_samples, dtype=bool)
    ap_mask[kmeans_selected_indices[ap_mask_reduced]] = True
    masks.append(ap_mask)

    # 3. Final Filter
    masks.append(ap_mask.copy())

    return masks

def create_colored_pointcloud(pc_xyz, rgb_image, ignore_nan=True):
    """
    Create a colored point cloud by combining XYZ data with RGB values

    Args:
        pc_xyz: Point cloud data of shape (H, W, 3)
        rgb_image: RGB image of shape (H, W, 3) or (H, W, 4) for RGBA
        ignore_nan: Whether to filter out NaN values (default: True)
    Returns:
        colored_pc: Combined XYZ+RGB data of shape (N, 6) where N is the number of valid points
    """
    height, width = pc_xyz.shape[:2]
    points = pc_xyz.reshape(-1, 3)
    rgb = rgb_image[:, :, :3] if rgb_image.shape[2] == 4 else rgb_image
    colors = rgb.reshape(-1, 3)
    valid_mask = np.ones(points.shape[0], dtype=bool)
    if ignore_nan:
        nan_mask = ~np.isnan(points).any(axis=1)
        valid_mask = valid_mask & nan_mask
    valid_points = points[valid_mask]
    valid_colors = colors[valid_mask]
    if valid_colors.max() > 1.0:
        valid_colors = valid_colors / 255.0
    colored_pc = np.hstack((valid_points, valid_colors))
    return colored_pc


def process_point_cloud(point_cloud, cfg=None):
    """
    Process point cloud: filter, downsample, etc.
    
    Args:
        point_cloud: Point cloud with shape (N, 6) - positions (xyz) and colors (rgb)
        cfg: Configuration dict with optional keys:
            - random_drop_points: int, default 5000
            - n_points: int, default 1024
            - USE_FPS: bool, default True
        
    Returns:
        Processed point cloud
    """
    if point_cloud is None or point_cloud.shape[0] == 0:
        print("Empty point cloud provided. Returning None.")
        return None
    
    # Default configuration
    if cfg is None:
        cfg = {}
    random_drop_points = cfg.get('random_drop_points', 5000)
    n_points = cfg.get('n_points', 1024)
    USE_FPS = cfg.get('USE_FPS', True)
    
    # 1. Filter out invalid points
    valid_mask = ~np.isnan(point_cloud[:, :3]).any(axis=1) & ~np.isinf(point_cloud[:, :3]).any(axis=1)
    points = point_cloud[valid_mask]
    
    # 2. Randomly drop points if too many
    if points.shape[0] > random_drop_points:
        indices = np.random.choice(points.shape[0], random_drop_points, replace=False)
        points = points[indices]
    
    # 3. Downsample to target number of points
    if points.shape[0] > n_points:
        if USE_FPS:
            try:
                import fpsample
                # Use farthest point sampling
                indices = fpsample.bucket_fps_kdline_sampling(points[:, :3], n_points, h=3)
            except ImportError:
                # Fallback to random sampling if fpsample not available
                indices = np.random.choice(points.shape[0], n_points, replace=False)
        else:
            # Fallback to random sampling
            indices = np.random.choice(points.shape[0], n_points, replace=False)
        points = points[indices]
    elif points.shape[0] < n_points:
        # 4. Pad with duplicated points if not enough
        n_pad = n_points - points.shape[0]
        if n_pad > 0:
            pad_indices = np.random.choice(points.shape[0], n_pad)
            padding = points[pad_indices]
            points = np.concatenate([points, padding], axis=0)
    
    return points