"""
Sensor rig for managing cameras and sensors in Isaac Sim.

IMPORTANT: SimulationApp must be initialized before using this module.
Use automoma.simulation.sim_app_manager.get_simulation_app() first.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
import os
import numpy as np

from automoma.core.types import PoseType
from curobo.types.math import Pose
from automoma.utils.math_utils import create_colored_pointcloud, process_point_cloud

logger = logging.getLogger(__name__)


# Lazy imports for omni modules
_omni_imported = False
Camera = None


def _import_omni_modules():
    """Lazily import omni modules after SimulationApp is initialized."""
    global _omni_imported, Camera
    
    if _omni_imported:
        return
    
    # Check if SimulationApp is initialized
    from automoma.utils.sim_utils import require_simulation_app
    require_simulation_app()
    
    # Now safe to import omni modules
    from omni.isaac.sensor import Camera as _Camera
    
    Camera = _Camera
    
    _omni_imported = True
    logger.debug("Omni modules for sensors imported successfully")


class SensorRig:
    def __init__(self, sim):
        self.sim = sim
        # Import omni modules when SensorRig is created
        _import_omni_modules()
        self.cameras = {}
    def setup_sensors(self, sensor_cfgs):
        # Setup sensors based on the provided configurations
        """
        Setup 3 fixed cameras for data collection:
        1. ego_topdown: attached to robot end effector (panda_hand)
        2. ego_wrist: attached to robot end effector (panda_hand)
        3. fix_local: attached to object
        """
        
        self.sensor_cfgs = sensor_cfgs
                
        camera_configs = sensor_cfgs.get("cameras", {})
        
        # Compute scene bounds if available (for bounds checking)
        scene_bounds = self.compute_bounds("/World/scene")
            
        print("scene_bounds:", self.compute_bounds("/World/scene"))
        print("world_bounds:", self.compute_bounds("/World"))

        for camera_name, config in camera_configs.items():
            # Create camera prim path based on cuakr structure
            prim_path = config.get("prim_path")
            frequency = config.get("frequency", 30)
            resolution = config.get("resolution", (320, 240))
            
            # Create camera with matching cuakr resolution [320, 240]
            camera = Camera(
                prim_path=prim_path,
                frequency=frequency,
                resolution=resolution  # Height x Width - matching cuakr config
            )
            camera.initialize()
            camera.add_motion_vectors_to_frame()
            camera.add_distance_to_image_plane_to_frame()
            
            # Set focal length if specified
            if config.get("focal_length") is not None:
                camera.set_focal_length(config["focal_length"])
                print(f"Camera {camera_name} focal length set to {config['focal_length']}")
            
            # Check if camera pose is within bounds, if not use mirror_pose
            pose_to_use = config.get("pose", [])
            if scene_bounds is not None and self._is_pose_out_of_bounds(pose_to_use, scene_bounds):
                if "mirror_pose" in config:
                    logger.info(f"Camera {camera_name} pose is out of bounds, using mirror_pose")
                    pose_to_use = config.get("mirror_pose", pose_to_use)
            
            # Set camera pose based on type
            self._set_camera_pose(camera, camera_name, pose_to_use, PoseType[config.get("pose_type").upper()])
            
            self.cameras[camera_name] = camera
                
    def update(self):
        return
        # Update sensor states in the simulation
        if self.cameras.get("ego_topdown") is not None:
            # Get robot end effector pose using proper prim pose detection
            robot_link_prim_path = os.path.dirname(self.sensor_cfgs["cameras"]["ego_topdown"]["prim_path"])
            robot_link_pose_matrix = self.sim.get_prim_pose(robot_link_prim_path)
            
            if robot_link_pose_matrix is not None:
                # Convert matrix to Pose object and extract position
                robot_link_pose = Pose.from_matrix(robot_link_pose_matrix)
                robot_pos = robot_link_pose.position.cpu().numpy()
                
                # Update camera position (ego_topdown follows end effector)
                ego_pose = self.sensor_cfgs["cameras"]["ego_topdown"]["pose"]
                camera_pos = robot_pos + np.array(ego_pose["translate"])
                
                self.cameras["ego_topdown"].set_world_pose(
                    camera_pos.tolist(),
                    ego_pose["quat"],
                    camera_axes="usd"
                )
            else:
                print(f"Failed to get robot pose from {robot_link_prim_path}, skipping ego_topdown camera update")
    
    def get_obs(self):
        # Update sensor states
        self.update()
        
        # Retrieve sensor data from the simulation
        observations = {
            "images": {},
            "depth": {},
            "pointcloud": {}
        }
        for camera_name, camera in self.cameras.items():
            camera_cfg = self.sensor_cfgs["cameras"][camera_name]
            
            # Get RGB data (remove alpha channel) 
            rgba = camera.get_rgba()
            if rgba is not None:
                image = rgba[:, :, :3]
                observations["images"][camera_name] = image
            else:
                logger.warning(f"Camera {camera_name} RGB data is None")
                # Fallback to zeros if data is not ready
                res = camera.get_resolution()
                observations["images"][camera_name] = np.zeros((res[1], res[0], 3), dtype=np.uint8)

            # Get depth data only if depth is enabled
            if camera_cfg.get("depth", False):
                depth = camera.get_depth()
                if depth is not None:
                    observations["depth"][camera_name] = depth
                else:
                    logger.warning(f"Camera {camera_name} depth data is None")
                    # Fallback to zeros if data is not ready
                    res = camera.get_resolution()
                    observations["depth"][camera_name] = np.zeros((res[1], res[0]), dtype=np.float32)

            # Get point cloud data only if pointcloud is enabled
            if camera_cfg.get("pointcloud", False):
                pointcloud = camera.get_pointcloud()    
                pointcloud = self._process_pointcloud(camera_name, pointcloud, observations["images"].get(camera_name))
                if pointcloud is not None:
                    observations["pointcloud"][camera_name] = pointcloud
                
        return observations
    
    def compute_bounds(self, prim_path: str) -> np.ndarray:
        """
        Compute scene bounding box for camera positioning.
        Returns bounding box as [min_x, min_y, min_z, max_x, max_y, max_z]
        """
        from omni.usd import get_context
        min_point_gf, max_point_gf = get_context().compute_path_world_bounding_box(prim_path)
        min_arr = np.array(min_point_gf)
        max_arr = np.array(max_point_gf)
        
        bounding_box = np.concatenate((min_arr, max_arr))
        
        return bounding_box
    
    def _is_pose_out_of_bounds(self, pose: Any, bounds: np.ndarray) -> bool:
        """
        Check if pose's x and y coordinates are both out of scene bounds.
        Z coordinate is ignored.
        bounds: [min_x, min_y, min_z, max_x, max_y, max_z]
        """
        # Extract translate from pose
        if hasattr(pose, "translate"):
            translate = pose.translate
        elif isinstance(pose, dict) and "translate" in pose:
            translate = pose["translate"]
        elif isinstance(pose, (list, tuple)) and len(pose) >= 3:
            translate = pose[:3]
        else:
            return False
        
        translate = np.array(translate)
        
        # Extract bounds
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        
        # Check if both x and y are out of bounds
        x_out = translate[0] < min_x or translate[0] > max_x
        y_out = translate[1] < min_y or translate[1] > max_y
        
        return x_out and y_out

    def _process_pointcloud(self, camera_name: str, pointcloud: Optional[Any], image: Optional[Any]) -> Optional[Any]:
        """Process point cloud data if needed."""
        if pointcloud is None:
            return None
        
        # Get pointcloud config for this camera
        camera_cfg = self.sensor_cfgs.get("cameras", {}).get(camera_name, {})
        pc_cfg = camera_cfg.get("pointcloud")
        if pc_cfg is None:
            return None
        
        # Create colored pointcloud by combining xyz and rgb
        if image is not None:
            colored_pc = create_colored_pointcloud(pointcloud, image, ignore_nan=True)
        else:
            # If no image, just use pointcloud with dummy colors
            colored_pc = np.hstack([pointcloud.reshape(-1, 3), np.zeros((pointcloud.reshape(-1, 3).shape[0], 3))])
        
        # Process the colored pointcloud
        cfg = {
            'random_drop_points': pc_cfg.get('random_drop_points', 5000),
            'n_points': pc_cfg.get('num_points', 1024),
            'USE_FPS': pc_cfg.get('use_fps', True)
        }
        
        processed_pc = process_point_cloud(colored_pc, cfg)
        return processed_pc
    
    def _set_camera_pose(self, camera, camera_name, pose, pose_type: PoseType) -> None:
        """Set camera pose based on configuration."""
        # Handle dictionary/Config format with translate and orient
        if hasattr(pose, "translate") and hasattr(pose, "orient"):
            translate = pose.translate
            orient = pose.orient
        elif isinstance(pose, dict) and "translate" in pose and "orient" in pose:
            translate = pose["translate"]
            orient = pose["orient"]
        elif isinstance(pose, (list, tuple)) and len(pose) >= 6:
            translate = pose[:3]
            orient = pose[3:]
        else:
            logger.warning(f"Invalid pose format: {pose}. Using default [0,0,0], [0,0,0]")
            translate = [0, 0, 0]
            orient = [0, 0, 0]

        # Convert Euler angles to quaternion following cuakr pattern
        from automoma.utils.math_utils import euler_to_quat, euler_to_quat_omni
        if len(orient) == 3:
            try:
                rot_quat = euler_to_quat_omni(orient)
            except:
                # Fallback to our own euler_to_quat if Isaac Sim not available
                rot_quat = euler_to_quat(
                    np.array(orient) * np.pi / 180, order='xyz'
                )
                if hasattr(rot_quat, 'tolist'):
                    rot_quat = rot_quat.tolist()
        elif len(orient) == 4:
            rot_quat = orient
        else:
            rot_quat = [1, 0, 0, 0]  # Identity quaternion
            
        # Update sensor_cfgs with processed pose
        self.sensor_cfgs["cameras"][camera_name]["pose"]["quat"] = rot_quat

        if pose_type == PoseType.LOCAL:
            # Local pose relative to parent prim (robot end effector or object)
            camera.set_local_pose(
                translate,
                rot_quat,
                camera_axes="usd"
            )
        elif pose_type == PoseType.WORLD:
            # World pose
            camera.set_world_pose(
                translate,
                rot_quat,
                camera_axes="usd"
            )