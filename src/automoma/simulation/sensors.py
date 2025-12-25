"""
Sensor rig for managing cameras and sensors in Isaac Sim.

IMPORTANT: SimulationApp must be initialized before using this module.
Use automoma.simulation.sim_app_manager.get_simulation_app() first.
"""

from typing import Any, Dict, List, Optional
import logging

from automoma.core.types import PoseType


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
    from automoma.simulation.sim_app_manager import require_simulation_app
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
    def setup_sensors(self, sensor_cfgs):
        # Setup sensors based on the provided configurations
        """
        Setup 3 fixed cameras for data collection:
        1. ego_topdown: attached to robot end effector (panda_hand)
        2. ego_wrist: attached to robot end effector (panda_hand)
        3. fix_local: attached to object
        """
        
        self.sensor_cfgs = sensor_cfgs
        
        cameras = {}
        
        camera_configs = sensor_cfgs.get("cameras", {})

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
            
            # Set camera pose based on type
            self._set_camera_pose(camera, config.get("pose", []), PoseType[config.get("pose_type").upper()])
            
            cameras[camera_name] = camera
            
        self.cameras = cameras
    
    def update(self):
        # Update sensor states in the simulation
        pass
    
    def get_obs(self):
        # Retrieve sensor data from the simulation
        observations = {
            "images": {},
            "depth": {},
            "pointcloud": {}
        }
        for camera_name, camera in self.cameras.items():
            # Get RGB data (remove alpha channel) 
            rgba = camera.get_rgba()
            image = rgba[:, :, :3]
            if rgba is not None:
                observations["images"][camera_name] = image
            # Get depth data
            depth = camera.get_depth()
            if depth is not None:
                observations["depth"][camera_name] = depth
            # Get point cloud data
            pointcloud = camera.get_pointcloud()    
            pointcloud = self._process_pointcloud(camera_name, pointcloud, image)
            if pointcloud is not None:
                observations["pointcloud"][camera_name] = pointcloud
            
        return observations
    
    def _process_pointcloud(self, camera_name: str, pointcloud: Optional[Any], image: Optional[Any]) -> Optional[Any]:
        """Process point cloud data if needed."""
        if pointcloud is None:
            return None
        # Not required
        pc_cfg = self.sensor_cfgs.get("pointcloud", {}).get(camera_name)
        if pc_cfg is None:
            return None
        
    
    def _set_camera_pose(self, camera: Camera, pose: List[float], pose_type: PoseType) -> None:
        """Set camera pose based on configuration."""
        if pose_type == PoseType.LOCAL:
            # Local pose relative to parent prim (robot end effector or object)
            camera.set_local_pose(
                pose[:3],
                pose[3:],
                camera_axes="usd"
            )
        elif pose_type == PoseType.WORLD:
            # World pose
            camera.set_world_pose(
                pose[:3],
                pose[3:],
                camera_axes="usd"
            )