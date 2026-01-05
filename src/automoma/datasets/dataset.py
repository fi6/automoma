import os
import shutil
from pathlib import Path
import numpy as np
from automoma.utils.logging import logger

class BaseDatasetWrapper:
    '''
    Format:
    {
        "obs_data": {
            "images": {
                "camera_1": np.array,
                "camera_2": np.array,
                ...
            },
            "depth": {
                "camera_1": np.array,
                "camera_2": np.array,
                ...
            },
        }
        "joint_data": joint_data,
        "eef_pose_data": eef_pose_data,
        "action_data": action_data
    }
    '''
    def __init__(self):
        raise NotImplementedError("BaseDatasetWrapper is an abstract class and cannot be instantiated directly.")
    def create(self):
        raise NotImplementedError("The create method must be implemented by subclasses.")
    def add(self, data):
        raise NotImplementedError("The add method must be implemented by subclasses.")
    def save(self):
        raise NotImplementedError("The save method must be implemented by subclasses.")

    def close(self):
        raise NotImplementedError("The close method must be implemented by subclasses.")

class DataStorageProxy:
    def __init__(self, physical_root, use_ramdisk=False, ramdisk_base="/dev/shm"):
        self.physical_root = Path(physical_root)
        self.use_ramdisk = use_ramdisk
        
        if self.use_ramdisk:
            # We use a subfolder to avoid collisions with other processes
            self.active_root = Path(ramdisk_base) / "lerobot_buffer" / self.physical_root.name
        else:
            self.active_root = self.physical_root

    def setup(self):
        """Prepare the working directory."""
        if self.active_root.exists():
            shutil.rmtree(self.active_root)
        # self.active_root.mkdir(parents=True, exist_ok=True)
        return self.active_root

    def post_save_sync(self):
        """Called after save_episode() to offload data and keep RAM clean."""
        if not self.use_ramdisk:
            return

        # 1. Sync the current state to physical disk
        # This moves the .mp4 and .parquet files created by LeRobot
        self.physical_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.active_root, self.physical_root, dirs_exist_ok=True)
        
        # 2. Critical cleanup: Remove the temporary raw images to free up RAM
        # LeRobot keeps these in 'videos/temp_images' until the episode is closed
        temp_dir = self.active_root / "videos" / "temp_images"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.debug("RAMDisk: Cleared temporary raw frames.")

    def finalize(self):
        """Clean up the RAMDisk entirely at the end of the session."""
        if self.use_ramdisk and self.active_root.exists():
            shutil.rmtree(self.active_root)
            logger.info("RAMDisk: Final cleanup complete.")

class LeRobotDatasetWrapper(BaseDatasetWrapper):
    def __init__(self, cfg):
        
        self.dataset = None
        self.cfg = cfg
        self.task = cfg.get("task", "manipulation")
        
        # Initialize the Proxy for IO management
        self.storage_proxy = DataStorageProxy(
            physical_root=Path(self.cfg.root)/self.cfg.repo_id,
            use_ramdisk=self.cfg.get("use_ramdisk", False),
            ramdisk_base=self.cfg.get("ramdisk_path", "/dev/shm")
        )
        
    def _init_features(self):
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (self.cfg.state_dim,),
                "names": [self.cfg.state_names],
            },
            "observation.eef": {
                "dtype": "float32",
                "shape": (7,),  # [x, y, z, qx, qy, qz, qw]
                "names": [["x", "y", "z", "qx", "qy", "qz", "qw"]],
            },
            "action": {
                "dtype": "float32",
                "shape": (self.cfg.state_dim,),
                "names": [self.cfg.state_names],
            },
        }
        
        # Add image features for all cameras
        for cam in self.cfg.camera.names:
            features[f"observation.images.{cam}"] = {
                "dtype": "video" if self.cfg.use_videos else "image",
                "shape": (3, self.cfg.camera.height, self.cfg.camera.width),
                "names": ["channels", "height", "width"],
            }
        
        # Add depth features only for cameras that have depth enabled
        for cam in self.cfg.camera.names:
            cam_cfg = self.cfg.camera.cameras.get(cam, {})
            if cam_cfg.get("depth", False):
                features[f"observation.depth.{cam}"] = {
                    "dtype": "float32",
                    "shape": (1, self.cfg.camera.height, self.cfg.camera.width),
                    "names": ["channels", "height", "width"],
                }
        
        # Add pointcloud features only for cameras that have pointcloud enabled
        total_pc_points = 0
        for cam in self.cfg.camera.names:
            cam_cfg = self.cfg.camera.cameras.get(cam, {})
            if cam_cfg.get("pointcloud", False) and "pointcloud" in cam_cfg:
                total_pc_points += cam_cfg.get("pointcloud", {}).get("num_points", 4096)
        
        if total_pc_points > 0:
            features[f"observation.pointcloud"] = {
                "dtype": "float32",
                "shape": (total_pc_points, 6),
                "names": ["points", "xyzrgb"],
            }
        
        self.features = features

    def create(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        import shutil
        from pathlib import Path
        
        # init features
        self._init_features()
        
        # The proxy decides if this is /dev/shm or /data/...
        active_path = self.storage_proxy.setup()
        
        # create dataset
        self.dataset = LeRobotDataset.create(
            repo_id = self.cfg.repo_id,
            root = active_path,
            fps = self.cfg.fps,
            features= self.features,
            robot_type= self.cfg.robot_type,
            use_videos= self.cfg.use_videos,
        )
        logger.info(f"Created dataset at {active_path}")
        
    def add(self, data):
        frame = {
            "observation.state": np.array(data["joint_data"], dtype=np.float32),
            "observation.eef": np.array(data["eef_pose_data"], dtype=np.float32),
            "action": np.array(data["action_data"], dtype=np.float32),
        }
        
        # Add images
        for cam_name, img in data["obs_data"]["images"].items():
            # Transpose from (H, W, C) to (C, H, W)
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.transpose(2, 0, 1)
            frame[f"observation.images.{cam_name}"] = img
        
        # Add depth only for cameras that have depth enabled
        for cam_name, depth in data["obs_data"]["depth"].items():
            # Check if this camera has depth enabled
            cam_cfg = self.cfg.camera.cameras.get(cam_name, {})
            if cam_cfg.get("depth", False):
                # Ensure depth has channel dimension (1, H, W)
                if depth.ndim == 2:
                    depth = depth[np.newaxis, ...]
                frame[f"observation.depth.{cam_name}"] = np.array(depth, dtype=np.float32)
        
        # Add pointcloud - combine all camera pointclouds that have it enabled
        if "pointcloud" in data["obs_data"] and data["obs_data"]["pointcloud"]:
            all_pointclouds = []
            for cam_name, pc in data["obs_data"]["pointcloud"].items():
                # Check if this camera has pointcloud enabled
                cam_cfg = self.cfg.camera.cameras.get(cam_name, {})
                if cam_cfg.get("pointcloud", False) and pc is not None and len(pc) > 0:
                    all_pointclouds.append(pc)
            
            if all_pointclouds:
                combined_pc = np.concatenate(all_pointclouds, axis=0)
                frame["observation.pointcloud"] = np.array(combined_pc, dtype=np.float32)
            elif "observation.pointcloud" in self.features:
                # If no pointclouds available but feature is defined, create zeros
                total_points = self.features["observation.pointcloud"]["shape"][0]
                frame["observation.pointcloud"] = np.zeros((total_points, 6), dtype=np.float32)
        elif "observation.pointcloud" in self.features:
            # If pointcloud feature exists but no data, create zeros
            total_points = self.features["observation.pointcloud"]["shape"][0]
            frame["observation.pointcloud"] = np.zeros((total_points, 6), dtype=np.float32)
        
        frame["task"] = self.task 
        
        self.dataset.add_frame(frame)
        
    def save(self):
        """Saves the current episode and triggers the proxy sync."""
        # LeRobot writes the video to the active_path
        self.dataset.save_episode()
        
        # Proxy handles moving the video to physical disk and clearing RAM
        self.storage_proxy.post_save_sync()
        
    
    def close(self):
        """Finalizes the dataset and cleans up RAMDisk."""
        self.dataset.finalize()
        self.storage_proxy.finalize()
        
        if self.cfg.push_to_hub:
            self.dataset.push_to_hub(tags=["automoma"], private=True)
    

class HDF5DatasetWrapper(BaseDatasetWrapper):
    pass

class ZarrDatasetWrapper(BaseDatasetWrapper):
    pass
    
class ImageDatasetWrapper(BaseDatasetWrapper):
    pass