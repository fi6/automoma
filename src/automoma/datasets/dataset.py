import numpy as np
import os

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

class LeRobotDatasetWrapper(BaseDatasetWrapper):
    def __init__(self, cfg):
        
        self.dataset = None
        self.cfg = cfg
        self.task = cfg.get("task", "manipulation")
        
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
        
        # Check if dataset already exists and remove it to avoid FileExistsError
        dataset_path = Path(self.cfg.root) / self.cfg.repo_id
        os.makedirs(dataset_path, exist_ok=True)
        if dataset_path.exists():
            print(f"Removing existing dataset at {dataset_path}")
            shutil.rmtree(dataset_path)
        
        print(f"Creating dataset at {dataset_path}")
        
        # create dataset
        self.dataset = LeRobotDataset.create(
            repo_id = self.cfg.repo_id,
            root = dataset_path,
            fps = self.cfg.fps,
            features= self.features,
            robot_type= self.cfg.robot_type,
            use_videos= self.cfg.use_videos,
        )
        print(f"Created dataset at {dataset_path}")
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
        self.dataset.save_episode()
    
    def close(self):
        self.dataset.finalize()
        # push to hub if required
        if self.cfg.push_to_hub:
            self.dataset.push_to_hub(tags=["automoma"], private=True)
    

class HDF5DatasetWrapper(BaseDatasetWrapper):
    pass

class ZarrDatasetWrapper(BaseDatasetWrapper):
    pass
    
class ImageDatasetWrapper(BaseDatasetWrapper):
    pass