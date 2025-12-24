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
        for cam in self.cfg.camera.names:
            features[f"observation.images.{cam}"] = {
                "dtype": "video" if self.cfg.use_videos else "image",
                "shape": (3, self.cfg.camera.height, self.cfg.camera.width),
                "names": ["channels", "height", "width"],
            }
            
        for cam in self.cfg.camera.names:
            features[f"observation.depth.{cam}"] = {
                "dtype": "float32",
                "shape": (1, self.cfg.camera.height, self.cfg.camera.width),
                "names": ["channels", "height", "width"],
            }   
        self.features = features

    def create(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        # init features
        self._init_features()
        
        # create dataset
        self.dataset = LeRobotDataset.create(
            repo_id = self.cfg.repo_id,
            root= self.cfg.root,
            fps = self.cfg.fps,
            features= self.features,
            robot_type= self.cfg.robot_type,
            use_videos= self.cfg.use_videos,
        )
    def add(self, data):
        frame = {
            "observation.state": data["joint_data"],
            "observation.eef": data["eef_pose_data"],
            "action": data["action_data"],
        }
        # Add images
        for cam_name, img in data["obs_data"]["images"].items():
            frame[f"observation.images.{cam_name}"] = img
        
        # Add depth
        for cam_name, depth in data["obs_data"]["depth"].items():
            frame[f"observation.depth.{cam_name}"] = depth
        
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