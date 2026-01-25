# from scene_synthesizer import URDFAsset
from yourdfpy import URDF
import numpy as np


class ObjectDescription:
    scale: float | list[float] = 1.0
    def __init__(self, urdf_path: str, scale: float | list[float] = 1.0, asset_type: str = None, asset_id: str = None):

        self.urdf_path = urdf_path
        self.asset_type = asset_type
        self.asset_id = asset_id
        # self.urdf_asset = URDF(urdf_path)
        self.scale = scale

    def set_scale(self, scale_factor: float | list[float]):
        self.scale = scale_factor

    def set_pose(self, pose: list[float]):
        self.pose = pose
        
    def set_handle_link(self, handle_link: str):
        self.handle_link = handle_link