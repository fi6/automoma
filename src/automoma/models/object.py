from scene_synthesizer import URDFAsset
import numpy as np


class ObjectDescription:
    scale: float | list[float] = 1.0
    def __init__(self, urdf_path: str, scale: float | list[float] = 1.0):
        self.urdf_asset = URDFAsset(urdf_path)
        self.scale = scale

    def set_scale(self, scale_factor: float | list[float]):
        self.scale = scale_factor
