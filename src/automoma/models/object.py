from scene_synthesizer import URDFAsset
import numpy as np


class ObjectDescription:
    def __init__(self, urdf_path: str):
        self.urdf_asset = URDFAsset(urdf_path)

    def scale(self, scale_factor: float | list[float]):
        raise NotImplementedError("Scaling is not implemented yet.")
