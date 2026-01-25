import json
import numpy as np
from automoma.models.object import ObjectDescription
from automoma.utils.transform import matrix_to_pose, pose_to_matrix


class SceneDescription:
    def __init__(self, scene_usd_path: str, metadata_path: str):
        self.scene_usd_path = scene_usd_path
        self.metadata_path = metadata_path
        self.metadata = json.load(open(metadata_path, "r"))
        self.pose = [0, 0, 0, 1, 0, 0, 0]  # default identity

    def get_object_info(self, object: ObjectDescription) -> dict:
        # find the object info in the metadata
        static_objects = self.metadata.get("static_objects", {})
        for _, value in static_objects.items():
            if value.get("asset_type") == object.asset_type and value.get("asset_id") == object.asset_id:
                return value
        else:
            raise ValueError(f"Object {object.asset_type} {object.asset_id} not found in scene metadata")
        
    def get_object_matrix(self, object: ObjectDescription) -> np.ndarray:
        object_info = self.get_object_info(object)
        object_matrix = np.array(object_info["matrix"])
        return object_matrix
    def get_object_pose(self, object: ObjectDescription) -> np.ndarray:
        object_pose = matrix_to_pose(self.get_object_matrix(object))
        return object_pose

    def get_object_scale(self, object: ObjectDescription) -> float | list[float]:
        object_info = self.get_object_info(object)
        return object_info.get("scale", [1.0, 1.0, 1.0])

    def get_object_bounding_box(self, object: ObjectDescription) -> np.ndarray:
        object_info = self.get_object_info(object)
        return np.array(object_info.get("bbox_corners", None))

    def set_pose(self, pose: list[float]):
        self.pose = pose