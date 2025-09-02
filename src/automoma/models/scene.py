import json
from automoma.models.object import ObjectDescription

class SceneDescription:
    def __init__(self, scene_usd_path: str, metadata_path: str):
        self.scene_usd_path = scene_usd_path
        self.metadata = json.load(open(metadata_path, 'r'))

    def get_object_info(self, object: ObjectDescription) -> dict:
        # find the object info in the metadata
        static_objects = self.metadata.get("static_objects", {})
        for _, value in static_objects.items():
            if value.get("asset_type") == object.asset_type and value.get("asset_id") == object.asset_id:
                return value
        else:
            raise ValueError(f"Object {object.asset_type} {object.asset_id} not found in scene metadata")
        
        
        