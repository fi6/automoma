from automoma.models.object import ObjectDescription
from automoma.models.scene import SceneDescription

from typing import List
import json
import time
from automoma.utils.config import *


class ScenePipeline:
    def __init__(self):
        pass

    def generate_scene(self, objects: list[ObjectDescription], seed: int) -> SceneDescription:
        raise NotImplementedError("ScenePipeline is not implemented yet.")


class InfinigenScenePipeline(ScenePipeline):
    def __init__(self, version: str = SCENE_GENERATION_VERSION):
        self.version = version

    def create_requirement_file(self, objects: list[ObjectDescription], requirement_path: str):
        # create requirement.json for scene generation
        requirement = {"static_objects": []}
        for obj in objects:
            obj_info = {
                "asset_type": obj.asset_type,
                "asset_id": obj.asset_id,
                "urdf_path": obj.urdf_path,
                "scale": obj.scale,
            }
            requirement["static_objects"].append(obj_info)

        with open(requirement_path, "w") as f:
            json.dump(requirement, f, indent=4)

    def generate_scene_infinigen(self, objects: list[ObjectDescription], seed: int) -> SceneDescription:
        # generate scene with given asset, or replace a object with the given asset

        # version_seed_timestamp
        scene_dir = abs_path(f"{SCENE_OUTPUT_DIR}/{SCENE_GENERATION_VERSION}_seed{seed}_{int(time.time())}")
        blender_dir = make_dir(f"{scene_dir}/blender")
        export_dir = make_dir(f"{scene_dir}/export")
        info_dir = make_dir(f"{scene_dir}/info")

        # create requirement.json under info_dir for object category and urdf path

        requirement_path = f"{info_dir}/requirement.json"
        metadata_path = f"{info_dir}/metadata.json"

        usd_path = f"{export_dir}/export_scene.blend/export_scene.usdc"

        # check if the usd file exists
        if not os.path.exists(usd_path):
            raise FileNotFoundError(f"USD file not found at {usd_path}, scene generation failed")

        # create scene description
        scene = SceneDescription(scene_usd_path=usd_path, metadata_path=metadata_path)

        return scene

    def generate_scene(self, objects: list[ObjectDescription], seed: int) -> SceneDescription:
        # generate scene with given asset, or replace a object with the given asset

        return self.generate_scene_infinigen(objects, seed)


if __name__ == "__main__":
    objects = [
        ObjectDescription(
            asset_type="Microwave", asset_id="7221", scale=0.4, urdf_path="assets/object/Microwave/7221/mobility.urdf"
        ),
        ObjectDescription(
            asset_type="Dishwasher",
            asset_id="11622",
            scale=0.6,
            urdf_path="assets/object/Dishwasher/11622/mobility.urdf",
        ),
    ]
    scene = InfinigenScenePipeline()
    scene.generate_scene(objects, seed=0)
