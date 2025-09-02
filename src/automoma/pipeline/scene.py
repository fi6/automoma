from automoma.models.object import ObjectDescription
from automoma.models.scene import SceneDescription

from typing import List
import json
import time
from automoma.utils.config import *
class ScenePipeline:
    def __init__(self):
        raise NotImplementedError("ScenePipeline is not implemented yet.")
    
    
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
        
        scene, updated_objects = self.generate_scene_infinigen(objects, seed)

        return scene, updated_objects
