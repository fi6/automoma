# automoma/pipeline/scene_pipeline.py

"""
Scene generation pipeline using Infinigen.
Generates 3D scenes with specified objects via subprocess calls.
"""

from typing import List, Tuple
import json
import os
import sys

sys.path.append("/home/xinhai/automoma/third_party/infinigen")
import shutil
import time
import subprocess
from pathlib import Path
from dataclasses import dataclass

from automoma.models.object import ObjectDescription
from automoma.models.scene import SceneDescription
from automoma.utils.config import (
    SCENE_OUTPUT_DIR,
    SCENE_GENERATION_VERSION,
    SCENE_GENERATION_REPO,
    abs_path,
    make_dir,
)
from automoma.utils.file import save_json


# ----------------------------
# Constants
# ----------------------------
REQUIREMENT_FILE_NAME = "requirement.json"
METADATA_FILE_NAME = "metadata.json"
USD_EXPORT_PATH = "export_scene.blend/export_scene.usdc"
TEXTURE_RESOLUTION = "1024"
DEFAULT_CONFIG = "kitchen_only.gin"  # Change as needed

@dataclass
class SceneGenerationResult:
    """Result of a successful scene generation."""
    scene: SceneDescription
    valid_objects: List[ObjectDescription]


class ScenePipeline:
    def __init__(self, version: str = SCENE_GENERATION_VERSION):
        self.version = version
    def generate_scene(self, objects: List[ObjectDescription], seed: int) -> SceneGenerationResult:
        raise NotImplementedError("This method should be implemented by subclasses.")

class InfinigenScenePipeline(ScenePipeline):
    """
    Pipeline for generating 3D scenes with specified objects using Infinigen.

    Steps:
        1. Write requirement.json
        2. Run Infinigen scene generation
        3. Export to USD
        4. Extract object metadata
        5. Validate and return SceneDescription
    """

    def __init__(self, version: str = SCENE_GENERATION_VERSION):
        super().__init__(version)

    def _create_requirement_data(self, objects: List[ObjectDescription]) -> dict:
        """Build requirement dict from object list."""
        return {
            "static_objects": [
                {
                    "asset_type": obj.asset_type,
                    "asset_id": obj.asset_id,
                    "urdf_path": obj.urdf_path,
                    "scale": obj.scale,
                }
                for obj in objects
            ]
        }

    def _copy_requirement_to_asset_dir(self, requirement_path: str) -> None:
        """Copy requirement.json to Infinigen's static asset directory."""
        static_asset_dir = abs_path(os.path.join(SCENE_GENERATION_REPO, "infinigen/assets/static_assets"))
        os.makedirs(static_asset_dir, exist_ok=True)
        dst_path = os.path.join(static_asset_dir, REQUIREMENT_FILE_NAME)
        shutil.copy2(requirement_path, dst_path)
        print(f"📋 Requirement copied to: {dst_path}")

    def _create_directories(self, base_dir: str) -> Tuple[str, str, str]:
        """Create blender/, export/, info/ directories."""
        blender_dir = make_dir(os.path.join(base_dir, "blender"))
        export_dir = make_dir(os.path.join(base_dir, "export"))
        info_dir = make_dir(os.path.join(base_dir, "info"))
        return blender_dir, export_dir, info_dir

    def _validate_usd_file(self, usd_path: str) -> None:
        """Ensure USD file exists."""
        if not os.path.exists(usd_path):
            raise FileNotFoundError(f"USD file not found: {usd_path}. Export may have failed.")

    def _check_generation_validity(
        self,
        objects: List[ObjectDescription],
        metadata_path: str,
        requirement_path: str,
    ) -> Tuple[int, List[ObjectDescription]]:
        """
        Validate which objects were generated and update requirement.json.

        Returns:
            Count of valid objects and list of valid ObjectDescription.
        """
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        with open(requirement_path, "r", encoding="utf-8") as f:
            requirement = json.load(f)

        generated_ids = {obj["asset_id"] for obj in metadata.get("static_objects", {}).values()}
        valid_objects = []

        for obj in requirement["static_objects"]:
            obj["valid"] = obj["asset_id"] in generated_ids

        for obj_desc in objects:
            if obj_desc.asset_id in generated_ids:
                valid_objects.append(obj_desc)

        save_json(requirement, requirement_path)

        num_valid = len(valid_objects)
        if num_valid == 0:
            raise ValueError("No valid objects found in the generated scene. Try again.")

        print(f"✅ {num_valid}/{len(objects)} objects successfully generated.")
        return num_valid, valid_objects

    def _run_infinigen_generation(
        self,
        seed: int,
        scene_dir: str,
        blender_dir: str,
        export_dir: str,
        info_dir: str,
        requirement_path: str,
    ) -> None:
        """
        Execute full Infinigen pipeline: generate -> export -> extract.
        """
        infinigen_root = abs_path(SCENE_GENERATION_REPO)
        log_dir = make_dir(os.path.join(scene_dir, "logs"))
        # Using conda run (simplest and most reliable)
        def make_conda_command(args: list) -> list:
            # use the interpreter that belongs to the 'infinigen' env
            return args

        # 1. Generate Scene
        print(f"🌱 Generating scene with seed={seed}...")
        generate_cmd = [
            sys.executable,
            "-m", "infinigen_examples.generate_indoors",
            "--seed", str(seed),
            "--task", "coarse",
            "--output_folder", blender_dir,
            "--configs", DEFAULT_CONFIG,
        ]
        log_file_1 = os.path.join(log_dir, f"generate_scene_seed_{seed}.log")
        print(f"📝 Logging to: {log_file_1}")
        with open(log_file_1, "w") as log_f:
            result1 = subprocess.run(make_conda_command(generate_cmd), cwd=infinigen_root, stdout=log_f, stderr=subprocess.STDOUT)
        if not os.path.exists(os.path.join(blender_dir, "scene.blend")):
            raise RuntimeError(f"❌ Scene generation failed. See log: {log_file_1}")

        # 2. Export to USD
        print("📦 Exporting scene to USD...")
        export_cmd = [
            sys.executable,
            "-m", "infinigen.tools.export",
            "--input_folder", blender_dir,
            "--output_folder", export_dir,
            "--format", "usdc",
            "--resolution", TEXTURE_RESOLUTION,
        ]
        log_file_2 = os.path.join(log_dir, f"export_scene_seed_{seed}.log")
        print(f"📝 Logging to: {log_file_2}")
        with open(log_file_2, "w") as log_f:
            result2 = subprocess.run(make_conda_command(export_cmd), cwd=infinigen_root, stdout=log_f, stderr=subprocess.STDOUT)
        if not os.path.exists(os.path.join(export_dir, USD_EXPORT_PATH)):
            raise RuntimeError(f"❌ USD export failed. See log: {log_file_2}")

        # 3. Extract Object Info
        print("🔍 Extracting object metadata...")
        blend_files = list(Path(blender_dir).rglob("*.blend"))
        if not blend_files:
            raise FileNotFoundError("No .blend file found for metadata extraction.")
        blend_file = str(blend_files[0])

        metadata_file = os.path.join(info_dir, METADATA_FILE_NAME)
        extract_cmd = [
            sys.executable,
            "third_party/infinigen/scripts/automoma/extract_object_info.py",
            "--blend_file", blend_file,
            "--output", metadata_file,
        ]
        log_file_3 = os.path.join(log_dir, f"extract_objects_seed_{seed}.log")
        print(f"📝 Logging to: {log_file_3}")
        with open(log_file_3, "w") as log_f:
            result3 = subprocess.run(make_conda_command(extract_cmd), stdout=log_f, stderr=subprocess.STDOUT)
        if not os.path.exists(metadata_file):
            raise RuntimeError(f"❌ Object extraction failed. See log: {log_file_3}")

        print("✅ All Infinigen steps completed.")

    def generate_scene(self, objects: List[ObjectDescription], seed: int) -> SceneGenerationResult:
        """Generate scene using Infinigen backend."""
        timestamp = int(time.time())
        scene_dir = abs_path(f"{SCENE_OUTPUT_DIR}/{self.version}_seed{seed}_{timestamp}")
        blender_dir, export_dir, info_dir = self._create_directories(scene_dir)

        # 1. Prepare requirement.json
        requirement_path = os.path.join(info_dir, REQUIREMENT_FILE_NAME)
        requirement_data = self._create_requirement_data(objects)
        save_json(requirement_data, requirement_path)
        self._copy_requirement_to_asset_dir(requirement_path)

        # 2. Run full Infinigen pipeline
        self._run_infinigen_generation(seed, scene_dir, blender_dir, export_dir, info_dir, requirement_path)

        # 3. Validate outputs
        usd_path = os.path.join(export_dir, USD_EXPORT_PATH)
        metadata_path = os.path.join(info_dir, METADATA_FILE_NAME)
        self._validate_usd_file(usd_path)
        num_valid, valid_objects = self._check_generation_validity(objects, metadata_path, requirement_path)

        # 4. Create SceneDescription
        scene = SceneDescription(scene_usd_path=usd_path, metadata_path=metadata_path)

        print(f"🎨 Scene generated at: {scene_dir}")
        return SceneGenerationResult(scene=scene, valid_objects=valid_objects)
    
    def load_scene(self, scene_dir: str, objects: List[ObjectDescription]) -> SceneGenerationResult:
        """Load existing scene from directory."""
        scene_dir = abs_path(scene_dir)
        export_dir = os.path.join(scene_dir, "export")
        info_dir = os.path.join(scene_dir, "info")
        usd_path = os.path.join(export_dir, USD_EXPORT_PATH)
        metadata_path = os.path.join(info_dir, METADATA_FILE_NAME)
        self._validate_usd_file(usd_path)
        num_valid, valid_objects = self._check_generation_validity(objects, metadata_path, os.path.join(info_dir, REQUIREMENT_FILE_NAME))
        scene = SceneDescription(scene_usd_path=usd_path, metadata_path=metadata_path)
        print(f"🎨 Scene loaded from: {scene_dir}")
        return SceneGenerationResult(scene=scene, valid_objects=valid_objects)


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    # Example object list
    test_objects = [
        ObjectDescription(
            asset_type="Microwave",
            asset_id="7221",
            scale=0.3563,
            urdf_path="assets/object/Microwave/7221/mobility.urdf",
        ),
        ObjectDescription(
            asset_type="Dishwasher",
            asset_id="11622",
            scale=0.6446,
            urdf_path="assets/object/Dishwasher/11622/mobility.urdf",
        ),
    ]

    # Run pipeline
    pipeline = InfinigenScenePipeline(version="v1")
    result = pipeline.generate_scene(test_objects, seed=101)
    # result = pipeline.load_scene("/home/xinhai/Documents/automoma/third_party/infinigen/output/kitchen/v1_seed100_1756884596", test_objects)
    print(f"Scene USD Path: {result.scene.scene_usd_path}")