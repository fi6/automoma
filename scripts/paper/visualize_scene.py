"""
Script to visualize and capture images from Infinigen scenes.
Loads each scene, sets up camera, and saves RGB images.
"""

import isaacsim
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width": 1920,
    "height": 1280
})

import os
import re
import numpy as np
import torch
from pathlib import Path
import omni.kit.actions.core
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.torch.rotations as rot_utils


def setup_lighting():
    """Setup lighting similar to example_replay.py"""
    action_registry = omni.kit.actions.core.get_action_registry()
    action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_rig")
    action.execute(lighting_mode=2)


def deactivate_prims(stage, prim_patterns):
    """Deactivate prims matching given patterns."""
    for prim in stage.Traverse():
        prim_name = prim.GetName()
        for pattern in prim_patterns:
            if pattern in prim_name:
                prim.SetActive(False)
                print(f"Deactivated: {prim.GetPath()}")
                break


def create_camera_at_default_prim(world, translate, rotate, focal_length, focus_distance, resolution=(1920, 1280)):
    """
    Create a camera under the default prim with specified transform.
    Uses Isaac Sim Camera class for proper setup.
    
    Args:
        world: Isaac Sim World instance
        translate: (x, y, z) position
        rotate: (rx, ry, rz) rotation in degrees
        focal_length: Camera focal length in mm
        focus_distance: Camera focus distance
        resolution: (width, height) tuple
    
    Returns:
        Camera instance
    """
    camera_path = "/World/visualization_camera"
    
    # Create camera using Isaac Sim Camera class
    camera = Camera(
        prim_path=camera_path,
        frequency=20,
        resolution=resolution,
    )
    
    camera.initialize()
    
    # Convert Euler angles (degrees) to quaternion using Isaac Sim's rotation utilities
    # This matches the _get_transform method in replayer.py
    rot_quat = rot_utils.euler_angles_to_quats(
        torch.tensor([rotate], dtype=torch.float32), 
        degrees=True, 
        extrinsic=False
    )[0].tolist()
    
    # Set camera world pose
    # rot_quat format is [w, x, y, z]
    camera.set_world_pose(
        position=np.array(translate),
        orientation=np.array(rot_quat),  # (w, x, y, z) format
        camera_axes="usd"
    )
    
    # Set focal length
    camera.set_focal_length(focal_length)
    
    # Note: focus_distance is set via USD attributes if needed
    # but typically not necessary for rendered output
    
    print(f"Created camera at {camera_path}")
    print(f"  Position: {translate}")
    print(f"  Rotation: {rotate} degrees")
    print(f"  Focal Length: {focal_length}mm")
    print(f"  Resolution: {resolution}")
    
    return camera


def capture_image(camera, output_path, world):
    """
    Capture RGB image from camera and save to file.
    
    Args:
        camera: Isaac Sim Camera instance
        output_path: Path to save image
        world: World instance for stepping simulation
    """
    
    # Get RGB data
    rgb_data = camera.get_rgba()
    
    if rgb_data is not None:
        # Convert to uint8 and remove alpha channel
        rgb_image = (rgb_data[:, :, :3] * 255).astype(np.uint8)
        
        # Save image using PIL
        from PIL import Image
        img = Image.fromarray(rgb_image)
        img.save(output_path)
        print(f"Saved image to {output_path}")
    else:
        print(f"Warning: Failed to capture image from camera")


def visualize_scene(scene_path, output_dir, camera_config):
    """
    Load a scene, setup camera, and capture image.
    
    Args:
        scene_path: Path to scene directory
        output_dir: Directory to save output images
        camera_config: Dict with camera parameters
    """
    scene_name = os.path.basename(scene_path)
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene_name}")
    print(f"{'='*60}")
    
    # Initialize world
    world = World(stage_units_in_meters=1.0)
    stage = world.stage
    
    # Set default prim
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    world.clear()
    
    # Load scene USD
    usd_path = os.path.join(scene_path, "export/export_scene.blend/export_scene.usdc")
    if not os.path.exists(usd_path):
        print(f"Error: USD file not found at {usd_path}")
        return
    
    print(f"Loading USD: {usd_path}")
    add_reference_to_stage(usd_path=usd_path, prim_path="/World/scene")
    
    # Setup lighting
    setup_lighting()
    
    # Deactivate specific prims (similar to example_replay.py)
    # Adjust these patterns based on what you want to hide
    deactivate_patterns = [
        "exterior",  # walls
        "ceiling",   # ceiling geometry
        "Ceiling",   # ceiling lights
    ]
    deactivate_prims(stage, deactivate_patterns)
    
    # Initialize physics and reset
    world.initialize_physics()
    world.reset()
    
    # Create camera under default prim
    camera = create_camera_at_default_prim(
        world,
        translate=camera_config["translate"],
        rotate=camera_config["rotate"],
        focal_length=camera_config["focal_length"],
        focus_distance=camera_config["focus_distance"],
        resolution=(1920, 1280)
    )
    
    # Step simulation a few times to ensure everything is loaded
    for _ in range(500):
        world.step(render=True)
    
    if "scene_0" in scene_name:
        for _ in range(1500):
            world.step(render=True)
    
    # Capture image
    # os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(output_dir, f"{scene_name}.png")
    # capture_image(camera, output_path, world)
    
    # Clear world for next scene
    world.clear()
    world.stop()


def main():
    """Main function to process all scenes."""
    # Configuration
    scenes_base_path = "/home/xinhai/Documents/automoma/assets/scene/infinigen/kitchen_1130"
    output_base_dir = "/home/xinhai/Documents/automoma/output/paper/scene_visualization"
    
    camera_config = {
        "translate": (18, 18, 20),
        "rotate": (45, 0, 135),  # Euler angles in degrees
        "focal_length": 2,  # mm
        "focus_distance": 400
    }
    
    # Find all scene directories
    def natural_sort_key(path):
        # Extract numbers from the filename for proper numerical sorting
        filename = os.path.basename(path)
        # Find all numbers in the string and convert them to integers
        numbers = [int(match) for match in re.findall(r'\d+', filename)]
        return numbers

    scene_dirs = sorted([
        os.path.join(scenes_base_path, d)
        for d in os.listdir(scenes_base_path)
        if os.path.isdir(os.path.join(scenes_base_path, d)) and d.startswith("scene_")
    ], key=natural_sort_key)
    
    # TODO: add filtering if needed
    # scene_dirs = [i for i in scene_dirs if ('scene_8_seed_8' in i)]
    
    print(f"Found {len(scene_dirs)} scenes to process")
    
    # Process each scene
    for scene_path in scene_dirs:
        try:
            visualize_scene(scene_path, output_base_dir, camera_config)
        except Exception as e:
            scene_name = os.path.basename(scene_path)
            print(f"Error processing {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Completed processing all scenes")
    print(f"Images saved to: {output_base_dir}")
    print(f"{'='*60}")
    
    # Close simulation
    simulation_app.close()


if __name__ == "__main__":
    main()
