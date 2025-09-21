import os
import json
import shutil

def analyze_scenes(base_directory):
    """
    Analyzes all scene directories under the base_directory.
    For each scene, checks if requirement objects are in metadata and compiles results & stats.
    Scenes are processed in sorted order.
    A scene is "valid" if ALL its requirement objects are satisfied.
    """
    results = []  # List to hold results for each scene
    valid_scene_names = []  # List of scenes that meet ALL requirements
    total_scenes = 0
    total_valid_objects = 0
    total_requirement_objects = 0

    # Get all immediate subdirectories (scene_X_seed_X) under base_directory
    scene_dirs = [
        d for d in os.listdir(base_directory)
        if os.path.isdir(os.path.join(base_directory, d)) and d.startswith("scene_")
    ]
    # Sort scene directories by name to ensure consistent order
    scene_dirs.sort()

    for scene_name in scene_dirs:
        scene_path = os.path.join(base_directory, scene_name)
        info_dir = os.path.join(scene_path, "info")
        metadata_path = os.path.join(info_dir, "metadata.json")
        requirement_path = os.path.join(info_dir, "requirement.json")

        if not (os.path.exists(metadata_path) and os.path.exists(requirement_path)):
            print(f"⚠️  Skipping {scene_name}: Missing metadata.json or requirement.json")
            # TODO: hack handle missing files more gracefully
            if not os.path.exists(requirement_path):
                print(f"   - Missing requirement.json")
                shutil.copy(
                    "/home/xinhai/Documents/automoma/tests/assets/requirement.json",
                    requirement_path
                )
            else:
                continue

        total_scenes += 1
        print(f"Processing scene: {scene_name}")

        # Load metadata and requirement
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        with open(requirement_path, "r", encoding="utf-8") as f:
            requirement = json.load(f)

        # Get set of generated asset IDs from metadata
        generated_ids = {obj["asset_id"] for obj in metadata.get("static_objects", {}).values()}

        # Check validity for each object in requirement
        requirement_objects = requirement.get("static_objects", [])
        valid_objects_in_scene = []
        all_valid = True  # Assume valid until proven otherwise

        for obj in requirement_objects:
            asset_id = obj.get("asset_id")
            is_valid = asset_id in generated_ids
            obj["valid"] = is_valid  # Mark validity in the object dict
            if is_valid:
                valid_objects_in_scene.append(obj)
                total_valid_objects += 1
            else:
                all_valid = False  # At least one missing → scene invalid
            total_requirement_objects += 1

        # If all requirement objects are valid, add scene to valid_scene_names
        if all_valid and len(requirement_objects) > 0:
            valid_scene_names.append(scene_name)

        # Compile result for this scene
        scene_result = {
            "scene_path": scene_path,
            "scene_name": scene_name,
            "valid_objects": valid_objects_in_scene,  # List of valid objects (dicts)
            "invalid_objects": [
                obj for obj in requirement_objects if not obj.get("valid", False)
            ],
            "total_requirement_objects": len(requirement_objects),
            "total_valid_objects": len(valid_objects_in_scene),
            "is_fully_valid": all_valid and len(requirement_objects) > 0
        }
        # results.append(scene_result) # TODO: now no need to save results

    # Compile overall statistics
    statistics = {
        "total_scenes_processed": total_scenes,
        "total_requirement_objects_across_all_scenes": total_requirement_objects,
        "total_valid_objects_across_all_scenes": total_valid_objects,
        "overall_validity_rate": (total_valid_objects / total_requirement_objects) if total_requirement_objects > 0 else 0.0,
        "fully_valid_scenes_count": len(valid_scene_names),
        "fully_valid_scenes_list": valid_scene_names  # ✅ List of scenes meeting ALL requirements
    }

    return results, statistics

# --- Main Execution ---
if __name__ == "__main__":
    base_dir = "/home/xinhai/Documents/automoma/output/test/kitchen_0919"

    if not os.path.exists(base_dir):
        print(f"Error: Base directory does not exist: {base_dir}")
        exit(1)

    results, stats = analyze_scenes(base_dir)

    # Print Statistics
    print("\n" + "="*60)
    print("📊 OVERALL STATISTICS")
    print("="*60)
    for key, value in stats.items():
        if key != "fully_valid_scenes_list":
            print(f"{key}: {value}")
    print("\n✅ Fully Valid Scenes (All Requirements Met):")
    for scene in stats["fully_valid_scenes_list"]:
        print(f"  - {scene}")

    # Save detailed results to JSON
    output_file = os.path.join(base_dir, "scene_analysis_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "scene_results": results,  # ✅ All scenes in sorted order
            "statistics": stats
        }, f, indent=4, ensure_ascii=False)

    print(f"\n📁 Detailed results saved to: {output_file}")

    # Print summary per scene
    print("\n" + "="*60)
    print("📋 SCENE SUMMARY (In Order)")
    print("="*60)
    for result in results:
        status = "✅ VALID" if result["is_fully_valid"] else "❌ INVALID"
        print(f"{result['scene_name']} → {status} ({result['total_valid_objects']}/{result['total_requirement_objects']} objects met)")