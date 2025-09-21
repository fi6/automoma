import os
from pxr import Usd

def deactivate_prims_by_name_pattern(usd_file_path):
    """
    Load a USD file, deactivate prims whose path includes 'exterior', 'ceiling', or 'Ceiling',
    and save the file back to the same path.
    """
    try:
        # Load the USD stage
        stage = Usd.Stage.Open(usd_file_path)

        deactivated_count = 0
        # Traverse all prims in the stage
        for prim in stage.Traverse():
            prim_path_str = str(prim.GetPath())
            # Check if the prim path contains target keywords
            # 'ceiling' for ceiling, 'Ceiling' for the light
            if "exterior" in prim_path_str or "ceiling" in prim_path_str or "Ceiling" in prim_path_str:
                prim.SetActive(False)
                print(f"  ➤ Deactivated: {prim_path_str}")
                deactivated_count += 1

        # Save the stage back to the original file path
        stage.Save()
        print(f"✅ Saved changes to: {usd_file_path} (Deactivated {deactivated_count} prims)")

    except Exception as e:
        print(f"❌ Error processing {usd_file_path}: {e}")


def process_all_scenes(base_dir):
    """
    Find all scene_X_seed_X directories, locate export_scene.usdc, and process them in sorted order.
    """
    # Get all immediate subdirectories matching scene_* pattern
    scene_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("scene_")
    ]

    # Sort naturally: scene_1_seed_1, scene_2_seed_2, ..., scene_10_seed_10
    scene_dirs.sort(key=lambda x: [
        int(c) if c.isdigit() else c
        for c in __import__('re').split(r'(\d+)', x)
    ])

    total_processed = 0
    total_errors = 0

    print(f"🔍 Found {len(scene_dirs)} scene directories. Processing in order...\n")

    for scene_name in scene_dirs:
        usd_path = os.path.join(
            base_dir, scene_name, "export", "export_scene.blend", "export_scene.usdc"
        )

        print(f"📁 Processing: {scene_name}")
        if os.path.exists(usd_path):
            deactivate_prims_by_name_pattern(usd_path)
            total_processed += 1
        else:
            print(f"⚠️  USD file not found: {usd_path}")
            total_errors += 1
        print("-" * 80)

    print("\n" + "="*80)
    print("📊 PROCESSING SUMMARY")
    print("="*80)
    print(f"✅ Successfully processed: {total_processed} scenes")
    print(f"❌ Errors / Missing files: {total_errors} scenes")
    print(f"📊 Total scenes attempted: {len(scene_dirs)}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    BASE_DIR = "/home/xinhai/Documents/automoma/output/test/kitchen_0919"

    if not os.path.exists(BASE_DIR):
        print(f"❌ Base directory does not exist: {BASE_DIR}")
        exit(1)

    process_all_scenes(BASE_DIR)