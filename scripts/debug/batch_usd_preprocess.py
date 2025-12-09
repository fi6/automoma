import os
import sys
import json
import re
from pathlib import Path
from pxr import Sdf, Usd

# Add curobo to path if needed
try:
    from curobo.util.usd_helper import UsdHelper
except ImportError:
    print("Warning: curobo not found. Using basic USD operations only.")
    UsdHelper = None

# Try to import Omniverse kit for lighting (optional)
try:
    import omni.kit.actions.core
    OMNI_AVAILABLE = True
except ImportError:
    OMNI_AVAILABLE = False
    print("Warning: Omniverse kit not available. Skipping lighting mode changes.")

# Configuration
Z_LOWERING_AMOUNT = 0.03  # Amount to lower objects in Z axis


class USDPreprocessor:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        
    def step1_restructure_stage(self, usd_file_path):
        """
        Move all prims from /World to /World/scene, excluding materials and existing /World/scene
        """
        print("  Step 1: Restructuring stage hierarchy...")
        
        try:
            # Use UsdHelper if available, otherwise use basic USD
            if UsdHelper:
                usd_helper = UsdHelper()
                usd_helper.load_stage_from_file(str(usd_file_path))
                stage = usd_helper.stage
            else:
                stage = Usd.Stage.Open(str(usd_file_path))
            
            # Define new parent path
            new_parent_path_str = "/World/scene"
            new_parent_path = Sdf.Path(new_parent_path_str)
            
            # Check if /World/scene already exists
            if stage.GetPrimAtPath(new_parent_path):
                print("    ✅ /World/scene already exists, skipping restructure")
                return True
            
            # Create /World/scene
            stage.DefinePrim(new_parent_path, "Xform")
            
            # Get direct children of /World
            world_prim = stage.GetPrimAtPath("/World")
            if not world_prim:
                print("    ⚠️ No /World prim found")
                return False
                
            children_paths = [child.GetPath() for child in world_prim.GetChildren()]
            
            # Filter: only move prims that are NOT already under /World/scene,
            # NOT named with '_materials', and NOT /World/scene itself
            prims_to_move = [
                path for path in children_paths
                if path != new_parent_path
                and not path.HasPrefix(new_parent_path)
                and "_materials" not in path.name
                and "material" not in path.name.lower()
            ]
            
            moved_count = 0
            # Reparent each prim
            for old_path in prims_to_move:
                prim_name = old_path.name
                new_path = new_parent_path.AppendChild(prim_name)
                
                print(f"    Moving {old_path} -> {new_path}")
                
                # Copy spec to new location
                layer = stage.GetRootLayer()
                if Sdf.CopySpec(layer, old_path, layer, new_path):
                    # Remove original
                    stage.RemovePrim(old_path)
                    moved_count += 1
                else:
                    print(f"    ❌ Failed to move {old_path}")
            
            # Save changes
            stage.Save()
            print(f"    ✅ Moved {moved_count} prims under /World/scene")
            return True
            
        except Exception as e:
            print(f"    ❌ Error in step 1: {e}")
            return False
    
    def step2_deactivate_elements(self, usd_file_path):
        """
        Deactivate ceiling, exterior, and wall elements for better visualization
        """
        print("  Step 2: Deactivating ceiling/wall elements...")
        
        try:
            stage = Usd.Stage.Open(str(usd_file_path))
            deactivated_count = 0
            
            # Keywords to look for in prim paths
            deactivate_keywords = [
                "exterior", "ceiling", "Ceiling"
            ]
            
            # Traverse all prims in the stage
            for prim in stage.Traverse():
                prim_path_str = str(prim.GetPath())
                
                # Check if the prim path contains any target keywords
                should_deactivate = any(keyword in prim_path_str for keyword in deactivate_keywords)
                
                if should_deactivate and prim.IsActive():
                    prim.SetActive(False)
                    print(f"    ➤ Deactivated: {prim_path_str}")
                    deactivated_count += 1
            
            # Save the stage
            stage.Save()
            print(f"    ✅ Deactivated {deactivated_count} elements")
            return True
            
        except Exception as e:
            print(f"    ❌ Error in step 2: {e}")
            return False
    
    def step3_set_lighting_mode(self, usd_file_path):
        """
        Set lighting mode to 2 using Omniverse actions (if available)
        """
        print("  Step 3: Setting lighting mode...")
        
        if not OMNI_AVAILABLE and True:
            print("    ⚠️ Omniverse kit not available, skipping lighting mode")
            return True
        
        try:
            action_registry = omni.kit.actions.core.get_action_registry()
            action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_rig")
            
            if action:
                action.execute(lighting_mode=2)
                print("    ✅ Set lighting mode to 2")
                return True
            else:
                print("    ⚠️ Lighting action not found")
                return True
                
        except Exception as e:
            print(f"    ❌ Error in step 3: {e}")
            return True  # Non-critical error
    
    def step4_lower_kitchen_space_objects(self, scene_dir, usd_file_path):
        """
        Lower objects that are on KitchenSpaceFactory by Z_LOWERING_AMOUNT
        """
        print("  Step 4: Lowering objects on KitchenSpaceFactory...")
        
        try:
            # Load solve_state.json
            solve_state_path = scene_dir / "export" / "solve_state.json"
            if not solve_state_path.exists():
                print(f"    ⚠️ solve_state.json not found: {solve_state_path}")
                return True
            
            with open(solve_state_path, 'r') as f:
                solve_state = json.load(f)
            
            # Find objects on KitchenSpaceFactory
            objects_to_lower = []
            seen_obj_ids = set()  # Track which object IDs we've already found
            for obj_id, obj_data in solve_state.get("objs", {}).items():
                if isinstance(obj_data, dict) and "relations" in obj_data:
                    for relation in obj_data.get("relations", []):
                        if "KitchenSpaceFactory" in relation.get("target_name", ""):
                            obj_str = obj_data.get("obj", "")
                            if obj_str and obj_id not in seen_obj_ids:  # Only add if not already seen
                                objects_to_lower.append((obj_id, obj_str, obj_data))
                                seen_obj_ids.add(obj_id)
                                print(f"    Found object on KitchenSpaceFactory: {obj_id} -> {obj_str}")
            
            if not objects_to_lower:
                print("    ℹ️  No objects found on KitchenSpaceFactory")
                return True
            
            # Load USD stage
            stage = Usd.Stage.Open(str(usd_file_path))
            
            # Lower each object in USD
            lowered_count = 0
            usd_search_attempts = []
            lowered_prims = set()  # Track which prims we've already lowered
            
            for obj_id, obj_str, obj_data in objects_to_lower:
                found_in_usd = False
                
                # Build specific search patterns based on object type
                search_patterns = []
                
                if "StaticCategoryFactory" in obj_str:
                    # For StaticCategoryFactory, extract type and asset_id
                    # Format: "StaticCategoryFactory(Microwave_7221_9421643_mobility)"
                    type_match = re.search(r'StaticCategoryFactory\((\w+)_(\d+)', obj_str)
                    if type_match:
                        obj_type = type_match.group(1)  # e.g., "Microwave"
                        asset_id = type_match.group(2)   # e.g., "7221"
                        # Search for specific StaticCategoryFactory with type and asset_id
                        search_patterns.append(f"StaticCategoryFactory_{obj_type}_{asset_id}")
                else:
                    # For other factories, extract all numbers from obj_str for matching
                    # Format: "FactoryName(number).spawn_asset(spawn_number)"
                    matches = re.findall(r'(\w+)\((\d+)\)\.spawn_asset\((\d+)\)', obj_str)
                    if matches:
                        factory_name, factory_num, spawn_num = matches[0]
                        # Search with spawn_asset number for precise matching
                        search_patterns.append(spawn_num)
                        search_patterns.append(factory_num)
                    else:
                        # Fallback: try to extract just factory number
                        matches = re.findall(r'(\w+)\((\d+)', obj_str)
                        if matches:
                            factory_name, factory_num = matches[0]
                            search_patterns.append(factory_num)
                
                print(f"      Searching for: {search_patterns}")
                
                # Search for this pattern in USD prims
                for search_pattern in search_patterns:
                    for prim in stage.Traverse():
                        prim_path_str = str(prim.GetPath())
                        prim_id = prim.GetPath()
                        
                        # Skip if already lowered
                        if prim_id in lowered_prims:
                            continue
                        
                        # Look for prim names matching the object
                        if search_pattern in prim_path_str and prim.GetTypeName() == "Xform":
                            # Get translation
                            translate_attr = prim.GetAttribute("xformOp:translate")
                            if translate_attr:
                                current_translate = translate_attr.Get()
                                if current_translate and len(current_translate) >= 3:
                                    # Lower the z-coordinate
                                    new_translate = (
                                        current_translate[0],
                                        current_translate[1],
                                        current_translate[2] - Z_LOWERING_AMOUNT
                                    )
                                    translate_attr.Set(new_translate)
                                    print(f"      ✅ Lowered {prim_path_str}: z {current_translate[2]:.5f} -> {new_translate[2]:.5f}")
                                    lowered_count += 1
                                    lowered_prims.add(prim_id)  # Mark as lowered
                                    found_in_usd = True
                                    break
                    
                    if found_in_usd:
                        break
                
                if not found_in_usd:
                    print(f"      ⚠️  Could not find prim for {search_patterns} in USD")
                    print(f"         obj_str: {obj_str}")
                    usd_search_attempts.append((obj_id, search_patterns))
            
            # Save USD changes
            stage.Save()
            print(f"    ✅ Lowered {lowered_count} objects in USD")
            
            if usd_search_attempts:
                print(f"    ℹ️  Debug: Failed USD searches: {usd_search_attempts}")
            
            # Load and update metadata.json
            metadata_path = scene_dir / "info" / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Build a mapping of obj_id to metadata object names
                obj_id_to_metadata = {}
                for static_obj_name, static_obj_data in metadata.get("static_objects", {}).items():
                    # Try to match by looking at the object name
                    obj_id_to_metadata[static_obj_name] = static_obj_name
                
                # Update static_objects in metadata
                updated_count = 0
                updated_objects = set()  # Track which metadata objects we've updated
                
                for obj_id, obj_str, obj_data in objects_to_lower:
                    # Try to find matching metadata object
                    matched = False
                    
                    # For StaticCategoryFactory, match by specific type and asset_id
                    if "StaticCategoryFactory" in obj_str:
                        # Extract type info from obj_str like "Microwave_7221"
                        type_match = re.search(r'StaticCategoryFactory\((\w+)_(\d+)', obj_str)
                        if type_match:
                            obj_type = type_match.group(1)  # e.g., "Microwave"
                            asset_id = type_match.group(2)   # e.g., "7221"
                            
                            for static_obj_name, static_obj_data in metadata.get("static_objects", {}).items():
                                # Skip if already updated
                                if static_obj_name in updated_objects:
                                    continue
                                
                                # Match by type and asset_id
                                if obj_type in static_obj_name and asset_id in static_obj_name:
                                    if "position" in static_obj_data and len(static_obj_data["position"]) >= 3:
                                        old_z = static_obj_data["position"][2]
                                        static_obj_data["position"][2] -= Z_LOWERING_AMOUNT
                                        
                                        # Update matrix (Z-axis scale component at [2][3])
                                        if "matrix" in static_obj_data and len(static_obj_data["matrix"]) >= 3:
                                            if len(static_obj_data["matrix"][2]) >= 3:
                                                static_obj_data["matrix"][2][3] -= Z_LOWERING_AMOUNT
                                        
                                        # Update bbox_corners
                                        if "bbox_corners" in static_obj_data:
                                            for corner in static_obj_data["bbox_corners"]:
                                                if len(corner) >= 3:
                                                    corner[2] -= Z_LOWERING_AMOUNT
                                        
                                        print(f"      Updated metadata for {static_obj_name}: z {old_z:.5f} -> {static_obj_data['position'][2]:.5f}")
                                        updated_count += 1
                                        updated_objects.add(static_obj_name)  # Mark as updated
                                        matched = True
                                        break
                    else:
                        # For regular factories (BowlFactory, JarFactory, etc.)
                        # Extract spawn_asset number for precise matching
                        spawn_match = re.search(r'\.spawn_asset\((\d+)\)', obj_str)
                        if spawn_match:
                            spawn_num = spawn_match.group(1)
                            
                            for static_obj_name, static_obj_data in metadata.get("static_objects", {}).items():
                                # Skip if already updated
                                if static_obj_name in updated_objects:
                                    continue
                                
                                # Try to match by spawn number
                                if spawn_num in static_obj_name or spawn_num in str(static_obj_data):
                                    if "position" in static_obj_data and len(static_obj_data["position"]) >= 3:
                                        old_z = static_obj_data["position"][2]
                                        static_obj_data["position"][2] -= Z_LOWERING_AMOUNT
                                        
                                        # Update matrix (Z-axis scale component at [2][2])
                                        if "matrix" in static_obj_data and len(static_obj_data["matrix"]) >= 3:
                                            if len(static_obj_data["matrix"][2]) >= 3:
                                                static_obj_data["matrix"][2][2] -= Z_LOWERING_AMOUNT
                                        
                                        # Update bbox_corners
                                        if "bbox_corners" in static_obj_data:
                                            for corner in static_obj_data["bbox_corners"]:
                                                if len(corner) >= 3:
                                                    corner[2] -= Z_LOWERING_AMOUNT
                                        
                                        print(f"      Updated metadata for {static_obj_name}: z {old_z:.5f} -> {static_obj_data['position'][2]:.5f}")
                                        updated_count += 1
                                        updated_objects.add(static_obj_name)  # Mark as updated
                                        matched = True
                                        break
                        else:
                            # Fallback: try generator matching
                            if "generator" in obj_data:
                                generator = obj_data["generator"]
                                numbers = re.findall(r'\d+', generator)
                                
                                for static_obj_name, static_obj_data in metadata.get("static_objects", {}).items():
                                    # Skip if already updated
                                    if static_obj_name in updated_objects:
                                        continue
                                    
                                    # Try to match by checking if any number in generator appears in metadata name
                                    if any(num in static_obj_name for num in numbers):
                                        if "position" in static_obj_data and len(static_obj_data["position"]) >= 3:
                                            old_z = static_obj_data["position"][2]
                                            static_obj_data["position"][2] -= Z_LOWERING_AMOUNT
                                            
                                            # Update matrix (Z-axis scale component at [2][2])
                                            if "matrix" in static_obj_data and len(static_obj_data["matrix"]) >= 3:
                                                if len(static_obj_data["matrix"][2]) >= 3:
                                                    static_obj_data["matrix"][2][2] -= Z_LOWERING_AMOUNT
                                            
                                            # Update bbox_corners
                                            if "bbox_corners" in static_obj_data:
                                                for corner in static_obj_data["bbox_corners"]:
                                                    if len(corner) >= 3:
                                                        corner[2] -= Z_LOWERING_AMOUNT
                                            
                                            print(f"      Updated metadata for {static_obj_name}: z {old_z:.5f} -> {static_obj_data['position'][2]:.5f}")
                                            updated_count += 1
                                            updated_objects.add(static_obj_name)  # Mark as updated
                                            matched = True
                                            break
                    
                    if not matched:
                        # Debug: show what we were trying to match
                        print(f"      ⚠️  No metadata found for {obj_id}: {obj_str}")
                
                # Save updated metadata
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"    ✅ Updated {updated_count} objects in metadata.json")
            else:
                print(f"    ⚠️  metadata.json not found: {metadata_path}")
            
            return True
            
        except Exception as e:
            print(f"    ❌ Error in step 4: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_single_scene(self, scene_dir):
        """
        Process a single scene directory
        """
        scene_name = scene_dir.name
        usd_path = scene_dir / "export" / "export_scene.blend" / "export_scene.usdc"
        
        print(f"\n📁 Processing: {scene_name}")
        
        if not usd_path.exists():
            print(f"    ⚠️ USD file not found: {usd_path}")
            return False
        
        success = True
        
        # # Step 1: Restructure stage
        # if not self.step1_restructure_stage(usd_path):
        #     success = False
        
        # # Step 2: Deactivate elements
        # if not self.step2_deactivate_elements(usd_path):
        #     success = False
        
        # # Step 3: Set lighting mode
        # if not self.step3_set_lighting_mode(usd_path):
        #     success = False
        
        # Step 4: Lower KitchenSpaceFactory objects
        if not self.step4_lower_kitchen_space_objects(scene_dir, usd_path):
            success = False
        
        if success:
            print(f"    ✅ {scene_name} processed successfully")
        else:
            print(f"    ❌ {scene_name} had errors")
        
        return success
    
    def process_all_scenes(self):
        """
        Process all scene directories in the base directory
        """
        # Get all scene directories
        scene_dirs = [
            d for d in self.base_dir.iterdir()
            if d.is_dir() and d.name.startswith("scene_")
        ]
        
        # Sort naturally
        scene_dirs.sort(key=lambda x: [
            int(c) if c.isdigit() else c
            for c in __import__('re').split(r'(\d+)', x.name)
        ])
        
        if not scene_dirs:
            print(f"❌ No scene directories found in {self.base_dir}")
            return
        
        print(f"🔍 Found {len(scene_dirs)} scene directories")
        print("="*80)
        
        successful = 0
        failed = 0
        
        for scene_dir in scene_dirs:
            try:
                if self.process_single_scene(scene_dir):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"    ❌ Unexpected error processing {scene_dir.name}: {e}")
                failed += 1
            
            print("-" * 60)
        
        # Summary
        print("\n" + "="*80)
        print("📊 PROCESSING SUMMARY")
        print("="*80)
        print(f"✅ Successfully processed: {successful} scenes")
        print(f"❌ Failed: {failed} scenes")
        print(f"📊 Total scenes: {len(scene_dirs)}")


def main():
    # Configuration
    BASE_DIR = "/home/xinhai/Documents/automoma/assets/scene/infinigen/kitchen_1130"
    
    # Validate base directory
    base_path = Path(BASE_DIR)
    if not base_path.exists():
        print(f"❌ Base directory does not exist: {BASE_DIR}")
        sys.exit(1)
    
    print(f"🚀 Starting USD preprocessing for: {BASE_DIR}")
    print(f"📋 Operations to perform:")
    print("   1. Restructure stage hierarchy (/World → /World/scene)")
    print("   2. Deactivate ceiling/wall elements")
    print("   3. Set lighting mode to 2")
    print("   4. Lower objects on KitchenSpaceFactory")
    print()
    
    # Create processor and run
    processor = USDPreprocessor(BASE_DIR)
    processor.process_all_scenes()
    
    print("\n🎉 Batch preprocessing complete!")


if __name__ == "__main__":
    main()