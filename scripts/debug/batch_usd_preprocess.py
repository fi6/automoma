import os
import sys
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
        
        # Step 1: Restructure stage
        if not self.step1_restructure_stage(usd_path):
            success = False
        
        # Step 2: Deactivate elements
        if not self.step2_deactivate_elements(usd_path):
            success = False
        
        # Step 3: Set lighting mode
        if not self.step3_set_lighting_mode(usd_path):
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
    BASE_DIR = "/home/xinhai/Documents/automoma/output/infinigen_scene_10"
    
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
    print()
    
    # Create processor and run
    processor = USDPreprocessor(BASE_DIR)
    processor.process_all_scenes()
    
    print("\n🎉 Batch preprocessing complete!")


if __name__ == "__main__":
    main()