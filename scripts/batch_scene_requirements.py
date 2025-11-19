import os
import shutil
import sys
from pathlib import Path
import re


class BatchSceneRequirementsProcessor:
    def __init__(self, base_dir, source_requirement_file):
        self.base_dir = Path(base_dir)
        self.source_requirement_file = Path(source_requirement_file)
        
    def validate_inputs(self):
        """Validate that the source file and base directory exist"""
        if not self.source_requirement_file.exists():
            print(f"❌ Source requirement file not found: {self.source_requirement_file}")
            return False
        
        if not self.base_dir.exists():
            print(f"❌ Base directory does not exist: {self.base_dir}")
            return False
        
        return True
    
    def get_scene_directories(self):
        """
        Get all scene directories matching pattern scene_*_seed_*
        """
        scene_dirs = [
            d for d in self.base_dir.iterdir()
            if d.is_dir() and re.match(r'scene_\d+_seed_\d+', d.name)
        ]
        
        # Sort naturally by scene number and seed
        scene_dirs.sort(key=lambda x: [
            int(c) if c.isdigit() else c
            for c in re.split(r'(\d+)', x.name)
        ])
        
        return scene_dirs
    
    def process_single_scene(self, scene_dir):
        """
        Copy requirement.json to the info directory of a single scene
        """
        info_dir = scene_dir / "info"
        
        if not info_dir.exists():
            print(f"    ⚠️ info directory not found in {scene_dir.name}, creating it...")
            try:
                info_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"    ❌ Failed to create info directory: {e}")
                return False
        
        destination_file = info_dir / "requirement.json"
        
        try:
            shutil.copy2(self.source_requirement_file, destination_file)
            print(f"    ✅ Copied to {scene_dir.name}/info/requirement.json")
            return True
        except Exception as e:
            print(f"    ❌ Failed to copy to {scene_dir.name}: {e}")
            return False
    
    def process_all_scenes(self):
        """
        Process all scene directories
        """
        scene_dirs = self.get_scene_directories()
        
        if not scene_dirs:
            print(f"❌ No scene directories found matching pattern 'scene_*_seed_*' in {self.base_dir}")
            return False
        
        print(f"🔍 Found {len(scene_dirs)} scene directories")
        print("="*80)
        
        successful = 0
        failed = 0
        
        for scene_dir in scene_dirs:
            try:
                print(f"📁 Processing: {scene_dir.name}")
                if self.process_single_scene(scene_dir):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"    ❌ Unexpected error processing {scene_dir.name}: {e}")
                failed += 1
        
        # Summary
        print("\n" + "="*80)
        print("📊 PROCESSING SUMMARY")
        print("="*80)
        print(f"✅ Successfully processed: {successful} scenes")
        print(f"❌ Failed: {failed} scenes")
        print(f"📊 Total scenes: {len(scene_dirs)}")
        
        return failed == 0


def main():
    # Configuration
    SOURCE_FILE = "/home/yida/projects/automoma/scripts/debug/requirement.json"
    BASE_DIR = "/home/yida/projects/automoma/output/collect/infinigen_scene_100"
    
    print(f"🚀 Starting batch scene requirements processing")
    print(f"📝 Source file: {SOURCE_FILE}")
    print(f"📁 Base directory: {BASE_DIR}")
    print()
    
    processor = BatchSceneRequirementsProcessor(BASE_DIR, SOURCE_FILE)
    
    # Validate inputs
    if not processor.validate_inputs():
        sys.exit(1)
    
    # Process all scenes
    success = processor.process_all_scenes()
    
    if success:
        print("\n🎉 Batch processing complete successfully!")
        sys.exit(0)
    else:
        print("\n⚠️ Batch processing completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
