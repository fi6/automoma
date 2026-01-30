#!/usr/bin/env python3
"""
Test the updated MMD analysis with config-based approach.

This script creates mock cache files and verifies the MMD analysis works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import torch
import numpy as np
from pathlib import Path
from config import IK_CUROBO, OUTPUT_ROOT

def create_mock_cache():
    """Create mock cache files for testing."""
    cache_dir = IK_CUROBO['cache_dir']
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating mock cache files in: {cache_dir}")
    
    # Mock configuration
    scene_name = "scene_0_seed_0"
    grasp_id = 0
    seed_values = [5000, 10000, 20000, 40000, 80000]
    
    for num_seeds in seed_values:
        # Generate mock IK data (11D: 10 robot joints + 1 object joint)
        # Use different distributions to simulate different seed quantities
        np.random.seed(num_seeds)
        
        # Simulate: more seeds = better coverage of the distribution
        n_samples = min(num_seeds // 100, 1000)  # Cap at 1000 for memory
        
        start_ik = torch.tensor(np.random.randn(n_samples, 11), dtype=torch.float32)
        goal_ik = torch.tensor(np.random.randn(n_samples, 11), dtype=torch.float32)
        
        # Save cache file with correct naming format
        cache_file = cache_dir / f"{scene_name}_grasp_{grasp_id:04d}_seeds_{num_seeds}_ik.pt"
        
        torch.save({
            'start_ik': start_ik,
            'goal_ik': goal_ik,
            'num_seeds': num_seeds,
            'scene_name': scene_name,
            'grasp_id': grasp_id
        }, cache_file)
        
        print(f"  ✓ Created: {cache_file.name}")
        print(f"    Start IK: {start_ik.shape}, Goal IK: {goal_ik.shape}")
    
    print(f"\nMock cache files created successfully!")
    return cache_dir


def verify_cache_files():
    """Verify cache files can be loaded."""
    cache_dir = IK_CUROBO['cache_dir']
    
    print(f"\nVerifying cache files...")
    
    scene_name = "scene_0_seed_0"
    grasp_id = 0
    seed_values = [5000, 10000, 20000, 40000, 80000]
    
    for num_seeds in seed_values:
        cache_file = cache_dir / f"{scene_name}_grasp_{grasp_id:04d}_seeds_{num_seeds}_ik.pt"
        
        if not cache_file.exists():
            print(f"  ✗ Missing: {cache_file.name}")
            return False
        
        try:
            data = torch.load(cache_file)
            start_ik = data['start_ik']
            goal_ik = data['goal_ik']
            print(f"  ✓ Loaded: {cache_file.name}")
            print(f"    Start IK: {start_ik.shape}, Goal IK: {goal_ik.shape}")
        except Exception as e:
            print(f"  ✗ Error loading {cache_file.name}: {e}")
            return False
    
    print(f"\nAll cache files verified successfully!")
    return True


def test_load_function():
    """Test the load_ik_from_cache function."""
    print(f"\nTesting load_ik_from_cache function...")
    
    from analyze_ik_mmd import load_ik_from_cache
    
    scene_name = "scene_0_seed_0"
    grasp_id = 0
    
    # Test loading existing file
    result = load_ik_from_cache(scene_name, grasp_id, 5000)
    if result is not None:
        start_ik, goal_ik = result
        print(f"  ✓ Successfully loaded 5000 seeds")
        print(f"    Start IK: {start_ik.shape}, Goal IK: {goal_ik.shape}")
    else:
        print(f"  ✗ Failed to load 5000 seeds")
        return False
    
    # Test loading non-existent file
    result = load_ik_from_cache(scene_name, grasp_id, 999999)
    if result is None:
        print(f"  ✓ Correctly returned None for non-existent file")
    else:
        print(f"  ✗ Should have returned None for non-existent file")
        return False
    
    print(f"\nload_ik_from_cache function works correctly!")
    return True


def print_config_info():
    """Print current configuration."""
    print("\n" + "="*80)
    print("Current Configuration (from config.py)")
    print("="*80)
    print(f"Seed values: {IK_CUROBO['num_seeds_values']}")
    print(f"Analysis scenes: {IK_CUROBO.get('analysis_scenes', 'Not set')}")
    print(f"Analysis grasp IDs: {IK_CUROBO.get('analysis_grasp_ids', 'Not set')}")
    print(f"Cache directory: {IK_CUROBO['cache_dir']}")
    print(f"Cache enabled: {IK_CUROBO['enable_cache']}")
    print("="*80)


def main():
    """Run all tests."""
    print("="*80)
    print("Testing Updated MMD Analysis (Config-based)")
    print("="*80)
    
    # Print configuration
    print_config_info()
    
    # Create mock cache files
    cache_dir = create_mock_cache()
    
    # Verify cache files
    if not verify_cache_files():
        print("\n✗ Cache verification failed!")
        return 1
    
    # Test load function
    if not test_load_function():
        print("\n✗ Load function test failed!")
        return 1
    
    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)
    print("\nYou can now run the MMD analysis:")
    print("  python scripts/statistic/analyze_ik_mmd.py")
    print("\nMock cache files are in:")
    print(f"  {cache_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
