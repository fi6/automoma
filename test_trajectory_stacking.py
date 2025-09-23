#!/usr/bin/env python3
"""
Test script to validate trajectory stacking functionality without Isaac Sim
"""
import os
import sys
import torch
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Import the functions we want to test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
from pipeline_collect import load_and_stack_trajectory_data, randomly_sample_trajectories, save_trajectory_data

def test_trajectory_stacking():
    """Test trajectory loading and stacking"""
    
    # Test with scene_1_seed_1/7221
    scene_asset_dir = "/home/xinhai/automoma/output/traj/summit_franka/scene_1_seed_1/7221"
    
    if not os.path.exists(scene_asset_dir):
        print(f"Test directory not found: {scene_asset_dir}")
        return
    
    print(f"Testing trajectory stacking for: {scene_asset_dir}")
    
    try:
        # Step 1: Load and stack trajectory data
        stacked_start, stacked_goal, stacked_traj, stacked_success, grasp_sources = load_and_stack_trajectory_data(scene_asset_dir)
        print(f"Successfully stacked trajectories!")
        print(f"Total trajectories: {len(stacked_start)}")
        print(f"Start states shape: {stacked_start.shape}")
        print(f"Goal states shape: {stacked_goal.shape}")  
        print(f"Trajectories shape: {stacked_traj.shape}")
        print(f"Success rate: {stacked_success.sum().item()}/{len(stacked_success)}")
        
        # Count by grasp source
        from collections import Counter
        source_counts = Counter(grasp_sources)
        print(f"Grasp source distribution: {dict(source_counts)}")
        
        # Step 2: Test random sampling
        num_episodes = min(100, len(stacked_start))  # Sample at most 100 or all available
        print(f"\nTesting random sampling of {num_episodes} trajectories...")
        
        sampled_start, sampled_goal, sampled_traj, sampled_success, sampled_sources, selected_indices = randomly_sample_trajectories(
            stacked_start, stacked_goal, stacked_traj, stacked_success, grasp_sources, num_episodes, 42)
        
        print(f"Successfully sampled {len(sampled_start)} trajectories")
        print(f"Selected indices (first 10): {selected_indices[:10]}")
        
        sampled_source_counts = Counter(sampled_sources)  
        print(f"Sampled grasp source distribution: {dict(sampled_source_counts)}")
        
        # Step 3: Test saving trajectory data
        print(f"\nTesting saving trajectory data...")
        test_output_dir = "/tmp/test_traj_output"
        os.makedirs(test_output_dir, exist_ok=True)
        
        save_trajectory_data(test_output_dir, stacked_start, stacked_goal, stacked_traj, 
                           stacked_success, grasp_sources, "test_total_filtered_traj_data.pt")
        
        save_trajectory_data(test_output_dir, sampled_start, sampled_goal, sampled_traj,
                           sampled_success, sampled_sources, "test_selected_filtered_traj_data.pt")
        
        print(f"Successfully saved trajectory data to {test_output_dir}")
        
        # Verify saved data can be loaded
        total_data = torch.load(os.path.join(test_output_dir, "test_total_filtered_traj_data.pt"), weights_only=False)
        selected_data = torch.load(os.path.join(test_output_dir, "test_selected_filtered_traj_data.pt"), weights_only=False)
        
        print(f"Loaded total data: {total_data['num_trajectories']} trajectories")
        print(f"Loaded selected data: {selected_data['num_trajectories']} trajectories")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trajectory_stacking()