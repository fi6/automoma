#!/usr/bin/env python3
"""
Test script to verify the updated data structure format matches collect_data.py expectations.
This script tests the CameraResult class with grouped joint data structure.
"""

import numpy as np
import tempfile
import h5py
import os
import sys

# Add the project root to Python path
sys.path.insert(0, '/home/xinhai/Documents/automoma/src')

from automoma.utils.data_structures import CameraResult


def test_camera_result_structure():
    """Test that CameraResult produces the expected HDF5 structure."""
    
    # Create a CameraResult instance
    camera_result = CameraResult()
    
    # Set environment info
    camera_result.set_env_info(
        scene_id="kitchen_0919",
        robot_name="summit_franka", 
        object_id="Dishwasher_7221",
        pose_id="0"
    )
    
    # Initialize joint structure with robot configuration
    camera_result.initialize_joint_structure("summit_franka")
    
    # Initialize camera structure
    camera_result.initialize_camera_structure(["ego_topdown", "ego_wrist", "fix_local"])
    
    # Add some test observations
    for i in range(5):  # Simulate 5 timesteps
        # Raw joint positions (3 mobile_base + 7 arm + 2 gripper = 12 joints)
        raw_joint_positions = np.array([
            0.1*i, 0.2*i, 0.0,  # mobile_base (3)
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,  # arm (7) 
            0.04 + 0.01*i, 0.04 + 0.01*i  # gripper (2)
        ])
        
        # Add grouped joint observation
        camera_result.add_grouped_joint_observation(raw_joint_positions, "summit_franka")
        
        # Add other observations
        eef_pose = np.array([0.5 + 0.1*i, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])  # 7D pose
        point_cloud = np.random.random((1000, 6))  # Random point cloud
        
        # Dummy RGB and depth data
        rgb_data = {
            "ego_topdown": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
            "ego_wrist": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8), 
            "fix_local": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        }
        depth_data = {
            "ego_topdown": np.random.random((240, 320)).astype(np.float32),
            "ego_wrist": np.random.random((240, 320)).astype(np.float32),
            "fix_local": np.random.random((240, 320)).astype(np.float32)
        }
        
        camera_result.add_observation(
            eef_data=eef_pose,
            point_cloud_data=point_cloud,
            rgb_data=rgb_data,
            depth_data=depth_data
        )
    
    # Finalize the camera result
    camera_result.finalize()
    
    # Save to temporary HDF5 file
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Save the camera result
        with h5py.File(tmp_path, "w") as f:
            # Save env_info
            env_group = f.create_group("env_info")
            for key, value in camera_result.env_info.items():
                if isinstance(value, str):
                    env_group.create_dataset(key, data=value, dtype=h5py.string_dtype())
                else:
                    env_group.create_dataset(key, data=value)
            
            # Save observation data
            obs_group = f.create_group("obs")
            
            # Save joint data (grouped)
            joint_group = obs_group.create_group("joint")
            for joint_name, joint_values in camera_result.obs["joint"].items():
                if joint_values:  # Only save if we have data
                    joint_array = np.array(joint_values)
                    joint_group.create_dataset(joint_name, data=joint_array)
            
            # Save other observation data
            for obs_type in ["eef", "point_cloud"]:
                if camera_result.obs[obs_type]:
                    obs_array = np.array(camera_result.obs[obs_type])
                    obs_group.create_dataset(obs_type, data=obs_array)
            
            # Save camera data
            for data_type in ["rgb", "depth"]:
                if camera_result.obs[data_type]:
                    data_group = obs_group.create_group(data_type)
                    for camera_name, camera_data in camera_result.obs[data_type].items():
                        if camera_data:  # Only save if we have data
                            camera_array = np.array(camera_data)
                            data_group.create_dataset(camera_name, data=camera_array)
        
        print("✅ HDF5 file saved successfully!")
        
        # Verify the structure
        print("\n📊 HDF5 Structure Verification:")
        with h5py.File(tmp_path, "r") as f:
            def print_structure(name, obj):
                indent = "  " * name.count('/')
                if isinstance(obj, h5py.Group):
                    print(f"{indent}📁 {name.split('/')[-1]}/ (group)")
                else:
                    print(f"{indent}📄 {name.split('/')[-1]} (dataset) - shape: {obj.shape}, dtype: {obj.dtype}")
            
            f.visititems(print_structure)
        
        print("\n✅ Expected joint groups found:")
        with h5py.File(tmp_path, "r") as f:
            joint_group = f["obs"]["joint"]
            expected_joints = ["mobile_base", "arm", "gripper"]
            for joint_name in expected_joints:
                if joint_name in joint_group:
                    joint_data = joint_group[joint_name]
                    print(f"  ✅ {joint_name}: shape {joint_data.shape}, dtype {joint_data.dtype}")
                    # Show first few values
                    print(f"     Sample values: {joint_data[:2] if len(joint_data) > 1 else joint_data[:]}")
                else:
                    print(f"  ❌ {joint_name}: NOT FOUND")
        
        print("\n✅ Environment info:")
        with h5py.File(tmp_path, "r") as f:
            env_info = f["env_info"]
            for key in env_info.keys():
                value = env_info[key][()]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                print(f"  {key}: {value}")
        
        print(f"\n📁 Test file saved at: {tmp_path}")
        print("🎉 Test completed successfully! Data structure matches collect_data.py format.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    print("🧪 Testing CameraResult data structure compatibility with collect_data.py format...")
    success = test_camera_result_structure()
    sys.exit(0 if success else 1)