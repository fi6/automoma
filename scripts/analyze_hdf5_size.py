#!/usr/bin/env python3
"""
Script to analyze HDF5 file structure and estimate output size after processing.
This helps understand the compression and storage requirements.
"""

import h5py
import numpy as np
import os
import sys

def analyze_hdf5_file(file_path):
    """Analyze the structure and size of an HDF5 file"""
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    print("="*80)
    print(f"Analyzing: {file_path}")
    print("="*80)
    
    # Get original file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"\n📁 Original HDF5 file size: {file_size_mb:.2f} MB")
    
    with h5py.File(file_path, 'r') as f:
        print("\n📊 File Structure:")
        print("-"*80)
        
        total_uncompressed = 0
        component_sizes = {}
        
        def visit_item(name, obj):
            nonlocal total_uncompressed
            if isinstance(obj, h5py.Dataset):
                # Calculate uncompressed size
                dtype_size = obj.dtype.itemsize
                num_elements = np.prod(obj.shape)
                size_bytes = dtype_size * num_elements
                size_mb = size_bytes / (1024 * 1024)
                
                total_uncompressed += size_bytes
                
                # Store by component type
                component = name.split('/')[0] if '/' in name else name
                if component not in component_sizes:
                    component_sizes[component] = 0
                component_sizes[component] += size_bytes
                
                print(f"  📄 {name}")
                print(f"     Shape: {obj.shape}, Dtype: {obj.dtype}")
                print(f"     Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
        
        f.visititems(visit_item)
        
        print("\n" + "="*80)
        print("📈 Component Breakdown:")
        print("-"*80)
        for component, size in sorted(component_sizes.items(), key=lambda x: -x[1]):
            size_mb = size / (1024 * 1024)
            percentage = (size / total_uncompressed) * 100
            print(f"  {component:20s}: {size_mb:8.2f} MB ({percentage:5.1f}%)")
        
        print("\n" + "="*80)
        print("💾 Storage Analysis:")
        print("-"*80)
        total_uncompressed_mb = total_uncompressed / (1024 * 1024)
        compression_ratio = total_uncompressed_mb / file_size_mb
        print(f"  Total uncompressed size: {total_uncompressed_mb:.2f} MB")
        print(f"  Actual file size:        {file_size_mb:.2f} MB")
        print(f"  Compression ratio:       {compression_ratio:.2f}x")
        
        # Estimate zarr output sizes for different methods
        print("\n" + "="*80)
        print("🔮 Estimated Output Sizes (per episode):")
        print("-"*80)
        
        # DP3 uses point clouds (assuming 8192 points, 6 dims, float32)
        if '/obs/point_cloud' in f or 'point_cloud' in str(component_sizes.keys()):
            # Get actual point cloud data
            timestamps = 0
            if '/obs/point_cloud' in f:
                pc_data = f['/obs/point_cloud']
                timestamps = pc_data.shape[0]
                pc_size = pc_data.dtype.itemsize * np.prod(pc_data.shape)
            else:
                # Estimate based on typical values
                timestamps = 100  # typical trajectory length
                pc_size = 4 * timestamps * 8192 * 6  # float32 * timestamps * points * dims
            
            print(f"\n  DP3 (Point Cloud based):")
            print(f"    - Point cloud:  {pc_size / (1024*1024):.2f} MB")
            
            # State and action sizes
            state_size = 4 * timestamps * 10  # assuming 10 DOF
            action_size = state_size
            
            print(f"    - State:        {state_size / (1024*1024):.2f} MB")
            print(f"    - Action:       {action_size / (1024*1024):.2f} MB")
            
            total_dp3 = pc_size + state_size + action_size
            # Apply zarr compression (zstd typically 2-3x for numeric data)
            compressed_dp3 = total_dp3 / 2.5
            print(f"    - Total (uncompressed): {total_dp3 / (1024*1024):.2f} MB")
            print(f"    - Estimated zarr (compressed): {compressed_dp3 / (1024*1024):.2f} MB")
        
        # DP and ACT use RGB images
        if '/obs/rgb' in f:
            rgb_group = f['/obs/rgb']
            camera_names = list(rgb_group.keys())
            
            total_image_size = 0
            timestamps = 0
            
            print(f"\n  DP/ACT (Image based):")
            print(f"    Available cameras: {camera_names}")
            
            for cam_name in camera_names:
                cam_data = rgb_group[cam_name]
                timestamps = cam_data.shape[0]
                cam_size = cam_data.dtype.itemsize * np.prod(cam_data.shape)
                total_image_size += cam_size
                print(f"    - {cam_name}: {cam_data.shape}, {cam_size / (1024*1024):.2f} MB")
            
            # State and action sizes
            state_size = 4 * timestamps * 10  # assuming 10 DOF, float32
            action_size = state_size
            
            print(f"    - State:  {state_size / (1024*1024):.2f} MB")
            print(f"    - Action: {action_size / (1024*1024):.2f} MB")
            
            # DP: stores images as uint8 in zarr (NCHW format)
            total_dp = total_image_size + state_size + action_size
            # Image compression with zstd is typically 1.5-2x for uint8 images
            compressed_dp = total_image_size / 1.7 + state_size / 2.5 + action_size / 2.5
            
            print(f"\n    DP (zarr with 3 cameras):")
            print(f"      - Total (uncompressed): {total_dp / (1024*1024):.2f} MB")
            print(f"      - Estimated zarr (compressed): {compressed_dp / (1024*1024):.2f} MB")
            
            # ACT: encodes images as JPEG
            # JPEG compression is typically 10-20x for RGB images at good quality
            jpeg_size = total_image_size / 15  # conservative estimate
            total_act = jpeg_size + state_size + action_size
            
            print(f"\n    ACT (HDF5 with JPEG encoding):")
            print(f"      - Images (JPEG): {jpeg_size / (1024*1024):.2f} MB")
            print(f"      - Total: {total_act / (1024*1024):.2f} MB")
        
        print("\n" + "="*80)
        print("📝 Notes:")
        print("-"*80)
        print("  - DP3: Uses point clouds, zarr with zstd compression")
        print("  - DP:  Uses RGB images from 3 cameras, zarr with zstd compression")
        print("  - ACT: Uses RGB images from 3 cameras with JPEG encoding (smaller but lossy)")
        print("  - Compression ratios are estimates and may vary based on data")
        print("="*80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_hdf5_size.py <path_to_hdf5_file>")
        print("\nExample:")
        print("  python analyze_hdf5_size.py /home/xinhai/automoma/baseline/RoboTwin/data/automoma_manip_summit_franka/task_1object_3scene_20pose/data/episode000000.hdf5")
        sys.exit(1)
    
    file_path = sys.argv[1]
    analyze_hdf5_file(file_path)
