import os
import torch
import yaml
from typing import Dict, Any

def get_file_stats(file_path: str, file_type: str) -> Dict[str, int]:
    """Extracts counts from .pt files."""
    try:
        # Using weights_only=False as seen in your file_utils.py
        data = torch.load(file_path, weights_only=False, map_location='cpu')
        
        if file_type == "ik":
            # Count entries in the 'iks' or 'start_iks' tensor
            # Adjust key names if your actual .pt files differ
            iks = data.get("iks") if data.get("iks") is not None else data.get("start_iks")
            return {"num_ik": iks.shape[0] if torch.is_tensor(iks) else 0}
        
        elif file_type == "traj":
            # trajectories is usually [Batch, Time, Dims]
            trajs = data.get("trajectories")
            success = data.get("success")
            
            num_traj = trajs.shape[0] if torch.is_tensor(trajs) else 0
            # Success is typically a boolean mask/tensor
            num_success = int(success.sum().item()) if torch.is_tensor(success) else 0
            
            return {
                "num_traj": num_traj,
                "num_success": num_success
            }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}

def recursive_stats(path: str) -> Dict[str, Any]:
    """Recursively crawls the directory to build the tree stats."""
    node_stats = {}
    items = sorted(os.listdir(path))
    
    # Track totals at this level for the YAML summary
    current_level_files = []
    
    for item in items:
        full_path = os.path.join(path, item)
        
        if os.path.isdir(full_path):
            child_stats = recursive_stats(full_path)
            if child_stats: # Only add if directory contains relevant data
                node_stats[item] = child_stats
        
        elif item.endswith(".pt"):
            if "iks" in item:
                stats = get_file_stats(full_path, "ik")
                node_stats[item] = stats
            elif "traj" in item:
                stats = get_file_stats(full_path, "traj")
                node_stats[item] = stats

    return node_stats

def main():
    # Set your data root path here
    data_root = "data" 
    output_yaml = f"{data_root}/data_statistics.yaml"
    
    print(f"Starting statistics collection for: {data_root}")
    tree_data = recursive_stats(data_root)
    
    with open(output_yaml, 'w') as f:
        yaml.dump(tree_data, f, sort_keys=False, default_flow_style=False)
    
    print(f"Statistics successfully saved to {output_yaml}")

if __name__ == "__main__":
    main()