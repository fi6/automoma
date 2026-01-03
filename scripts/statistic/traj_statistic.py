import os
import torch
import re
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

def analyze_experimental_results(file_path):
    with open(file_path, 'r') as f:
        # 使用 safe_load 避免安全风险
        data = yaml.safe_load(f)

    # 1. 对 Docker 容器名进行排序 (automoma-docker-1, 2, 3...)
    docker_keys = sorted(
        data.keys(), 
        key=lambda x: int(re.search(r'docker-(\d+)', x).group(1)) if re.search(r'docker-(\d+)', x) else x
    )

    for docker_node in docker_keys:
        docker_content = data[docker_node]
        docker_success_sum = 0
        
        print(f"\n{'='*20} {docker_node} {'='*20}")

        # 遍历任务 (如 multi_object_open)
        for task_name, task_content in docker_content.items():
            # 这里的路径根据你的 YAML 结构：traj -> summit_franka
            robot_data = task_content.get('traj', {}).get('summit_franka', {})
            
            # 2. 对场景 ID 进行排序 (scene_0, scene_1, scene_2...)
            sorted_scene_ids = sorted(
                robot_data.keys(),
                key=lambda x: int(re.search(r'scene_(\d+)', x).group(1)) if re.search(r'scene_(\d+)', x) else x
            )

            for scene_id in sorted_scene_ids:
                scene_data = robot_data[scene_id]
                scene_total_success = 0
                
                # 遍历内部嵌套结构：seed_id -> grasp_id -> stage_id -> traj_data.pt
                if isinstance(scene_data, dict):
                    for seed_id, seed_content in scene_data.items():
                        if isinstance(seed_content, dict):
                            for grasp_id, grasp_content in seed_content.items():
                                if isinstance(grasp_content, dict):
                                    for stage_id, stage_content in grasp_content.items():
                                        # 提取具体成功数
                                        success = stage_content.get('traj_data.pt', {}).get('num_success', 0)
                                        scene_total_success += success
                
                # 打印当前场景的统计
                print(f"  {scene_id:25} | Success: {scene_total_success}")
                docker_success_sum += scene_total_success

        # 3. 打印该 Docker 容器的总计
        print(f"{'-'*50}")
        print(f"TOTAL for {docker_node}: {docker_success_sum}")
        print(f"{'-'*50}")

def main():
    # Set your data root path here
    data_root = "data" 
    output_yaml = f"{data_root}/data_statistics.yaml"
    
    # print(f"Starting statistics collection for: {data_root}")
    # tree_data = recursive_stats(data_root)
    
    # with open(output_yaml, 'w') as f:
    #     yaml.dump(tree_data, f, sort_keys=False, default_flow_style=False)
    
    # print(f"Statistics successfully saved to {output_yaml}")
    
    print("\nStarting analysis of `num_success` from the generated YAML...")
    analyze_experimental_results(output_yaml)

if __name__ == "__main__":
    main()