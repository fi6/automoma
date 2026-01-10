import os
import sys
import glob
import re
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"

# --- 1. 环境配置 ---
lerobot_path = os.path.join(os.getcwd(), "third_party/lerobot/src")
if lerobot_path not in sys.path:
    sys.path.append(lerobot_path)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    print(f"❌ 错误: 无法导入 LeRobotDataset。\n请在 projects/automoma 根目录下运行。\n{e}")
    sys.exit(1)

# --- 新增: 自然排序算法 ---
def natural_sort_key(path):
    """
    将字符串中的数字提取出来作为排序键，实现 0, 1, 2, ..., 10 的自然顺序。
    """
    # 获取文件名（例如: multi_object_open_..._scene_2_seed_2）
    text = path.name
    # 使用正则将字符串拆分为 [非数字, 数字, 非数字, ...] 的列表
    # 并将数字字符串转换为整数，以便进行数值比较
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

def check_integrity():
    # --- 2. 搜索数据集 ---
    base_pattern = os.path.join("data", "automoma-docker-*", "multi_object_open", "lerobot", "*")
    
    print(f"🔍 正在扫描路径: {base_pattern} ...")
    all_paths = glob.glob(base_pattern)
    dataset_paths = [Path(p) for p in all_paths if os.path.isdir(p)]
    
    # [关键修改]: 使用自然排序键进行排序
    # 这将确保 scene_2 排在 scene_10 之前
    dataset_paths.sort(key=natural_sort_key)
    
    if not dataset_paths:
        print("未找到任何数据集目录。")
        return

    print(f"共发现 {len(dataset_paths)} 个数据集目录。开始按自然顺序检测...\n")
    print(f"{'数据集 ID':<55} | {'状态':<10} | {'详情'}")
    print("-" * 110)
    
    valid_count = 0
    empty_count = 0
    corrupt_count = 0
    
    empty_datasets = []
    corrupt_datasets = []

    # --- 3. 遍历检测 ---
    for path in dataset_paths:
        repo_id = path.name
        
        # 截断过长的名称以保持表格整洁
        display_name = (repo_id[:52] + '..') if len(repo_id) > 52 else repo_id
        print(f"{display_name:<55} | ", end="", flush=True)
        
        # [检测 1]: 判空 (忽略隐藏文件)
        files_in_dir = [f for f in os.listdir(path) if not f.startswith('.')]
        
        if len(files_in_dir) == 0:
            print("⚠️ 空目录  | 文件夹存在但无内容")
            empty_count += 1
            empty_datasets.append(str(path))
            continue

        # [检测 2]: 尝试加载
        try:
            ds = LeRobotDataset(repo_id=repo_id, root=path, episodes=[0])
            _ = ds.meta # 触发元数据读取
            
            print("✅ 正常    | OK")
            valid_count += 1
            
        except Exception as e:
            print("❌ 损坏    | 加载失败")
            
            error_msg = str(e)
            if "Parquet magic bytes not found" in error_msg:
                reason = "Parquet头损坏"
            elif "No such file or directory" in error_msg:
                reason = "关键文件缺失"
            else:
                reason = error_msg.split('\n')[-1]
            print(reason)
            
            corrupt_datasets.append({"path": str(path), "reason": reason})
            corrupt_count += 1

    # --- 4. 最终报告 ---
    print("\n" + "="*60)
    print("📊 检测结果汇总")
    print("="*60)
    print(f"总扫描: {len(dataset_paths)}")
    print(f"✅ 正常: {valid_count}")
    print(f"⚠️  空:   {empty_count}")
    print(f"❌ 损坏: {corrupt_count}")
    
    if empty_count > 0:
        print("\n⚠️  [空文件夹列表]:")
        # 同样对结果列表进行自然排序
        for p in sorted(empty_datasets, key=lambda x: natural_sort_key(Path(x))):
            print(f"  - {p}")

    if corrupt_count > 0:
        print("\n❌ [损坏数据列表]:")
        # 同样对结果列表进行自然排序
        sorted_corrupt = sorted(corrupt_datasets, key=lambda x: natural_sort_key(Path(x['path'])))
        for item in sorted_corrupt:
            print(f"  - {item['path']}")
            print(f"    原因: {item['reason']}")

if __name__ == "__main__":
    check_integrity()
