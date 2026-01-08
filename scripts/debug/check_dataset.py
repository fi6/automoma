import os
import sys
import glob
from pathlib import Path

# --- 1. 环境配置 ---
# 自动添加 lerobot 源码路径
lerobot_path = os.path.join(os.getcwd(), "third_party/lerobot/src")
if lerobot_path not in sys.path:
    sys.path.append(lerobot_path)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    print(f"❌ 错误: 无法导入 LeRobotDataset。")
    print(f"请确保在 'projects/automoma' 根目录下运行，且 '{lerobot_path}' 存在。")
    sys.exit(1)

def check_integrity():
    # --- 2. 搜索数据集 ---
    # 路径匹配: data/automoma-docker-*/multi_object_open/lerobot/*
    base_pattern = os.path.join("data", "automoma-docker-*", "multi_object_open", "lerobot", "*")
    
    print(f"🔍 正在扫描路径: {base_pattern} ...")
    all_paths = glob.glob(base_pattern)
    
    # 仅保留目录
    dataset_paths = [Path(p) for p in all_paths if os.path.isdir(p)]
    dataset_paths.sort()
    
    if not dataset_paths:
        print("未找到任何数据集目录。")
        return

    print(f"共发现 {len(dataset_paths)} 个数据集目录。开始检测...\n")
    print(f"{'数据集 ID':<50} | {'状态':<10} | {'详情'}")
    print("-" * 100)
    
    valid_count = 0
    empty_count = 0
    corrupt_count = 0
    
    # 用于存储异常数据的列表，方便最后汇总
    empty_datasets = []
    corrupt_datasets = []

    # --- 3. 遍历检测 ---
    for path in dataset_paths:
        repo_id = path.name
        
        # 格式化输出前缀
        print(f"{repo_id:<50} | ", end="", flush=True)
        
        # [检测 1]: 检查文件夹是否为空
        # 排除 .DS_Store 等隐藏文件干扰，查看是否有实质性内容
        files_in_dir = [f for f in os.listdir(path) if not f.startswith('.')]
        
        if len(files_in_dir) == 0:
            print("⚠️ 空目录  | 文件夹存在但无内容")
            empty_count += 1
            empty_datasets.append(str(path))
            continue # 如果是空文件夹，跳过后续加载测试

        # [检测 2]: 尝试加载 (检测数据损坏)
        try:
            # 尝试加载第0帧来触发 Parquet/Video 读取
            ds = LeRobotDataset(
                repo_id=repo_id,
                root=path,
                episodes=[0]
            )
            # 强制读取元数据属性
            _ = ds.meta
            
            print("✅ 正常    | OK")
            valid_count += 1
            
        except Exception as e:
            print("❌ 损坏    | 加载失败")
            
            # 分析错误原因
            error_msg = str(e)
            if "Parquet magic bytes not found" in error_msg:
                reason = "Parquet头损坏"
            elif "No such file or directory" in error_msg:
                reason = "关键文件缺失"
            else:
                reason = error_msg.split('\n')[-1][:50] # 取最后一行错误信息
            
            corrupt_datasets.append({
                "path": str(path),
                "reason": reason
            })
            corrupt_count += 1

    # --- 4. 最终报告 ---
    print("\n" + "="*60)
    print("📊 检测结果汇总")
    print("="*60)
    print(f"总扫描目录: {len(dataset_paths)}")
    print(f"✅ 正常可用: {valid_count}")
    print(f"⚠️  空文件夹: {empty_count}")
    print(f"❌ 数据损坏: {corrupt_count}")
    
    if empty_count > 0:
        print("\n⚠️  [空文件夹列表] (建议删除或重新下载):")
        for p in empty_datasets:
            print(f"  - {p}")

    if corrupt_count > 0:
        print("\n❌ [损坏数据列表] (建议检查 Parquet/Video 文件):")
        for item in corrupt_datasets:
            print(f"  - 路径: {item['path']}")
            print(f"    原因: {item['reason']}")

if __name__ == "__main__":
    check_integrity()
