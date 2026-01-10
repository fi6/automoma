import os
import sys
import re
import shutil
import yaml
import argparse
from pathlib import Path

# --- 环境配置 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
lerobot_path = PROJECT_ROOT / "third_party/lerobot/src"
if str(lerobot_path) not in sys.path:
    sys.path.append(str(lerobot_path))

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    LeRobotDataset = None

def natural_sort_key(path_obj):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(path_obj))]

def check_dataset_status(path):
    """检测逻辑"""
    files = [f for f in os.listdir(path) if not f.startswith('.')]
    if not files:
        return "empty", "文件夹为空"
    
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        # 仅加载元数据进行轻量级校验
        ds = LeRobotDataset(repo_id=path.name, root=path, episodes=[0])
        _ = ds.meta
        return "valid", "OK"
    except Exception as e:
        return "corrupt", str(e).split('\n')[-1]

def run_manager():
    parser = argparse.ArgumentParser(description="LeRobot 数据集管理工具")
    parser.add_argument("mode", choices=["check", "move"], help="模式: check (全量扫描) 或 move (根据 yaml 移动)")
    args = parser.parse_args()

    data_ing_dir = PROJECT_ROOT / "data"
    data_dir = PROJECT_ROOT / "data"
    output_log = PROJECT_ROOT / "outputs" / "data_check.yaml"

    # --- MODE: CHECK ---
    if args.mode == "check":
        print(f"🔍 启动 CHECK 模式，正在扫描 {data_ing_dir}...")
        dataset_paths = sorted(list(data_ing_dir.glob("multi_object_open/lerobot/*")), key=natural_sort_key)
        
        results = {"valid": [], "empty": [], "corrupt": []}
        
        for path in dataset_paths:
            status, reason = check_dataset_status(path)
            rel_path = path.relative_to(data_ing_dir)
            results[status].append({"path": str(rel_path), "reason": reason})
            
            icon = {"valid": "✅", "empty": "⚠️", "corrupt": "❌"}[status]
            print(f"{icon} [{status.upper()}] {rel_path}")

        output_log.parent.mkdir(parents=True, exist_ok=True)
        with open(output_log, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, allow_unicode=True, sort_keys=False)
        print(f"\n✨ 检测完成，结果已保存至: {output_log}")

    # --- MODE: MOVE ---
    elif args.mode == "move":
        print(f"📦 启动 MOVE 模式...")
        
        if not output_log.exists():
            print(f"❌ 错误: 未找到检测记录文件 {output_log}")
            print("请先运行: python manage_datasets.py check")
            return

        with open(output_log, 'r', encoding='utf-8') as f:
            data_map = yaml.safe_load(f)

        valid_list = data_map.get("valid", [])
        if not valid_list:
            print("ℹ️  没有发现可移动的正常数据集 (valid 为空)。")
            return

        print(f"确认移动 {len(valid_list)} 个通过校验的数据集...")
        
        move_count = 0
        for item in valid_list:
            rel_path_str = item['path']
            src_path = data_ing_dir / rel_path_str
            dest_path = data_dir / rel_path_str

            if src_path.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if not dest_path.exists():
                    shutil.move(str(src_path), str(dest_path))
                    print(f"✅ 已移动: {rel_path_str}")
                    move_count += 1
                else:
                    print(f"⏩ 跳过: {rel_path_str} (目标目录已存在)")
            else:
                print(f"⚠️  警告: 记录中的文件已不存在: {rel_path_str}")

        print(f"\n🎉 移动任务完成！共计移动 {move_count} 个文件夹。")

if __name__ == "__main__":
    run_manager()