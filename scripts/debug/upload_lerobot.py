import os
import subprocess
import yaml
from pathlib import Path

# --- 配置区域 ---
LOCAL_DATA_ROOT = "/home/xinhai/projects/automoma/data"
YUNPAN_DATA_ROOT = "/Research/automoma/data/friday_data" # 根据你之前的例子，云盘路径通常是 Research 开头
REPO_ID = "multi_object_open"
SUB_DIR = "lerobot"

OBJECTS = ["7221", "11622", "103634", "46197", "101773"]
LOG_FILE = os.path.join(LOCAL_DATA_ROOT, REPO_ID, "upload_log.yaml")
ALIYUNPAN_PATH = "/home/xinhai/env/aliyunpan-v0.3.7-linux-amd64/aliyunpan" # 确保该工具在当前目录下或在PATH中

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}

def save_log(log_data):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        yaml.dump(log_data, f, default_flow_style=False)

def upload_folder(folder_name):
    local_path = os.path.join(LOCAL_DATA_ROOT, REPO_ID, SUB_DIR, folder_name)
    remote_path = os.path.join(YUNPAN_DATA_ROOT, REPO_ID, SUB_DIR) + "/"
    
    if not os.path.exists(local_path):
        return False, "Not Found"

    print(f"--- 正在上传: {folder_name} ---")
    # 构建命令: ./aliyunpan upload [本地路径] [远程目录]
    cmd = [ALIYUNPAN_PATH, "upload", local_path, remote_path]
    
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            return True, "Success"
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e}"
    
    return False, "Failed"

def main():
    upload_log = load_log()
    # 确保结构存在
    if REPO_ID not in upload_log:
        upload_log[REPO_ID] = {}

    # 获取本地实际存在的文件夹列表
    local_lerobot_path = os.path.join(LOCAL_DATA_ROOT, REPO_ID, SUB_DIR)
    if not os.path.exists(local_lerobot_path):
        print(f"错误: 本地路径 {local_lerobot_path} 不存在")
        return

    # 按 Object ID 顺序处理
    for obj_id in OBJECTS:
        # 获取该 object 下的所有文件夹并排序 (按 scene_i 排序)
        folders = [f for f in os.listdir(local_lerobot_path) if f.startswith(f"{REPO_ID}_{obj_id}_scene_")]
        
        # 简单排序：确保 scene_0 在 scene_10 之前
        folders.sort(key=lambda x: int(x.split('scene_')[1].split('_')[0]))

        for folder in folders:
            # 检查是否已记录在日志中
            if upload_log[REPO_ID].get(folder) == "Success":
                print(f"跳过已上传: {folder}")
                continue

            # 执行上传
            success, status = upload_folder(folder)
            
            if success:
                print(f"成功上传并记录: {folder}")
                upload_log[REPO_ID][folder] = "Success"
                # 实时保存日志，防止中途崩溃丢失进度
                save_log(upload_log)
            else:
                print(f"跳过 {folder}, 原因: {status}")

if __name__ == "__main__":
    main()