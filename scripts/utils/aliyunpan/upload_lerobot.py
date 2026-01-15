import os
import subprocess
import yaml

# --- 配置区域 ---
LOCAL_DATA_ROOT = "/home/xinhai/projects/automoma/data"
YUNPAN_DATA_ROOT = "/Research/automoma/data/friday_data"
REPO_ID = "single_object_reach"
SUB_DIR = "lerobot"
OBJECTS = ["7221", "11622", "103634", "46197", "101773"]

# 自动获取脚本所在目录下的日志路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "upload_log.yaml")
ALIYUNPAN_PATH = "/home/xinhai/env/aliyunpan-v0.3.7-linux-amd64/aliyunpan"

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}

def save_log(log_data):
    with open(LOG_FILE, 'w') as f:
        yaml.dump(log_data, f, default_flow_style=False)

def upload_folder(folder_name):
    local_path = os.path.join(LOCAL_DATA_ROOT, REPO_ID, SUB_DIR, folder_name)
    remote_path = os.path.join(YUNPAN_DATA_ROOT, REPO_ID, SUB_DIR) + "/"
    
    if not os.path.exists(local_path):
        return False, "Local path not found"

    print(f"--- 正在上传: {folder_name} ---")
    cmd = [ALIYUNPAN_PATH, "upload", local_path, remote_path]
    
    try:
        result = subprocess.run(cmd, check=True)
        return True, "Success"
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e}"

def main():
    upload_log = load_log()
    if REPO_ID not in upload_log:
        upload_log[REPO_ID] = {}

    local_lerobot_path = os.path.join(LOCAL_DATA_ROOT, REPO_ID, SUB_DIR)
    if not os.path.exists(local_lerobot_path):
        print(f"错误: 本地路径 {local_lerobot_path} 不存在")
        return

    # 筛选并排序
    folders = [f for f in os.listdir(local_lerobot_path) 
               if f.startswith(f"{REPO_ID}_") and "scene_" in f]
    
    # 按物体ID和场景编号排序
    folders.sort() 

    for folder in folders:
        if upload_log[REPO_ID].get(folder) == "Success":
            print(f"跳过已上传: {folder}")
            continue

        success, status = upload_folder(folder)
        if success:
            upload_log[REPO_ID][folder] = "Success"
            save_log(upload_log)
            print(f"完成: {folder}")
        else:
            print(f"失败: {folder}, 原因: {status}")

if __name__ == "__main__":
    main()