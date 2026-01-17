import os
import subprocess
import yaml
import argparse

# --- 配置区域 ---
# 下载到哪？（可以按需修改）
LOCAL_DOWNLOAD_ROOT = "/home/xinhai/projects/automoma/data_aliyunpan_download"
YUNPAN_DATA_ROOT = "/Research/automoma/data/friday_data"
REPO_ID = "multi_object_open"
SUB_DIR = "lerobot"

# 路径处理
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_LOG_FILE = os.path.join(SCRIPT_DIR, "upload_log.yaml")
DOWNLOAD_LOG_FILE = os.path.join(SCRIPT_DIR, "download_log.yaml")
ALIYUNPAN_PATH = "/home/xinhai/env/aliyunpan-v0.3.7-linux-amd64/aliyunpan"

def parse_args():
    parser = argparse.ArgumentParser(description="Download lerobot folders from aliyunpan")
    parser.add_argument('--exp', type=str, help='Specific folder name to download (e.g., multi_object_open_7221_scene_25_seed_25)')
    parser.add_argument('--force', action='store_true', help='Force download even if marked Success in log')
    return parser.parse_args()

def load_yaml(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}

def save_yaml(path, data):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def download_folder(folder_name):
    remote_path = os.path.join(YUNPAN_DATA_ROOT, REPO_ID, SUB_DIR, folder_name)
    local_dest_dir = os.path.join(LOCAL_DOWNLOAD_ROOT)
    
    # 确保本地目标父目录存在
    os.makedirs(local_dest_dir, exist_ok=True)

    print(f"--- 正在下载: {folder_name} ---")
    # 语法: aliyunpan download <远程> --saveto <本地>
    cmd = [ALIYUNPAN_PATH, "download", remote_path, "--saveto", local_dest_dir]
    
    try:
        subprocess.run(cmd, check=True)
        return True, "Success"
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e}"

def main():
    args = parse_args()
    upload_log = load_yaml(UPLOAD_LOG_FILE)
    download_log = load_yaml(DOWNLOAD_LOG_FILE)

    if REPO_ID not in upload_log:
        print("错误: upload_log.yaml 中没有记录。")
        return
    
    if REPO_ID not in download_log:
        download_log[REPO_ID] = {}

    if args.exp:
        tasks = [args.exp]
    else:
        # 获取所有上传成功的文件夹
        tasks = [f for f, s in upload_log[REPO_ID].items() if s == "Success"]
        tasks.sort()

    for folder in tasks:
        if args.exp and upload_log[REPO_ID].get(folder) != "Success":
            print(f"警告: {folder} 在 upload_log 中没有标记为 Success，仍将尝试下载。")

        if download_log[REPO_ID].get(folder) == "Success" and not args.force:
            print(f"跳过已下载: {folder}")
            continue

        success, status = download_folder(folder)
        if success:
            download_log[REPO_ID][folder] = "Success"
            save_yaml(DOWNLOAD_LOG_FILE, download_log)
            print(f"下载成功: {folder}")
        else:
            print(f"下载失败: {folder}, 错误: {status}")

if __name__ == "__main__":
    main()