import os
import subprocess
import yaml

# --- Configuration ---
LOCAL_DOWNLOAD_ROOT = "/home/xinhai/projects/automoma/outputs/train_download"
YUNPAN_DATA_ROOT = "/Research/automoma/ckpt"

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_LOG_FILE = os.path.join(SCRIPT_DIR, "upload_ckpt_log.yaml")
DOWNLOAD_LOG_FILE = os.path.join(SCRIPT_DIR, "download_ckpt_log.yaml")
ALIYUNPAN_PATH = "/home/xinhai/env/aliyunpan-v0.3.7-linux-amd64/aliyunpan"
ALIYUNPAN_PATH_2 = "/home/xinhai/Documents/env/aliyunpan-v0.3.7-linux-amd64/aliyunpan"

def load_yaml(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def save_yaml(path, data):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def download_folder(folder_name):
    # remote path points to the checkpoints directory uploaded earlier
    remote_path = os.path.join(YUNPAN_DATA_ROOT, folder_name, 'checkpoints')
    local_dest_dir = os.path.join(LOCAL_DOWNLOAD_ROOT, folder_name)

    os.makedirs(local_dest_dir, exist_ok=True)

    print(f"--- Downloading: {folder_name} (checkpoints/) ---")
    # Determine which aliyunpan binary to use without reassigning the module constant.
    exec_path = ALIYUNPAN_PATH if os.path.exists(ALIYUNPAN_PATH) else ALIYUNPAN_PATH_2
    if not os.path.exists(exec_path):
        return False, f"Error: aliyunpan not found at {ALIYUNPAN_PATH} or {ALIYUNPAN_PATH_2}"
    cmd = [exec_path, "download", remote_path, "--saveto", local_dest_dir]

    try:
        subprocess.run(cmd, check=True)
        return True, "Success"
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e}"


def main():
    upload_log = load_yaml(UPLOAD_LOG_FILE)
    download_log = load_yaml(DOWNLOAD_LOG_FILE)

    if 'ckpt' not in upload_log:
        print(f"Error: {UPLOAD_LOG_FILE} missing 'ckpt' records or not present.")
        return

    if 'ckpt' not in download_log:
        download_log['ckpt'] = {}

    # Get all uploaded projects (those marked Success)
    tasks = [f for f, s in upload_log['ckpt'].items() if s == "Success"]
    tasks.sort()

    for folder in tasks:
        if download_log['ckpt'].get(folder) == "Success":
            print(f"Skipping already downloaded: {folder}")
            continue

        success, status = download_folder(folder)
        if success:
            download_log['ckpt'][folder] = "Success"
            save_yaml(DOWNLOAD_LOG_FILE, download_log)
            print(f"Downloaded: {folder}")
        else:
            print(f"Download failed: {folder}, error: {status}")


if __name__ == '__main__':
    main()
