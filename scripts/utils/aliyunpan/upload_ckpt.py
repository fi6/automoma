import os
import subprocess
import yaml

# --- Configuration ---
LOCAL_CKPT_ROOT = "/home/xinhai/projects/automoma/outputs/train"
YUNPAN_DATA_ROOT = "/Research/automoma/ckpt"

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "upload_ckpt_log.yaml")
ALIYUNPAN_PATH = "/home/xinhai/env/aliyunpan-v0.3.7-linux-amd64/aliyunpan"


def load_log(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def save_log(path, data):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def upload_folder(folder_name):
    local_checkpoints = os.path.join(LOCAL_CKPT_ROOT, folder_name, "checkpoints")
    remote_path = os.path.join(YUNPAN_DATA_ROOT, folder_name) + "/"

    if not os.path.exists(local_checkpoints):
        return False, "Local checkpoints not found"

    print(f"--- Uploading: {folder_name} (checkpoints/) ---")
    cmd = [ALIYUNPAN_PATH, "upload", local_checkpoints, remote_path]

    try:
        subprocess.run(cmd, check=True)
        return True, "Success"
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e}"


def main():
    upload_log = load_log(LOG_FILE)
    if not isinstance(upload_log, dict):
        upload_log = {}

    # Ensure top-level dict exists
    if 'ckpt' not in upload_log:
        upload_log['ckpt'] = {}

    if not os.path.exists(LOCAL_CKPT_ROOT):
        print(f"Error: local root {LOCAL_CKPT_ROOT} does not exist")
        return

    # Find all project folders containing a checkpoints/ subdir
    folders = [f for f in os.listdir(LOCAL_CKPT_ROOT)
               if os.path.isdir(os.path.join(LOCAL_CKPT_ROOT, f))
               and os.path.isdir(os.path.join(LOCAL_CKPT_ROOT, f, 'checkpoints'))]

    folders.sort()

    for folder in folders:
        if upload_log['ckpt'].get(folder) == "Success":
            print(f"Skipping already uploaded: {folder}")
            continue

        success, status = upload_folder(folder)
        if success:
            upload_log['ckpt'][folder] = "Success"
            save_log(LOG_FILE, upload_log)
            print(f"Uploaded: {folder}")
        else:
            print(f"Failed: {folder}, reason: {status}")


if __name__ == '__main__':
    main()
