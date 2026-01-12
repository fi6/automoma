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
    if not os.path.exists(local_checkpoints):
        return False, "Local checkpoints not found"

    # Prefer the `last` link if present; resolve it to the actual checkpoint folder.
    last_link = os.path.join(local_checkpoints, "last")
    target_dir = None

    if os.path.exists(last_link):
        # If it's a symlink, resolve directly.
        if os.path.islink(last_link):
            resolved = os.path.realpath(last_link)
            if os.path.isdir(resolved):
                target_dir = resolved
        else:
            # If it's a plain file that contains the name, try to read it.
            try:
                with open(last_link, 'r') as f:
                    name = f.read().strip()
                candidate = os.path.join(local_checkpoints, name)
                if os.path.isdir(candidate):
                    target_dir = candidate
            except Exception:
                target_dir = None

    # Fallback: choose the checkpoint folder with the largest numeric name.
    if target_dir is None:
        candidates = [d for d in os.listdir(local_checkpoints)
                      if os.path.isdir(os.path.join(local_checkpoints, d)) and d != 'last']
        # Prefer numeric step folders
        numeric = []
        for d in candidates:
            try:
                numeric.append((int(d), d))
            except Exception:
                pass
        if numeric:
            _, best = max(numeric)
            target_dir = os.path.join(local_checkpoints, best)
        elif candidates:
            candidates.sort()
            target_dir = os.path.join(local_checkpoints, candidates[-1])

    if target_dir is None or not os.path.isdir(target_dir):
        return False, "No checkpoint folder found to upload"

    last_name = os.path.basename(target_dir)
    print(f"--- Uploading: {folder_name}/checkpoints/{last_name} ---")

    # Upload only the resolved checkpoint folder into remote checkpoints/ path
    remote_checkpoints_path = os.path.join(YUNPAN_DATA_ROOT, folder_name, "checkpoints") + "/"
    cmd = [ALIYUNPAN_PATH, "upload", target_dir, remote_checkpoints_path]

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
