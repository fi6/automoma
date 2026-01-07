import time
import shutil
import numpy as np
import sys
import os
import logging
import subprocess
from pathlib import Path

# os.environ["HF_LEROBOT_HOME"] = "/dev/shm/lerobot_buffer"
os.environ["TMPDIR"] = "/dev/shm/lerobot_tmp"
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

# --- 环境配置 ---
# (保持不变)
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path: sys.path.append(str(src_path))
lerobot_src = project_root / "third_party" / "lerobot" / "src"
if str(lerobot_src) not in sys.path: sys.path.append(str(lerobot_src))

try:
    from automoma.datasets.dataset import LeRobotDatasetWrapper
except ImportError as e:
    print(f"Error: {e}"); sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 超参数 ---
NUM_EPISODES = 10          
STEPS_PER_EPISODE = 32 
FPS = 15               
WIDTH, HEIGHT = 320, 240
CAMERAS = ["ego_topdown", "ego_wrist", "fix_local"] 
STATE_NAMES = ["base_x", "base_y", "base_theta", "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7", "panda_finger_joint1", "panda_finger_joint2"]
STATE_DIM = len(STATE_NAMES)

# 目标设备名
DISK_DEVICE = "nvme0n1" 

# --- 测试路径配置 ---
DISK_PATH = project_root / "data" / "test_io_performance"
SHM_PATH = Path("/dev/shm/lerobot_buffer/test_io_performance")

LOG_DIR = Path("./io_logs")
LOG_DIR.mkdir(exist_ok=True)

# (Config 类, get_config 函数, generate_mock_data 函数, parse_iostat_log 函数保持不变)
class Config:
    def __init__(self, d):
        for k, v in d.items(): 
            setattr(self, k, Config(v) if isinstance(v, dict) else v)
    def get(self, k, default=None): return getattr(self, k, default)

def get_config(root_dir, repo_id):
    return Config({
        "root": str(root_dir),
        "repo_id": repo_id,
        "use_ramdisk": False,
        "task": "benchmark",
        "state_dim": STATE_DIM,
        "state_names": STATE_NAMES,
        "fps": FPS,
        "robot_type": "summit_franka",
        "use_videos": True,
        "push_to_hub": False,
        "camera": {
            "names": CAMERAS, "width": WIDTH, "height": HEIGHT, 
            "cameras": {n: {"depth": False, "pointcloud": False} for n in CAMERAS}
        }
    })

def generate_mock_data():
    return {
        "obs_data": {
            "images": {c: np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8) for c in CAMERAS},
            "depth": {}, "pointcloud": {}
        },
        "joint_data": np.random.randn(STATE_DIM).astype(np.float32),
        "eef_pose_data": np.random.randn(7).astype(np.float32),
        "action_data": np.random.randn(STATE_DIM).astype(np.float32)
    }

def parse_iostat_log(log_file):
    results = {"w/s": [], "wMB/s": [], "w_await": [], "util": []}
    if not Path(log_file).exists(): return results
    with open(log_file, "r") as f:
        lines = f.readlines()
    col_map = {}
    for line in lines:
        parts = line.split()
        if not parts: continue
        if "Device" in parts:
            col_map = {name: i for i, name in enumerate(parts)}
            continue
        if DISK_DEVICE in parts and col_map:
            try:
                results["w/s"].append(float(parts[col_map["w/s"]]))
                if "wMB/s" in col_map: results["wMB/s"].append(float(parts[col_map["wMB/s"]]))
                elif "wkB/s" in col_map: results["wMB/s"].append(float(parts[col_map["wkB/s"]]) / 1024.0)
                results["w_await"].append(float(parts[col_map.get("w_await", col_map.get("await", 0))]))
                results["util"].append(float(parts[col_map["%util"]]))
            except: continue
    return {
        "total_requests": sum(results["w/s"]),
        "peak_requests": max(results["w/s"]) if results["w/s"] else 0,
        "avg_throughput": sum(results["wMB/s"])/len(results["wMB/s"]) if results["wMB/s"] else 0,
        "avg_latency": sum(results["w_await"])/len(results["w_await"]) if results["w_await"] else 0,
        "max_util": max(results["util"]) if results["util"] else 0
    }

def run_baseline(duration=15):
    """
    对照组：仅运行 iostat 监控，不进行任何 LeRobot 录制。
    用于测量系统背景 IO。
    """
    mode_name = "System_Idle"
    log_file = LOG_DIR / f"iostat_{mode_name}.log"
    
    monitor_proc = subprocess.Popen(
        ["iostat", "-xd", "1", "-y"], 
        stdout=open(log_file, "w"), stderr=subprocess.DEVNULL
    )
    
    logger.info(f"--- 正在执行 {mode_name} 测试 (持续 {duration}s) ---")
    start_time = time.time()
    time.sleep(duration) # 仅等待，不进行 IO 操作
    
    monitor_proc.terminate()
    monitor_proc.wait()
    
    stats = parse_iostat_log(log_file)
    stats.update({"time": time.time() - start_time, "size": 0, "mode": mode_name})
    return stats

def run_test(target_root, mode_name):
    # (原有逻辑保持不变)
    target_root.mkdir(parents=True, exist_ok=True)
    repo_id = f"test_io_{mode_name}"
    cfg = get_config(target_root, repo_id)
    log_file = LOG_DIR / f"iostat_{mode_name}.log"
    
    monitor_proc = subprocess.Popen(
        ["iostat", "-xd", "1", "-y"], 
        stdout=open(log_file, "w"), stderr=subprocess.DEVNULL
    )
    
    logger.info(f"--- 正在执行 {mode_name} 测试 | 路径: {target_root} ---")
    start_time = time.time()
    
    try:
        dataset = LeRobotDatasetWrapper(cfg)
        dataset.create()
        for ep in range(NUM_EPISODES):
            for _ in range(STEPS_PER_EPISODE):
                dataset.add(generate_mock_data())
            dataset.save()
        dataset.close()
    finally:
        monitor_proc.terminate()
        monitor_proc.wait()

    duration = time.time() - start_time
    stats = parse_iostat_log(log_file)
    final_path = target_root / repo_id
    size_mb = sum(f.stat().st_size for f in final_path.glob('**/*') if f.is_file()) / (1024*1024)
    if final_path.exists(): shutil.rmtree(final_path)

    stats.update({"time": duration, "size": size_mb, "mode": mode_name})
    return stats

if __name__ == "__main__":
    print(f"硬件 IO 基准测试 | 监控物理设备: {DISK_DEVICE}\n")

    # 1. 对照组：系统背景 IO (15秒)
    res_idle = run_baseline(15)
    
    time.sleep(2) 

    # 2. 测试组 1: 物理磁盘 (Project Data)
    res_disk = run_test(DISK_PATH, "DISK_Path")
    
    time.sleep(2) 
    
    # 3. 测试组 2: 内存盘 (/dev/shm)
    res_shm = run_test(SHM_PATH, "SHM_Path")

    print("\n" + "="*105)
    print(f"{'IO 压力统计全报告 (Baseline vs Disk vs Ram)':^105}")
    print("="*105)
    print(f"{'指标 (Metric)':<30} | {'系统闲置(15s)':<18} | {'物理磁盘路径':<18} | {'/dev/shm 路径':<18}")
    print("-" * 105)
    print(f"{'测试总耗时 (s)':<30} | {res_idle['time']:<18.2f} | {res_disk['time']:<18.2f} | {res_shm['time']:<18.2f}")
    print(f"{'程序写入数据量 (MB)':<30} | {res_idle['size']:<18.2f} | {res_disk['size']:<18.2f} | {res_shm['size']:<18.2f}")
    print(f"{'物理盘总请求数(IO Count)':<30} | {res_idle['total_requests']:<18.0f} | {res_disk['total_requests']:<18.0f} | {res_shm['total_requests']:<18.0f}")
    print(f"{'物理盘峰值(req/s)':<30} | {res_idle['peak_requests']:<18.2f} | {res_disk['peak_requests']:<18.2f} | {res_shm['peak_requests']:<18.2f}")
    print(f"{'物理盘平均延迟(ms)':<30} | {res_idle['avg_latency']:<18.2f} | {res_disk['avg_latency']:<18.2f} | {res_shm['avg_latency']:<18.2f}")
    print(f"{'物理盘利用率(%util)':<30} | {res_idle['max_util']:<18.1f} | {res_disk['max_util']:<18.1f} | {res_shm['max_util']:<18.1f}")
    print("="*105)
    print("分析：如果 '/dev/shm' 的 IO Count 与 '系统闲置' 接近，说明 LeRobot 的写入确实被内存拦截。")