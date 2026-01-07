import time
import os
import subprocess
from pathlib import Path

# --- 配置 ---
DISK_DEVICE = "nvme0n1"  # 你的硬盘设备名
TEST_SIZE_MB = 500       # 写入 500MB 数据
CHUNK_SIZE = 1024 * 1024 # 1MB 缓冲区
LOG_DIR = Path("./io_pure_logs")
LOG_DIR.mkdir(exist_ok=True)

# 定义路径
DISK_ROOT = Path("./data_test_pure")
SHM_ROOT = Path("/dev/shm/pure_test")

def parse_iostat_log(log_file):
    # 这里的 key 和结构现在与 LeRobot 脚本完全一致
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
                if "wMB/s" in col_map: 
                    results["wMB/s"].append(float(parts[col_map["wMB/s"]]))
                elif "wkB/s" in col_map: 
                    results["wMB/s"].append(float(parts[col_map["wkB/s"]]) / 1024.0)
                
                # 增加 w_await (latency)
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

def run_pure_io_test(target_path, mode):
    log_file = LOG_DIR / f"iostat_{mode}.log"
    data_file = target_path / "test_payload.bin"
    target_path.mkdir(parents=True, exist_ok=True)

    # 启动监控
    monitor = subprocess.Popen(["iostat", "-xd", "1", "-y"], stdout=open(log_file, "w"))
    
    start_time = time.time()
    size_mb = 0.0
    
    if mode != "Idle":
        print(f"--- 正在执行 {mode} 测试 | 路径: {target_path} ---")
        data = os.urandom(CHUNK_SIZE) 
        with open(data_file, "wb") as f:
            for _ in range(TEST_SIZE_MB):
                f.write(data)
                f.flush()
                os.fsync(f.fileno()) 
        size_mb = TEST_SIZE_MB
    else:
        print(f"--- 正在执行 {mode} 测试 (持续 15s) ---")
        time.sleep(15)

    duration = time.time() - start_time
    monitor.terminate()
    monitor.wait()

    # 清理数据
    if data_file.exists(): data_file.unlink()
    
    stats = parse_iostat_log(log_file)
    stats.update({"time": duration, "size": float(size_mb), "mode": mode})
    return stats

if __name__ == "__main__":
    print(f"硬件 IO 基准测试 | 监控物理设备: {DISK_DEVICE}\n")

    # 1. 闲置 (保持与对比脚本一致的 15s)
    res_idle = run_pure_io_test(Path("/tmp"), "Idle")
    time.sleep(2)
    
    # 2. 物理磁盘
    res_disk = run_pure_io_test(DISK_ROOT, "DISK_Path")
    time.sleep(2)
    
    # 3. 内存盘
    res_shm = run_pure_io_test(SHM_ROOT, "SHM_Path")

    # 打印格式与 LeRobot 脚本完全一致
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
    print("分析：如果 '/dev/shm' 的 IO Count 与 '系统闲置' 接近，说明纯写入确实被内存拦截。")