#!/bin/bash

# 定义日志函数
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 项目路径定义 (保持和宿主机一致的绝对路径)
PROJECT_ROOT="/home/xinhai/projects/automoma"
DP3_PATH="${PROJECT_ROOT}/baseline/RoboTwin/policy/DP3"

log "启动自动处理流程，每 30 分钟执行一次..."

while true; do
    log "========== 开始新一轮任务 =========="

    # 1. 解决权限问题 (Docker 内部是 root，可以直接修改映射进来的文件权限)
    # 注意：这里假设你挂载进容器的路径依然是 /home/xinhai/...
    log "正在修正 output 目录权限..."
    if [ -d "${PROJECT_ROOT}/output" ]; then
        chmod -R 777 "${PROJECT_ROOT}/output"
    else
        log "警告: ${PROJECT_ROOT}/output 不存在"
    fi

    # 2. 运行 Python 转换脚本
    log "进入项目根目录: ${PROJECT_ROOT}"
    cd "${PROJECT_ROOT}" || exit 1

    log "步骤 2.1: Python Collect"
    python scripts/convert_automoma_to_dp3.py --mode collect
    
    log "步骤 2.2: Python Convert"
    python scripts/convert_automoma_to_dp3.py --mode convert
    
    log "步骤 2.3: Python Check"
    python scripts/convert_automoma_to_dp3.py --mode check
    
    log "步骤 2.4: Python Clean (删除 hdf5)"
    python scripts/convert_automoma_to_dp3.py --mode clean

    # 3. 运行 Bash 压缩脚本
    log "切换到 DP3 目录: ${DP3_PATH}"
    cd "${DP3_PATH}" || exit 1

    log "步骤 3.1: Compress Convert (打包 zarr)"
    bash scripts/manage_zarr_comp.sh compress convert
    
    log "步骤 3.2: Compress Check (校验 zstd)"
    bash scripts/manage_zarr_comp.sh compress check
    
    log "步骤 3.3: Compress Clean (删除原始 zarr)"
    bash scripts/manage_zarr_comp.sh compress clean

    # 4. Rsync 上传
    # 注意：需要挂载宿主机的 SSH key 到容器的 /root/.ssh
    log "步骤 4: Rsync 上传"
    rsync -avP --remove-source-files -e "ssh -o StrictHostKeyChecking=no" \
        "${DP3_PATH}/data_compressed/" \
        xinhai@192.168.31.227:~/projects/automoma/baseline/RoboTwin/policy/DP3/data_compressed/
    
    if [ $? -eq 0 ]; then
        log "上传并清理本地压缩包成功"
    else
        log "ERR: 上传失败，本地压缩包已保留"
    fi

    log "本轮任务完成。休眠 10 分钟..."
    sleep 600
done