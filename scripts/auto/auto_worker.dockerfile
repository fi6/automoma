# 使用官方 Python 基础镜像
FROM python:3.10-slim

# 设置非交互前端，防止 apt 卡住
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统工具
# zstd: 用于压缩脚本
# rsync: 用于上传
# openssh-client: rsync 需要 ssh
RUN apt-get update && apt-get install -y \
    zstd \
    rsync \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
# 根据你的脚本 import 内容
RUN pip install --no-cache-dir \
    h5py \
    numpy \
    zarr

# 设置工作目录（虽然脚本里有 cd，但设置一下是个好习惯）
WORKDIR /app

# 默认命令（启动时可以被覆盖，或者直接指定脚本路径）
CMD ["bash"]