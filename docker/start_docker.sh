#!/bin/bash
##
## Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
##
## NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
## property and proprietary rights in and to this material, related
## documentation and any modifications thereto. Any use, reproduction,
## disclosure or distribution of this material and related documentation
## without an express license agreement from NVIDIA CORPORATION or
## its affiliates is strictly prohibited.
##
mkdir -p $(pwd)/output/automoma-docker-$1
mkdir -p $(pwd)/logs/automoma-docker-$1


docker run --name automoma_$1 --entrypoint bash -it --rm \
    --gpus all \
    --mount type=bind,source=$(pwd)/scripts,target=/pkgs/automoma-docker/scripts,readonly=true \
    --mount type=bind,source=$(pwd)/assets,target=/pkgs/automoma-docker/assets,readonly=true \
    --volume $(pwd)/output/automoma-docker-$1:/pkgs/automoma-docker/output:rw \
    --volume $(pwd)/logs/automoma-docker-$1:/pkgs/automoma-docker/logs:rw \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y"  \
    -v ~/docker/automoma-docker-$1/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/automoma-docker-$1/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/automoma-docker-$1/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/automoma-docker-$1/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/automoma-docker-$1/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/automoma-docker-$1/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/automoma-docker-$1/data:/root/.local/share/ov/data:rw \
    -v ~/docker/automoma-docker-$1/documents:/root/Documents:rw \
    --volume /dev:/dev \
    automoma_docker:isaac_sim_4.2.0