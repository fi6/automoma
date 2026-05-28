#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=docker/common.sh
source "${SCRIPT_DIR}/common.sh"

IMAGE_TAG="${AUTOMOMA_DOCKER_IMAGE:-automoma:main-isaacsim5.1.0}"
CONTAINER_NAME="${AUTOMOMA_DOCKER_NAME:-automoma-main}"
GPU_SPEC="${AUTOMOMA_DOCKER_GPUS:-all}"
SHM_SIZE="${AUTOMOMA_DOCKER_SHM_SIZE:-32g}"
CACHE_ROOT="${AUTOMOMA_DOCKER_CACHE:-${HOME}/.cache/automoma-docker/${CONTAINER_NAME}}"
MOUNT_REPO=1
REMOVE=1
TTY_ARGS=(-it)
EXTRA_DOCKER_ARGS=()
CMD=(bash)

usage() {
    cat <<EOF_USAGE
Usage: bash docker/run_docker.sh [options] [-- command...]

Options:
  --image IMAGE       Docker image tag (default: ${IMAGE_TAG})
  --name NAME         Container name (default: ${CONTAINER_NAME})
  --gpu GPU_SPEC      all, none, or a Docker --gpus device spec such as 0 or 0,1 (default: ${GPU_SPEC})
  --cache DIR         Persistent Isaac/OV/PIP cache root (default: ${CACHE_ROOT})
  --no-mount-repo     Use image source only; assets/data are not baked into the image
  --keep              Do not pass --rm
  --no-tty            Disable interactive TTY flags, useful for CI scripts
  --docker-arg ARG    Extra raw argument passed to docker run; repeatable
  -h, --help          Show this help

Examples:
  bash docker/run_docker.sh
  bash docker/run_docker.sh --gpu 0 -- bash scripts/run_pipeline.sh record microwave_7221 scene_0_seed_0 1 --headless
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            IMAGE_TAG="$2"; shift 2 ;;
        --image=*)
            IMAGE_TAG="${1#*=}"; shift ;;
        --name)
            CONTAINER_NAME="$2"; shift 2 ;;
        --name=*)
            CONTAINER_NAME="${1#*=}"; shift ;;
        --gpu|--gpus)
            GPU_SPEC="$2"; shift 2 ;;
        --gpu=*|--gpus=*)
            GPU_SPEC="${1#*=}"; shift ;;
        --cache)
            CACHE_ROOT="$2"; shift 2 ;;
        --cache=*)
            CACHE_ROOT="${1#*=}"; shift ;;
        --no-mount-repo)
            MOUNT_REPO=0; shift ;;
        --keep)
            REMOVE=0; shift ;;
        --no-tty)
            TTY_ARGS=(); shift ;;
        --docker-arg)
            EXTRA_DOCKER_ARGS+=("$2"); shift 2 ;;
        --docker-arg=*)
            EXTRA_DOCKER_ARGS+=("${1#*=}"); shift ;;
        --help|-h)
            usage; exit 0 ;;
        --)
            shift; CMD=("$@"); break ;;
        *)
            CMD=("$@"); break ;;
    esac
done

find_docker_cmd

mkdir -p \
    "${CACHE_ROOT}/kit" \
    "${CACHE_ROOT}/ov" \
    "${CACHE_ROOT}/pip" \
    "${CACHE_ROOT}/glcache" \
    "${CACHE_ROOT}/computecache" \
    "${CACHE_ROOT}/omniverse-logs" \
    "${CACHE_ROOT}/ov-data" \
    "${CACHE_ROOT}/documents"

DOCKER_ARGS=(
    --name "${CONTAINER_NAME}"
    --shm-size "${SHM_SIZE}"
    --network host
    --ipc host
    --ulimit memlock=-1
    --ulimit stack=67108864
    --privileged
    -e ACCEPT_EULA=Y
    -e PRIVACY_CONSENT=Y
    -e OMNI_KIT_ALLOW_ROOT=1
    -e NVIDIA_DRIVER_CAPABILITIES=all
    -e WANDB_MODE="${WANDB_MODE:-offline}"
    -v /dev:/dev
    -v "${CACHE_ROOT}/kit:/isaac-sim/kit/cache:rw"
    -v "${CACHE_ROOT}/ov:/root/.cache/ov:rw"
    -v "${CACHE_ROOT}/pip:/root/.cache/pip:rw"
    -v "${CACHE_ROOT}/glcache:/root/.cache/nvidia/GLCache:rw"
    -v "${CACHE_ROOT}/computecache:/root/.nv/ComputeCache:rw"
    -v "${CACHE_ROOT}/omniverse-logs:/root/.nvidia-omniverse/logs:rw"
    -v "${CACHE_ROOT}/ov-data:/root/.local/share/ov/data:rw"
    -v "${CACHE_ROOT}/documents:/root/Documents:rw"
)

if [[ "${REMOVE}" == "1" ]]; then
    DOCKER_ARGS+=(--rm)
fi

if [[ "${GPU_SPEC}" != "none" ]]; then
    if [[ "${GPU_SPEC}" == "all" ]]; then
        DOCKER_ARGS+=(--gpus all)
    else
        DOCKER_ARGS+=(--gpus "device=${GPU_SPEC}")
    fi
fi

if [[ "${MOUNT_REPO}" == "1" ]]; then
    DOCKER_ARGS+=(-v "${REPO_ROOT}:/workspace/automoma:rw")
fi

if [[ -n "${DISPLAY:-}" ]]; then
    DOCKER_ARGS+=(-e "DISPLAY=${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
    if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
        DOCKER_ARGS+=(-e "XAUTHORITY=/tmp/.docker.xauth" -v "${XAUTHORITY}:/tmp/.docker.xauth:ro")
    fi
fi

DOCKER_ARGS+=("${EXTRA_DOCKER_ARGS[@]}")

exec "${DOCKER_CMD[@]}" run "${TTY_ARGS[@]}" "${DOCKER_ARGS[@]}" "${IMAGE_TAG}" "${CMD[@]}"
