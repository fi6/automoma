#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=docker/common.sh
source "${SCRIPT_DIR}/common.sh"

IMAGE_TAG="${AUTOMOMA_DOCKER_IMAGE:-automoma:main-isaacsim5.1.0}"
ISAACSIM_VERSION="${ISAACSIM_VERSION:-5.1.0}"
ISAACSIM_BASE_IMAGE="${ISAACSIM_BASE_IMAGE:-nvcr.io/nvidia/isaac-sim}"
CUDA_TOOLKIT_VERSION="${CUDA_TOOLKIT_VERSION:-12-8}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9+PTX}"
CUROBO_PRETEND_VERSION="${CUROBO_PRETEND_VERSION:-0.7.4.post1.dev6}"

usage() {
    cat <<EOF_USAGE
Usage: bash docker/build_docker.sh [--tag IMAGE] [--isaacsim VERSION] [--arch TORCH_CUDA_ARCH_LIST] [--no-cache] [extra docker build args...]

Environment overrides:
  AUTOMOMA_DOCKER_IMAGE     default: automoma:main-isaacsim5.1.0
  ISAACSIM_VERSION          default: 5.1.0
  ISAACSIM_BASE_IMAGE       default: nvcr.io/nvidia/isaac-sim
  CUDA_TOOLKIT_VERSION      default: 12-8
  TORCH_CUDA_ARCH_LIST      default: 8.9+PTX (RTX 4090)
EOF_USAGE
}

BUILD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag|-t)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --tag=*)
            IMAGE_TAG="${1#*=}"
            shift
            ;;
        --isaacsim)
            ISAACSIM_VERSION="$2"
            shift 2
            ;;
        --isaacsim=*)
            ISAACSIM_VERSION="${1#*=}"
            shift
            ;;
        --arch)
            TORCH_CUDA_ARCH_LIST="$2"
            shift 2
            ;;
        --arch=*)
            TORCH_CUDA_ARCH_LIST="${1#*=}"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            BUILD_ARGS+=("$1")
            shift
            ;;
    esac
done

find_docker_cmd

cd "${REPO_ROOT}"
export DOCKER_BUILDKIT=1

echo "Building AutoMoMa Docker image: ${IMAGE_TAG}"
echo "  Isaac Sim: ${ISAACSIM_BASE_IMAGE}:${ISAACSIM_VERSION}"
echo "  CUDA toolkit: ${CUDA_TOOLKIT_VERSION}"
echo "  TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"

"${DOCKER_CMD[@]}" build \
    --build-arg "ISAACSIM_VERSION=${ISAACSIM_VERSION}" \
    --build-arg "ISAACSIM_BASE_IMAGE=${ISAACSIM_BASE_IMAGE}" \
    --build-arg "CUDA_TOOLKIT_VERSION=${CUDA_TOOLKIT_VERSION}" \
    --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
    --build-arg "CUROBO_PRETEND_VERSION=${CUROBO_PRETEND_VERSION}" \
    -t "${IMAGE_TAG}" \
    -f docker/Dockerfile \
    "${BUILD_ARGS[@]}" \
    .

echo "Built ${IMAGE_TAG}"
