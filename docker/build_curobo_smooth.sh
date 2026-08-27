#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_TAG="${AUTOMOMA_DOCKER_IMAGE:-automoma:mw7221-akr-smoothing}"
BASE_IMAGE="${AUTOMOMA_BASE_IMAGE:-automoma:isaacsim5.1-sm86}"
CUROBO_COMMIT="$(git -C "${REPO_ROOT}/third_party/curobo" rev-parse HEAD)"
TRAJECTORY_SMOOTHING_COMMIT="bb1002674e5e80e64091658164e0b3d53f5ac1a4"

docker build \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "CUROBO_COMMIT=${CUROBO_COMMIT}" \
    --build-arg "TRAJECTORY_SMOOTHING_COMMIT=${TRAJECTORY_SMOOTHING_COMMIT}" \
    --tag "${IMAGE_TAG}" \
    --file docker/Dockerfile.curobo-smooth \
    "${REPO_ROOT}"
