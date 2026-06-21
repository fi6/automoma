#!/usr/bin/env bash
set -euo pipefail

export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
export OMNI_KIT_ALLOW_ROOT="${OMNI_KIT_ALLOW_ROOT:-1}"
export AUTOMOMA_ROOT="${AUTOMOMA_ROOT:-/workspace/automoma}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-${AUTOMOMA_ROOT}/third_party/IsaacLab-Arena/submodules/IsaacLab}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export ISAACSIM_ML_PIP="${ISAACSIM_ML_PIP:-/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle}"
export PATH="${CUDA_HOME}/bin:/isaac-sim/kit/python/bin:${PATH}"

# Keep the NGC Isaac Sim filesystem layout compatible with the conda package
# layout expected by IsaacLab-Arena's lighting helper.
if [[ -d /isaac-sim/extscache && -d /isaac-sim/python_packages/isaacsim && ! -e /isaac-sim/python_packages/isaacsim/extscache ]]; then
    ln -s /isaac-sim/extscache /isaac-sim/python_packages/isaacsim/extscache
fi

ISAACSIM_TORCH_LIBS=(
    "${ISAACSIM_ML_PIP}/cusparselt/lib"
    "${ISAACSIM_ML_PIP}/nvidia/cublas/lib"
    "${ISAACSIM_ML_PIP}/nvidia/cuda_cupti/lib"
    "${ISAACSIM_ML_PIP}/nvidia/cuda_nvrtc/lib"
    "${ISAACSIM_ML_PIP}/nvidia/cuda_runtime/lib"
    "${ISAACSIM_ML_PIP}/nvidia/cudnn/lib"
    "${ISAACSIM_ML_PIP}/nvidia/cufft/lib"
    "${ISAACSIM_ML_PIP}/nvidia/cufile/lib"
    "${ISAACSIM_ML_PIP}/nvidia/curand/lib"
    "${ISAACSIM_ML_PIP}/nvidia/cusolver/lib"
    "${ISAACSIM_ML_PIP}/nvidia/cusparse/lib"
    "${ISAACSIM_ML_PIP}/nvidia/nccl/lib"
    "${ISAACSIM_ML_PIP}/nvidia/nvjitlink/lib"
    "${ISAACSIM_ML_PIP}/nvidia/nvtx/lib"
)
LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
for lib_dir in "${ISAACSIM_TORCH_LIBS[@]}"; do
    if [[ -d "${lib_dir}" && ":${LD_LIBRARY_PATH}:" != *":${lib_dir}:"* ]]; then
        LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH}"
    fi
done
export LD_LIBRARY_PATH

# Keep runtime mounts semantically aligned with scripts/run_pipeline.sh.
export AUTOMOMA_OBJECT_ROOT="${AUTOMOMA_OBJECT_ROOT:-${AUTOMOMA_ROOT}/assets/object}"
export AUTOMOMA_SCENE_ROOT="${AUTOMOMA_SCENE_ROOT:-${AUTOMOMA_ROOT}/assets/scene/infinigen/scene_v2}"
export AUTOMOMA_ROBOT_ROOT="${AUTOMOMA_ROBOT_ROOT:-${AUTOMOMA_ROOT}/assets/robot}"

if [[ -d "${AUTOMOMA_ROOT}" ]]; then
    cd "${AUTOMOMA_ROOT}"
fi

exec "$@"
