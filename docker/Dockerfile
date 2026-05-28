# syntax=docker/dockerfile:1.7
# AutoMoMa Docker image for planning, IsaacLab-Arena recording/eval, and LeRobot conversion/training.
# The image follows the cvpr26-style Isaac Sim base image, updated to Isaac Sim 5.1.0,
# CUDA 12.8, Python 3.11, torch 2.7.0+cu128, and the current automoma conda runtime pins.

ARG ISAACSIM_VERSION=5.1.0
ARG ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim
FROM ${ISAACSIM_BASE_IMAGE}:${ISAACSIM_VERSION} AS automoma

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_TOOLKIT_VERSION=12-8
ARG TORCH_CUDA_ARCH_LIST="8.9+PTX"
ARG CUROBO_PRETEND_VERSION="0.7.4.post1.dev6"
ARG AUTOMOMA_ROOT=/workspace/automoma
ARG ISAACLAB_PATH=/workspace/automoma/third_party/IsaacLab-Arena/submodules/IsaacLab

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ALLOW_ROOT=1 \
    ISAACSIM_ROOT_PATH=/isaac-sim \
    AUTOMOMA_ROOT=${AUTOMOMA_ROOT} \
    ISAACLAB_PATH=${ISAACLAB_PATH} \
    CUDA_HOME=/usr/local/cuda-12.8 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PIP_NO_CACHE_DIR=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV PATH=${CUDA_HOME}/bin:/isaac-sim/kit/python/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

USER root
WORKDIR ${AUTOMOMA_ROOT}

# System packages for building cuRobo, IsaacLab native deps, headless rendering,
# and video conversion.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        git-lfs \
        gnupg \
        libegl1 \
        libfontconfig1 \
        libfreetype6 \
        libgl1 \
        libglib2.0-0 \
        libglvnd0 \
        libgomp1 \
        libsm6 \
        libx11-6 \
        libxau6 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-sync1 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxcb-xinput0 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxdmcp6 \
        libxext6 \
        libxfixes3 \
        libxi6 \
        libxinerama1 \
        libxkbcommon-x11-0 \
        libxrandr2 \
        libxrender1 \
        libxt6 \
        libxtst6 \
        pkg-config \
        python3 \
        python3-pip \
        rsync \
        unzip \
        vulkan-tools \
        wget \
        xauth && \
    rm -rf /var/lib/apt/lists/*

# The Isaac Sim image does not include nvcc. cuRobo needs CUDA toolkit headers and nvcc.
RUN --mount=type=cache,target=/var/cache/apt \
    set -euo pipefail && \
    . /etc/os-release && \
    case "$VERSION_ID" in \
        "20.04") cuda_repo="ubuntu2004" ;; \
        "22.04") cuda_repo="ubuntu2204" ;; \
        "24.04") cuda_repo="ubuntu2404" ;; \
        *) echo "Unsupported Ubuntu version: $VERSION_ID" >&2; exit 1 ;; \
    esac && \
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${cuda_repo}/x86_64/cuda-keyring_1.1-1_all.deb" && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm -f cuda-keyring_1.1-1_all.deb && \
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${cuda_repo}/x86_64/cuda-${cuda_repo}.pin" && \
    mv "cuda-${cuda_repo}.pin" /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-get update && \
    apt-get install -y --no-install-recommends "cuda-toolkit-${CUDA_TOOLKIT_VERSION}" && \
    apt-get -y autoremove && apt-get clean && rm -rf /var/lib/apt/lists/* && \
    test -x "${CUDA_HOME}/bin/nvcc"

# Make scripts that call `python`/`pip` use Isaac Sim's Python. Console scripts installed
# by pip live in /isaac-sim/kit/python/bin, which is already prepended to PATH.
RUN printf '#!/usr/bin/env bash\nexec /isaac-sim/python.sh "$@"\n' > /usr/local/bin/python && \
    printf '#!/usr/bin/env bash\nexec /isaac-sim/python.sh "$@"\n' > /usr/local/bin/python3 && \
    printf '#!/usr/bin/env bash\nexec /isaac-sim/python.sh -m pip "$@"\n' > /usr/local/bin/pip && \
    printf '#!/usr/bin/env bash\nexec /isaac-sim/python.sh -m pip "$@"\n' > /usr/local/bin/pip3 && \
    chmod +x /usr/local/bin/python /usr/local/bin/python3 /usr/local/bin/pip /usr/local/bin/pip3

# Copy source after the heavyweight base/system layers so dependency layers can be reused.
COPY . ${AUTOMOMA_ROOT}

# IsaacLab expects _isaac_sim to resolve to the Isaac Sim root when using the bundled python.sh.
# The working conda Isaac Sim package exposes extscache under site-packages/isaacsim.
# The NGC container keeps those extensions at /isaac-sim/extscache, while
# isaacsim.__file__ points at /isaac-sim/python_packages/isaacsim. Mirror that
# layout so IsaacLab-Arena's headless Grey_Studio light rig resolves in Docker.
RUN ln -sfn /isaac-sim ${ISAACLAB_PATH}/_isaac_sim && \
    if [[ ! -e /isaac-sim/python_packages/isaacsim/extscache ]]; then \
        ln -s /isaac-sim/extscache /isaac-sim/python_packages/isaacsim/extscache; \
    fi && \
    chmod +x ${ISAACLAB_PATH}/isaaclab.sh && \
    chmod +x ${AUTOMOMA_ROOT}/scripts/run_pipeline.sh

# Python build tooling, torch cu128, and AutoMoMa runtime/training pins matching the working
# local automoma conda environment. LeRobot is installed --no-deps below to avoid its newer
# pyproject metadata upgrading NumPy/HF Hub beyond the tested Isaac Sim stack.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -U pip && \
    python -m pip install \
        flit_core \
        ninja \
        tomli \
        wheel \
        "setuptools>=71.0.0,<81.0.0" \
        "setuptools-scm>=9.2.0" && \
    python -m pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.7.0" \
        "torchvision==0.22.0" \
        "torchaudio==2.7.0" && \
    python -m pip install -r ${AUTOMOMA_ROOT}/docker/requirements-automoma.txt && \
    python -m pip install --no-deps rerun-sdk==0.26.2 && \
    python -m pip install -e ${AUTOMOMA_ROOT} --no-deps

# Install IsaacLab source extensions for record/eval, then IsaacLab-Arena.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-build-isolation flatdict==4.0.1 && \
    ${ISAACLAB_PATH}/isaaclab.sh --install none && \
    python -m pip install -e ${AUTOMOMA_ROOT}/third_party/IsaacLab-Arena --no-deps

# Install LeRobot entrypoints against the pinned dependency set, then build cuRobo from the
# checked-out third_party source. cuRobo is intentionally installed non-editable so
# runtime bind-mounts do not hide the compiled CUDA extensions.
ENV ISAACSIM_ML_PIP=/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle
ENV LD_LIBRARY_PATH=${ISAACSIM_ML_PIP}/cusparselt/lib:${ISAACSIM_ML_PIP}/nvidia/cublas/lib:${ISAACSIM_ML_PIP}/nvidia/cuda_cupti/lib:${ISAACSIM_ML_PIP}/nvidia/cuda_nvrtc/lib:${ISAACSIM_ML_PIP}/nvidia/cuda_runtime/lib:${ISAACSIM_ML_PIP}/nvidia/cudnn/lib:${ISAACSIM_ML_PIP}/nvidia/cufft/lib:${ISAACSIM_ML_PIP}/nvidia/cufile/lib:${ISAACSIM_ML_PIP}/nvidia/curand/lib:${ISAACSIM_ML_PIP}/nvidia/cusolver/lib:${ISAACSIM_ML_PIP}/nvidia/cusparse/lib:${ISAACSIM_ML_PIP}/nvidia/nccl/lib:${ISAACSIM_ML_PIP}/nvidia/nvjitlink/lib:${ISAACSIM_ML_PIP}/nvidia/nvtx/lib:${LD_LIBRARY_PATH}
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps "huggingface-hub==0.35.3" "packaging==23.0" && \
    python -m pip install -e ${AUTOMOMA_ROOT}/third_party/lerobot --no-deps && \
    python -c 'import torch; import torch._vendor.packaging._structures; print(f"torch_import_ok={torch.__version__} cuda={torch.version.cuda}")' && \
    (find ${AUTOMOMA_ROOT}/third_party/curobo/src/curobo/curobolib -type f -name '*.so' -delete || true) && \
    SETUPTOOLS_SCM_PRETEND_VERSION=${CUROBO_PRETEND_VERSION} \
        python -m pip install "${AUTOMOMA_ROOT}/third_party/curobo[isaacsim]" --no-build-isolation && \
    python -m pip install --no-deps "packaging==23.0" "warp-lang==1.12.1" "huggingface-hub==0.35.3"

# Persistent cache targets used by docker/run_docker.sh.
RUN mkdir -p \
        /isaac-sim/kit/cache \
        /root/.cache/ov \
        /root/.cache/pip \
        /root/.cache/nvidia/GLCache \
        /root/.nv/ComputeCache \
        /root/.nvidia-omniverse/logs \
        /root/.local/share/ov/data \
        /root/Documents \
        ${AUTOMOMA_ROOT}/data \
        ${AUTOMOMA_ROOT}/logs \
        ${AUTOMOMA_ROOT}/outputs && \
    python ${AUTOMOMA_ROOT}/docker/check_env.py

COPY docker/entrypoint.sh /opt/automoma/entrypoint.sh
RUN chmod +x /opt/automoma/entrypoint.sh && \
    echo "export AUTOMOMA_ROOT=${AUTOMOMA_ROOT}" >> /root/.bashrc && \
    echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> /root/.bashrc && \
    echo "export CUDA_HOME=${CUDA_HOME}" >> /root/.bashrc && \
    echo "alias python=/usr/local/bin/python" >> /root/.bashrc && \
    echo "alias pip=/usr/local/bin/pip" >> /root/.bashrc && \
    echo "cd ${AUTOMOMA_ROOT}" >> /root/.bashrc

WORKDIR ${AUTOMOMA_ROOT}
ENTRYPOINT ["/opt/automoma/entrypoint.sh"]
CMD ["bash"]
