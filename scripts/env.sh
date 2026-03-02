# lerobot-arena
# 1. Create conda environment
conda create -y -n lerobot-arena python=3.11
conda activate lerobot-arena
conda install -y -c conda-forge ffmpeg=7.1.1
conda install -y -c conda-forge "libstdcxx-ng>=15" "libgcc-ng>=15"

# 2. Install Isaac Sim 5.1.0
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Accept NVIDIA EULA (required)
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

cd third_party/curobo
pip install -e .[isaacsim] --no-build-isolation
pip install nest_asyncio==1.5.6 packaging==23.0 scipy==1.15.3 tornado==6.5.1

# 3. Install IsaacLab Arena and its dependencies
git clone https://github.com/isaac-sim/IsaacLab-Arena.git
cd IsaacLab-Arena
git checkout release/0.1.1

git submodule init "submodules/IsaacLab"
git submodule update --init --recursive submodules/IsaacLab
cd submodules/IsaacLab
git checkout v2.3.0
./isaaclab.sh -i
cd ../..

pip install -e .
cd ..


# 4. Install LeRobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
cd ..

# 5. Install additional dependencies
pip install onnxruntime==1.23.2 lightwheel-sdk==1.0.1 vuer[all]==0.0.70 qpsolvers==4.8.1 yourdfpy usd-core
pip install numpy==1.26.0 # Isaac Sim 5.1 depends on numpy==1.26.0, this will be fixed in next release



