# taro-flows

# download
REMOTE_DIR="/home/xinhai/projects/automoma/outputs/train/dp3_single_object_reach_7221_scene_0_seed_0_1000_no_drop/checkpoints/100000/"
LOCAL_DIR=$REMOTE_DIR
mkdir -p ${LOCAL_DIR}
rsync -avP taro-flows:${REMOTE_DIR} ${LOCAL_DIR}