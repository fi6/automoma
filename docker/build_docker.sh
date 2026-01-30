echo "Building Isaac Sim headless docker"
dockerfile="isaac_sim.dockerfile"
image_tag="isaac_sim_4.2.0"
isaac_sim_version="4.2.0"


# build docker file:
# Make sure you enable nvidia runtime by:
# Edit/create the /etc/docker/daemon.json with content:
# {
#    "runtimes": {
#        "nvidia": {
#            "path": "/usr/bin/nvidia-container-runtime",
#            "runtimeArgs": []
#         }
#    },
#    "default-runtime": "nvidia" # ADD this line (the above lines will already exist in your json file)
# }
#
echo "${dockerfile}"

docker build --build-arg ISAAC_SIM_VERSION=${isaac_sim_version} -t automoma_docker:${image_tag} -f docker/${dockerfile} .

# Create logs and output directories for automoma-docker1 to automoma-docker9
# for i in {1..9}; do
#     mkdir -p $(pwd)/logs/automoma-docker-$i
#     mkdir -p $(pwd)/output/automoma-docker-$i
# done

# # build user docker file:
# USER_ID=$(id -g "$USER")
# user_dockerfile=docker/user_isaac_sim.dockerfile

# docker build --build-arg USERNAME=$USER --build-arg USER_ID=${USER_ID} --build-arg IMAGE_TAG=$input_arg -f $user_dockerfile --tag automoma_docker:user_$input_arg .

