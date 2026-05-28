#!/usr/bin/env bash

find_docker_cmd() {
    DOCKER_CMD=(docker)
    if docker info >/dev/null 2>&1; then
        return 0
    fi

    if ! command -v sudo >/dev/null 2>&1; then
        echo "Error: Docker daemon is not reachable by user '${USER}'." >&2
        echo "Add this user to the docker group or run Docker with sudo, then retry." >&2
        return 1
    fi

    echo "Docker daemon is not reachable by user '${USER}'; trying sudo." >&2
    if sudo -n docker info >/dev/null 2>&1 || { sudo -v && sudo -E docker info >/dev/null 2>&1; }; then
        DOCKER_CMD=(sudo -E docker)
        return 0
    fi

    echo "Error: Docker daemon is not reachable, even with sudo." >&2
    return 1
}
