#!/bin/bash

set -e

PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

# parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --pip-index-url)
            PIP_INDEX_URL="$2"
            shift 2
            ;;
        --pip-index-url=*)
            PIP_INDEX_URL="${1#*=}"
            shift 1
            ;;
	*)
	    shift 1
	    ;;
    esac
done
echo PIP_INDEX_URL: ${PIP_INDEX_URL}

IMAGE_NAME="soph_vllm_riscv"
IMAGE_TAG="0.7.3"
DOCKERFILE="Dockerfile.sophtpu_riscv"

COMMIT_ID=$(git rev-parse --short HEAD)
DATE=$(date +%Y%m%d)
#TORCH_TPU_COMMIT_ID=$(ls third-party/torch-tpu* | awk -F'[_.]' '{print $(NF-2)}')

# clean docker image
if [ -n "$(docker images -q ${IMAGE_NAME}:${IMAGE_TAG} 2> /dev/null)" ]; then
    docker image rm ${IMAGE_NAME}:${IMAGE_TAG}
fi

# build & export docker image
docker build --build-arg PIP_INDEX_URL=${PIP_INDEX_URL} -f ${DOCKERFILE} -t ${IMAGE_NAME}:${IMAGE_TAG} .
docker save ${IMAGE_NAME}:${IMAGE_TAG} | bzip2 > docker-${IMAGE_NAME}-${IMAGE_TAG}-${DATE}-${COMMIT_ID}-${TORCH_TPU_COMMIT_ID}.tar.bz2

# clean docker iamge
if docker images -q ${IMAGE_NAME}:${IMAGE_TAG}; then
    docker image rm -f ${IMAGE_NAME}:${IMAGE_TAG}
fi
