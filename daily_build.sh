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

IMAGE_NAME="soph_vllm"
IMAGE_TAG="0.11.0"
DOCKERFILE="Dockerfile.sophtpu"
DOCKERFILE_SHA256=$(sha256sum ${DOCKERFILE} | cut -c 1-8)

IMAGE_TAG_PREFIX=${IMAGE_TAG}
remote_branch=$(git branch -vv | grep '^\*' | awk '{print $4}' | cut -d'[' -f2 | cut -d']' -f1)
if [[ "$remote_branch" == "origin/master" || \
   ( -n "$FTP_RELEASE_DIR" && "$FTP_RELEASE_DIR" == *daily_build* ) ]]; then
    echo "当前跟踪的远程分支是 master"
    IMAGE_TAG_PREFIX=${IMAGE_TAG}-$(date +%Y%m%d)
fi

COMMIT_ID=$(git rev-parse --short HEAD)
TORCH_TPU_COMMIT_ID=$(ls third-party/torch-tpu* | awk -F'[_.]' '{print $(NF-2)}')

# clean docker image
if [ -n "$(docker images -q ${IMAGE_NAME}:${IMAGE_TAG}* 2> /dev/null)" ]; then
    docker image rm ${IMAGE_NAME}:${IMAGE_TAG}*
fi

# build & export docker image
docker build --build-arg PIP_INDEX_URL=${PIP_INDEX_URL} -f ${DOCKERFILE} -t ${IMAGE_NAME}:${IMAGE_TAG_PREFIX} .
docker save ${IMAGE_NAME}:${IMAGE_TAG_PREFIX} | bzip2 > docker-${IMAGE_NAME}-${IMAGE_TAG}-${DOCKERFILE_SHA256}-${COMMIT_ID}-${TORCH_TPU_COMMIT_ID}.tar.bz2

# Build docs
DOCKER_IMG_PANDOC=${IMAGE_NAME}:${IMAGE_TAG_PREFIX}
DOCKER_IMG_DOCUMENT=sophgo/torch_tpu:latest

docker run --entrypoint bash --rm -v $(pwd):/workspace/ $DOCKER_IMG_PANDOC -c 'cd /workspace && pandoc --from=markdown --to=rst --output=sophgo_docs/source_zh/readme.rst README_sophtpu.md'
# pandoc --from=markdown --to=rst --output=sophgo_docs/source_zh/readme.rst README.md
docker run --rm -v $(pwd):/workspace/ $DOCKER_IMG_DOCUMENT /bin/bash -c 'cd sophgo_docs/ && make pdf LANG=zh'

# clean docker iamge
if docker images -q ${IMAGE_NAME}:${IMAGE_TAG_PREFIX}; then
    docker image rm -f ${IMAGE_NAME}:${IMAGE_TAG_PREFIX}
fi
