#!/usr/bin/env bash

# https://docs.docker.com/develop/develop-images/build_enhancements/#to-enable-buildkit-builds
export DOCKER_BUILDKIT=1

if [[ -z $1 ]] || [[ -z $2 ]]; then
    echo "Usage: ${0##*/} [name] [linux/amd64|linux/arm64|linux/arm] [tag]" 1>&2
    exit 1
fi

NUMERIC='^[0-9]+$'
BUILD_DATE=$(/bin/date -u +%y%m%d)

# Kill old builder if still alive.
echo "build: removing existing model-docker-multibuilder..."
docker buildx rm model-docker-multibuilder 2>/dev/null

# Wait for 3s.
sleep 3

# Create builder.
docker buildx create --name model-docker-multibuilder --use  || { echo 'failed'; exit 1; }

echo "Starting 'photoprism/model-$1' multi-arch build based on $1/Dockerfile..."
echo "Build Arch: $2"

if [[ $1 ]] && [[ $2 ]] && [[ -z $3 ]]; then
    echo "Build Tags: preview"

    docker buildx build \
      --platform $2 \
      --pull \
      --no-cache \
      --build-arg BUILD_TAG=$BUILD_DATE \
      -f $1/Dockerfile \
      -t photoprism/model-$1:preview \
      --push .
elif [[ $3 =~ $NUMERIC ]]; then
    echo "Build Tags: $3, latest"

    if [[ $4 ]]; then
      echo "Build Params: $4"
    fi

    docker buildx build \
      --platform $2 \
      --pull \
      --no-cache \
      --build-arg BUILD_TAG=$3 \
      -f $1/Dockerfile \
      -t photoprism/model-$1:latest \
      -t photoprism/model-$1:$3 $4 \
      --push .
else
    echo "Build Tags: $3"

    if [[ $4 ]]; then
      echo "Build Params: $4"
    fi

    docker buildx build \
      --platform $2 \
      --pull \
      --no-cache \
      --build-arg BUILD_TAG=$BUILD_DATE \
      -f $1/Dockerfile \
      -t photoprism/model-$1:$3 $4 \
      --push .
fi

echo "Removing model-docker-multibuilder..."
docker buildx rm model-docker-multibuilder

echo "Done."
