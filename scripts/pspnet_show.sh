#!/usr/bin/env bash
PROJECT_DIR=$(pwd)

FOLDER_TO_DRAW="frankfurt"

INPUT_FOLDER="../target_image/"$FOLDER_TO_DRAW

LABELMAP_FILE="./colormapcs.mat"
MODEL_DEF="./model/pspnet101_473.prototxt"
MODEL_WEIGHTS="./model/pspnet101_cityscapes.caffemodel"

IMAGE_SIZE=473

DOCKER_IMAGE=leeesangwon/nvcaffe:19.01-py2.refinedet.pspnet

xhost +local:docker
docker run --runtime=nvidia -it --rm \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ${PROJECT_DIR}:/project \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=unix$DISPLAY \
    ${DOCKER_IMAGE} \
    bash -c "cd /project/segmentation_infer && \
            python segmentation_show.py $INPUT_FOLDER \
                --labelmap-file $LABELMAP_FILE \
                --model-def $MODEL_DEF \
                --model-weights $MODEL_WEIGHTS \
                --image-size $IMAGE_SIZE"
xhost -local:docker
