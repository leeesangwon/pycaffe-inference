#!/usr/bin/env bash
PROJECT_DIR=$(pwd)

FOLDER_TO_DRAW="frankfurt"

INPUT_FOLDER="../target_image/"$FOLDER_TO_DRAW

LABELMAP_FILE="./labelmap_voc.prototxt"
MODEL_DEF="./model/refinedet_320.prototxt"
MODEL_WEIGHTS="./model/refinedet_320_coco_voc07++12.caffemodel"

IMAGE_SIZE=320

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
    bash -c "cd /project/detection_infer && \
            python detection_show.py $INPUT_FOLDER \
                --labelmap-file $LABELMAP_FILE \
                --model-def $MODEL_DEF \
                --model-weights $MODEL_WEIGHTS \
                --image-size $IMAGE_SIZE"
xhost -local:docker
