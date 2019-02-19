#!/usr/bin/env bash
PROJECT_DIR=$(pwd)

FOLDER_TO_DRAW="frankfurt"

INPUT_FOLDER="../target_image/"$FOLDER_TO_DRAW
OUTPUT_FOLDER="./result_image/pspnet101_473_cityscapes/"$FOLDER_TO_DRAW

LABELMAP_FILE="./colormapcs.mat"
MODEL_DEF="./model/pspnet101_473.prototxt"
MODEL_WEIGHTS="./model/pspnet101_cityscapes.caffemodel"

IMAGE_SIZE=473

DOCKER_IMAGE=leeesangwon/nvcaffe:19.01-py2.refinedet.pspnet

docker run --runtime=nvidia -it --rm \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ${PROJECT_DIR}:/project \
    ${DOCKER_IMAGE} \
    bash -c "cd /project/segmentation_infer && \
            python segmentation_infer.py $INPUT_FOLDER $OUTPUT_FOLDER \
                --labelmap-file $LABELMAP_FILE \
                --model-def $MODEL_DEF \
                --model-weights $MODEL_WEIGHTS \
                --image-size $IMAGE_SIZE"
