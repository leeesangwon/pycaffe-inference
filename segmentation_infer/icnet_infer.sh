#!/usr/bin/env bash
PROJECT_DIR=$(pwd)

FOLDER_TO_DRAW="frankfurt"

INPUT_FOLDER="./target_image/"$FOLDER_TO_DRAW
OUTPUT_FOLDER="./result_image/icnet_cityscapes/"$FOLDER_TO_DRAW

LABELMAP_FILE="./colormapcs.mat"
MODEL_DEF="./model/icnet_cityscapes.prototxt"
MODEL_WEIGHTS="./model/icnet_cityscapes_trainval_90k.caffemodel"

IMAGE_SIZE="1025 2049"

DOCKER_IMAGE=leeesangwon/nvcaffe:19.01-py2.refinedet.pspnet

docker run --runtime=nvidia -it --rm \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ${PROJECT_DIR}:/project \
    ${DOCKER_IMAGE} \
    bash -c "cd /project && \
            python segmentation_infer.py $INPUT_FOLDER $OUTPUT_FOLDER \
                --labelmap-file $LABELMAP_FILE \
                --model-def $MODEL_DEF \
                --model-weights $MODEL_WEIGHTS \
                --image-size $IMAGE_SIZE"
