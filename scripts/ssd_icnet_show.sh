#!/usr/bin/env bash
PROJECT_DIR=$(pwd)

FOLDER_TO_DRAW="frankfurt"

INPUT_FOLDER="./target_image/"$FOLDER_TO_DRAW

DET_LABELMAP_FILE="./detection/labelmap_voc.prototxt"
DET_MODEL_DEF="./detection/model/ssd_300.prototxt"
DET_MODEL_WEIGHTS="./detection/model/ssd_300_coco_voc07++12.caffemodel"
DET_IMAGE_SIZE=300


SEG_LABELMAP_FILE="./segmentation_infer/colormapcs.mat"
SEG_MODEL_DEF="./segmentation_infer/model/icnet_cityscapes.prototxt"
SEG_MODEL_WEIGHTS="./segmentation_infer/model/icnet_cityscapes_trainval_90k.caffemodel"
SEG_IMAGE_SIZE="1025 2049"

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
    bash -c "cd /project && \
            python det_seg_show.py $INPUT_FOLDER \
                --labelmap-file $DET_LABELMAP_FILE $SEG_LABELMAP_FILE \
                --model-def $DET_MODEL_DEF $SEG_MODEL_DEF \
                --model-weights $DET_MODEL_WEIGHTS $SEG_MODEL_WEIGHTS \
                --image-size $DET_IMAGE_SIZE \
                --crop-size $SEG_IMAGE_SIZE"
xhost -local:docker
