# PyCaffe inference

## How to use

1. Install nvidia graphic driver  
엔비디아 그래픽 드라이버를 설치합니다.

2. Install Docker  
도커를 설치합니다.  
https://docs.docker.com/install/linux/docker-ce/ubuntu/  

3. Install nvidia-docker2  
엔비디아 도커를 설치합니다.  
https://github.com/NVIDIA/nvidia-docker

4. Download *.caffemodel & *.prototxt  
    카페모델과 프로토텍스트 파일을 다운로드 받습니다.

    - Object detection: Move the files to `./detection_infer/model` directory

        - SSD: [caffemodel&prototxt](https://github.com/weiliu89/caffe/tree/ssd#models)  
        Please delete __save_output_param__ parts of the prototxt.  
        프로토텍스트 파일에서 __save_output_param__ 부분을 지워주세요.

            ```prototxt
            save_output_param {
              label_map_file: "data/VOC0712Plus/labelmap_voc.prototxt"
            }
            ```

        - Refindet: [caffemodel&prototxt](https://github.com/sfzhang15/RefineDet#models)

    - Semantic segmentation: Move the files to `./segmentation_infer/model` directory

        - PSPNet: [caffemodel](https://github.com/hszhao/PSPNet/tree/master/evaluation/model), [prototxt](https://github.com/hszhao/PSPNet/tree/master/evaluation/prototxt)
        - ICNet: [caffemodel](https://github.com/hszhao/ICNet/tree/master/evaluation/model), [prototxt](https://github.com/hszhao/ICNet/tree/master/evaluation/prototxt)

5. Move images to process to the `./detection_infer/target_image` or `./segmentation_infer/target_image`  
처리할 이미지들을 `./detection_infer/target_image` 또는 `./segmentation_infer/target_image`에 옮겨줍니다.  

6. Edit shell scripts and run.
    - Object detection

        ```bash
        cd detection_infer
        bash ssd_infer.sh
        bash refinedet_infer.sh
        ```

    - Semantic segmentation

        ```bash
        cd segmentation_infer
        bash pspnet_infer.sh
        bash icnet_infer.sh
        ```

## Caffe code to build docker image

[Github](https://github.com/leeesangwon/NVCaffe)  
It base on NVCaffe and I add some code to support refinedet, pspnet and icnet.