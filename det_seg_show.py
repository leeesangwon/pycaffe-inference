# python2.7

from __future__ import print_function
import os
import glob
import argparse

from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

# suprress Caffe verbose prints
os.environ['GLOG_minloglevel'] = '2'
import caffe

from segmentation_infer.network import SegmentationNetwork
from detection_infer.network import DetectionNetwork

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

try:
    FileExistsError
except NameError:
    FileExistsError = OSError


def show_bbox_on_seg_images(input_folder, labelmap_file, model_def, model_weights, image_resize, crop_size, use_center=False):
    input_list = glob.glob(os.path.join(input_folder, "*.jpg"))
    input_list += glob.glob(os.path.join(input_folder, "*.JPG"))
    input_list += glob.glob(os.path.join(input_folder, "*.png"))
    input_list += glob.glob(os.path.join(input_folder, "*.PNG"))
    input_list.sort()

    assert input_list, "input_folder(%s) doesn't have image file(jpeg or png format)" % input_folder

    # get canvas size
    tmp = Image.open(input_list[0])
    h, w = np.asarray(tmp).shape[0:2]
    dpi = int(h/10)
    figsize = (float(w)/dpi, float(h)/dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.show()
    fig.canvas.draw()

    caffe.set_device(0)
    caffe.set_mode_gpu()
    det_network = DetectionNetwork(model_def[0], model_weights[0], labelmap_file[0], image_resize)
    seg_network = SegmentationNetwork(model_def[1], model_weights[1], labelmap_file[1], crop_size)

    for input_image in tqdm(input_list):

        image = caffe.io.load_image(input_image)  # np.float32 in range [0, 1]

        segmentation = seg_network(image)
        detections = det_network(image, use_center)
        segmentation = 0.7 * image + 0.3 * segmentation
        bboxed_img = det_network.bbox_painter(segmentation, detections)
        result = Image.fromarray(bboxed_img)

        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(result)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        fig.canvas.draw()
        ax.clear()
    plt.close(fig)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str)
    parser.add_argument("--labelmap-file", nargs='+', type=str, default="./labelmap_voc.prototxt",
                        help="path to labelmap file")
    parser.add_argument("--model-def", nargs='+', type=str, default="./refinedet_deploy.prototxt",
                        help="path to deploy.prototxt")
    parser.add_argument("--model-weights", nargs='+', type=str, default="./refinedet.caffemodel",
                        help="path to .caffemodel")
    parser.add_argument("--image-size", type=int, default=320,
                        help="size to resize input images")
    parser.add_argument("--use-center", action="store_true",
                        help="if True, detection is conducted again using center region of input image")
    parser.add_argument("--crop-size", nargs='+', type=int, default=473,
                        help="size to resize input images")
    return parser.parse_args()


def main():
    args = get_arguments()

    input_folder = args.input_folder
    labelmap_file_list = args.labelmap_file
    model_def_list = args.model_def
    model_weights_list = args.model_weights
    image_resize = args.image_size
    use_center = args.use_center
    crop_size = args.crop_size

    if not os.path.isdir(input_folder):
        raise FileNotFoundError("input_folder(%s) does not exist" % input_folder)
    for labelmap_file in labelmap_file_list:
        if not os.path.isfile(labelmap_file):
            raise FileNotFoundError("labelmap_file(%s) does not exist" % labelmap_file)
    for model_def in model_def_list:
        if not os.path.isfile(model_def):
            raise FileNotFoundError("model_def(%s) does not exist" % model_def)
    for model_weights in model_weights_list:
        if not os.path.isfile(model_weights):
            raise FileNotFoundError("model_weights(%s) does not exist" % model_weights)

    show_bbox_on_seg_images(
        input_folder=input_folder,
        labelmap_file=labelmap_file_list,
        model_def=model_def_list,
        model_weights=model_weights_list,
        image_resize=image_resize,
        crop_size=crop_size,
        use_center=use_center)


if __name__=="__main__":
    main()

