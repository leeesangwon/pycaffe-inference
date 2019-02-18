# python2.7

from __future__ import print_function
import os
import glob
import argparse

from tqdm import tqdm
from PIL import Image

# suprress Caffe verbose prints
os.environ['GLOG_minloglevel'] = '2'
import caffe

from network import SegmentationNetwork as Network


try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError
try:
    FileExistsError
except NameError:
    FileExistsError = OSError


def save_seg_of_images(input_folder, output_folder, labelmap_file, model_def, model_weights, crop_size):
    input_list = glob.glob(os.path.join(input_folder, "*.jpg"))
    input_list += glob.glob(os.path.join(input_folder, "*.JPG"))
    input_list += glob.glob(os.path.join(input_folder, "*.png"))
    input_list += glob.glob(os.path.join(input_folder, "*.PNG"))
    input_list.sort()

    assert input_list, "input_folder(%s) doesn't have image file(jpeg or png format)" % input_folder

    caffe.set_device(0)
    caffe.set_mode_gpu()
    network = Network(model_def, model_weights, labelmap_file, crop_size)

    for input_image_path in tqdm(input_list):
        output_image_path = input_image_path.replace(input_folder, output_folder)

        image = caffe.io.load_image(input_image_path)  # np.float32 in range [0, 1]

        # inference
        result = network(image)
        result = Image.fromarray((result * 255).astype('uint8'))
        result.save(output_image_path)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str)
    parser.add_argument("output_folder", type=str)
    parser.add_argument("--labelmap-file", type=str, default="./labelmap_cs.prototxt",
                        help="path to labelmap file")
    parser.add_argument("--model-def", type=str, default="./pspnet_deploy.prototxt",
                        help="path to deploy.prototxt")
    parser.add_argument("--model-weights", type=str, default="./pspnet.caffemodel",
                        help="path to .caffemodel")
    parser.add_argument("--image-size", nargs='+', type=int, default=473,
                        help="size to resize input images")
    return parser.parse_args()


def main():
    args = get_arguments()

    input_folder = args.input_folder
    output_folder = args.output_folder
    labelmap_file = args.labelmap_file
    model_def = args.model_def
    model_weights = args.model_weights
    crop_size = args.image_size

    if len(crop_size) == 1:
        crop_size = crop_size[0]

    if not os.path.isdir(input_folder):
        raise FileNotFoundError("input_folder(%s) does not exist" % input_folder)
    if not os.path.isfile(labelmap_file):
        raise FileNotFoundError("labelnap_file(%s) does not exist" % labelmap_file)
    if not os.path.isfile(model_def):
        raise FileNotFoundError("model_def(%s) does not exist" % model_def)
    if not os.path.isfile(model_weights):
        raise FileNotFoundError("model_weights(%s) does not exist" % model_weights)
    
    try:
        os.makedirs(output_folder)
    except FileExistsError:
        pass

    save_seg_of_images(
        input_folder=input_folder,
        output_folder=output_folder,
        labelmap_file=labelmap_file,
        model_def=model_def,
        model_weights=model_weights,
        crop_size=crop_size)


if __name__ == "__main__":
    main()
