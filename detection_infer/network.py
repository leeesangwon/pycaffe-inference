from __future__ import print_function
import numpy as np

import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

from helper import ImageCropper, BBoxPainter


class DetectionNetwork(object):
    def __init__(self, model_def, model_weights, labelmap_file, image_resize):
        self.labelmap = self.__get_labelmap(labelmap_file)
        self.net = self.__get_net(model_def, model_weights)
        self.transformer = self.__get_transformer()
        self.image_size = image_resize

        # set net to batch size of 1
        self.net.blobs['data'].reshape(1, 3, image_resize, image_resize)

        self.bbox_painter = BBoxPainter(self.labelmap)

    @staticmethod
    def __get_labelmap(labelmap_file):
        with open(labelmap_file, 'r') as f:
            labelmap = caffe_pb2.LabelMap()
            text_format.Merge(str(f.read()), labelmap)
        return labelmap

    @staticmethod
    def __get_net(model_def, model_weights):
        return caffe.Net(model_def,      # defines the structure of the model
                         model_weights,  # contains the trained weights
                         caffe.TEST)     # use test mode (e.g., don't perform dropout)

    def __get_transformer(self):
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        # (H x W x C) -> (1 x C x H x W)
        transformer.set_transpose('data', (2, 0, 1))
        # the reference model has channels in BGR order instead of RGB
        transformer.set_channel_swap('data', (2, 1, 0))
        # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_raw_scale('data', 255)
        # mean pixel
        transformer.set_mean('data', np.array([104, 117, 123]))
        return transformer

    def __call__(self, input_image, use_center=False):
        detections = self.forward(input_image)
        detections = self.postprocess(detections)
        if use_center:
            detections_center = self.forward_center(input_image)
            detections_center = self.postprocess(detections_center)
            detections = np.concatenate((detections, detections_center), axis=2)
        return detections

    def forward(self, image):
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self.net.forward()['detection_out']

        return detections

    def forward_center(self, image):
        image_cropper = ImageCropper(image)
        image = image_cropper.get_image_center()
        detections = self.forward(image)
        detections = image_cropper.align_coordinates(detections)

        return detections

    def postprocess(self, raw_result):
        # deep copy
        result = np.copy(raw_result)
        return result
