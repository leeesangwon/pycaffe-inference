from __future__ import print_function
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import caffe

from helper import ImageCropper, ImageAssembler


class SegmentationNetwork(object):
    def __init__(self, model_def, model_weights, colormap_file, crop_size):
        self.colormap = self.__get_colormap(colormap_file)
        self.net = self.__get_net(model_def, model_weights)
        self.transformer = self.__get_transformer()
        if type(crop_size) is int:
            self.crop_size = (crop_size, crop_size)
        elif type(crop_size) in [list, tuple]:
            self.crop_size = tuple(crop_size)
        else:
            raise ValueError

        # set net to batch size of 1
        self.net.blobs['data'].reshape(1, 3, self.crop_size[0], self.crop_size[1])

    @staticmethod
    def __get_colormap(colormap_file):
        colormap = scipy.io.loadmat(colormap_file)
        cm_ = colormap['colormapcs']
        custom_cm = plt.cm.get_cmap()
        for i, _ in enumerate(custom_cm.colors):
            custom_cm.colors[i] = list(cm_[i])
        custom_cm.name = 'cityscapes'
        return custom_cm

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

    def __call__(self, input_image):
        """
        Inferencing Segmentation Network
        :param input_image: np.float32 in range[0,1]
        :return: np.float32 in range[0,1]
        """
        image_cropper = ImageCropper(input_image, self.crop_size)
        result_grid = []
        for image_row in image_cropper.image_grid:
            result_row = []
            for image in image_row:
                raw_result = self.forward(image)
                segmentation = self.postprocess(raw_result)
                result_row.append(segmentation)
            result_grid.append(result_row)
        new_shape = (input_image.shape[0], input_image.shape[1], 3)
        image_assembler = ImageAssembler(result_grid, new_shape)
        return image_assembler.assembled_image

    def forward(self, image):
        """
        Forward path
        :param image: np.float32 in range[0,1]
        :return: np.float32 [b, c, h, w]
        """
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        score = self.net.forward()  # dict which contains last feature map
        return list(score.values())[-1]

    def postprocess(self, raw_result):
        # deep copy
        result = np.copy(raw_result)
        # squeeze batch dimension
        result = np.squeeze(result, axis=0)
        # (C x H x W) -> (H x W x C)
        result = np.transpose(result, axes=(1, 2, 0))
        # (H x W x C) -> (H x W x 1)
        result = np.argmax(result, axis=2)
        # color mapping
        result = self.__colormapping(result)
        return result

    def __colormapping(self, input_):
        out = self.colormap(input_)
        # remove alpha channel
        out = out[:, :, :3]
        out = np.float32(out)
        return out
