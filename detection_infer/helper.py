
from __future__ import print_function
from math import ceil

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


class ImageCropper(object):
    def __init__(self, image, grid=(3, 5)):
        self.image = image
        self.grid = grid  # (h, w)
        self.center = tuple((int(x/2) for x in grid))

        self.image_grid = self.__crop_image_by_grid()

    def get_cropped_image(self, grid_idx):
        return self.image_grid[grid_idx[0]][grid_idx[1]]

    def get_image_center(self):
        return self.get_cropped_image(self.center)

    def align_coordinates(self, detections):
        y_grid, x_grid = self.grid
        x_offset = 1. / x_grid * 2
        y_offset = 1. / y_grid * 1
        detections[:,:,:,3] = detections[:,:,:,3] / x_grid + x_offset  # xmin
        detections[:,:,:,4] = detections[:,:,:,4] / y_grid + y_offset  # ymin
        detections[:,:,:,5] = detections[:,:,:,5] / x_grid + x_offset  # xmax
        detections[:,:,:,6] = detections[:,:,:,6] / y_grid + y_offset  # ymax
        return detections

    def __crop_image_by_grid(self):
        """Crop image by grid, and return as mxn array of image.
        If size of image was not divisible by grid, images of last row and column
        would be smaller than others.
        Args
            img: numpy array
            grid: a tuple (m, n)
        """
        img = self.image
        grid = self.grid
        w, h = img.shape[0:2]
        w_crop = ceil(w / grid[0])
        h_crop = ceil(h / grid[1])
        img_list = []
        for c in range(grid[0]):
            column_list = []
            for r in range(grid[1]):
                left = int(c * w_crop)
                right = int(min(left + w_crop, w))
                upper = int(r * h_crop)
                lower = int(min(upper + h_crop, h))
                column_list.append(img[left:right, upper:lower, :])
            img_list.append(column_list)
        return img_list


class BBoxPainter(object):
    def __init__(self, labelmap, dpi=120):
        self.dpi = dpi
        self.labelmap = labelmap
        self.drawing_labels = None
        # self.drawing_labels = ['bicycle', 'bus', 'car', 'motorbike', 'person', 'train']

    def __call__(self, image, detections):
        h, w = image.shape[0:2]
        dpi = self.dpi
        figsize = (float(w)/dpi, float(h)/dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
        self.__draw(ax, image, detections)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        fig.canvas.draw()
        bboxed_img = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)

        # clear memory
        ax.clear()
        plt.close(fig)

        bboxed_img.shape = (h, w, 4)

        # argb to rgb
        bboxed_img = bboxed_img[:, :, 1:]
        return bboxed_img

    def __draw(self, ax, image, detections):
        labelmap = self.labelmap
        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        # Get detections with confidence higher than 0.6.
        top_conf_indices = [(i, conf) for i, conf in enumerate(det_conf) if conf >= 0.6]

        # Sort using conf
        top_indices = [i for i, conf in sorted(top_conf_indices, key=lambda x: x[1], reverse=True)]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = self.get_labelname(labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

        # draw bounding boxes
        ax.imshow(image, aspect='equal')
        ax.axis('off')

        for i in xrange(top_conf.shape[0]):
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            if self.drawing_labels is not None and label_name not in self.drawing_labels:
                continue
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            display_txt = '%s: %.2f' % (label_name, score)
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            color = colors[label]
            ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            ax.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

    @staticmethod
    def get_labelname(labelmap, labels):
        num_labels = len(labelmap.item)
        labelnames = []
        if type(labels) is not list:
            labels = [labels]
        for label in labels:
            found = False
            for i in xrange(0, num_labels):
                if label == labelmap.item[i].label:
                    found = True
                    labelnames.append(labelmap.item[i].display_name)
                    break
            assert found
        return labelnames