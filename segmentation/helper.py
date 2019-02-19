from __future__ import print_function
import numpy as np
from math import ceil


class ImageCropper(object):
    def __init__(self, image, crop_size=473):
        """
        :param image: np.array
        :param crop_size: a integer or a tuple (w, h)
        """
        self.image = image
        if type(crop_size) is int:
            self.crop_size = (crop_size, crop_size)
        elif type(crop_size) in (list, tuple):
            self.crop_size = tuple(crop_size)
        else:
            raise ValueError
        self.image_grid = self.__crop_image_by_size()

    def get_cropped_image(self, grid_idx):
        return self.image_grid[grid_idx[0]][grid_idx[1]]

    def __crop_image_by_size(self):
        """Crop image by crop_size, and return as mxn array of image.
        If size of image was not divisible by crop_size, images would be zero-padded.
        Return:
            list of images
        """
        img = self.image
        w, h = img.shape[0:2]
        w_crop = self.crop_size[0]
        h_crop = self.crop_size[1]
        grid = (int(ceil(float(w)/w_crop)), int(ceil(float(h)/h_crop)))
        w_pad = w_crop - w % w_crop
        h_pad = h_crop - h % h_crop
        img = self.__zero_padding(img, w_pad, h_pad)
        img_list = []
        for c in range(grid[0]):
            column_list = []
            for r in range(grid[1]):
                left = int(c * w_crop)
                right = int(left + w_crop)
                upper = int(r * h_crop)
                lower = int(upper + h_crop)
                column_list.append(img[left:right, upper:lower, :])
            img_list.append(column_list)
        return img_list

    @staticmethod
    def __zero_padding(img, w_pad, h_pad):
        b = np.pad(img[:, :, 0], ((0, w_pad), (0, h_pad)), 'constant', constant_values=0)
        g = np.pad(img[:, :, 1], ((0, w_pad), (0, h_pad)), 'constant', constant_values=0)
        r = np.pad(img[:, :, 2], ((0, w_pad), (0, h_pad)), 'constant', constant_values=0)
        b = np.expand_dims(b, axis=2)
        g = np.expand_dims(g, axis=2)
        r = np.expand_dims(r, axis=2)
        return np.concatenate((b, g, r), axis=2)


class ImageAssembler(object):
    def __init__(self, image_grid, original_shape):
        self.shape = original_shape
        self.image_grid = image_grid
        self.assembled_image = self.__assemble_image_grid()

    def __assemble_image_grid(self):
        image_grid = self.image_grid
        shape = self.shape
        output_image = None

        for c in image_grid:
            column_img = None
            for r in c:
                if column_img is None:
                    column_img = r
                else:
                    column_img = np.concatenate((column_img, r), axis=1)
            if output_image is None:
                output_image = column_img
            else:
                output_image = np.concatenate((output_image, column_img), axis=0)

        output_image = self.__remove_padding(output_image, shape)

        return output_image

    @staticmethod
    def __remove_padding(img, shape):
        return img[:shape[0], :shape[1], :]


if __name__ == "__main__":
    from PIL import Image
    i = Image.open("a0006-IMG_2787.jpg")
    i.load()
    i = np.asarray(i)
    ic = ImageCropper(i)
    ia = ImageAssembler(ic.image_grid, ic.image.shape)
    ii = Image.fromarray(ia.assembled_image)
    ii.show()
