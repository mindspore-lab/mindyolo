import numpy as np
import cv2

__all__ = ['Resize']


class Resize:
    def __init__(self, target_size=[640, 640], keep_ratio=False, interp=None, consider_poly=False):
        """
        Resize image to target size. if keep_ratio is True,
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int, option): the interpolation method
        """
        super(Resize, self).__init__()
        self.keep_ratio = keep_ratio
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.interp = interp
        self.consider_poly = consider_poly

    def resize_image(self, image, scale):
        im_scale_x, im_scale_y = scale
        interp = self.interp if self.interp else (cv2.INTER_AREA if min(scale) < 1 else cv2.INTER_LINEAR)

        return cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=interp)

    def resize_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)

        return bbox

    def resize_poly(self, polys, scale):
        im_scale_x, im_scale_y = scale
        for i, poly in enumerate(polys):
            poly[:, 0] *= im_scale_x
            poly[:, 1] *= im_scale_y
            polys[i] = poly

        return polys

    def __call__(self, img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class, gt_poly=[]):
        """ Resize the image numpy.
        """

        # apply image
        img_shape = img.shape
        if self.keep_ratio:

            img_size_min = np.min(img_shape[0:2])
            img_size_max = np.max(img_shape[0:2])

            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)

            im_scale = min(target_size_min / img_size_min,
                           target_size_max / img_size_max)

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            im_scale_y = self.target_size[1] / img_shape[0]
            im_scale_x = self.target_size[0] / img_shape[1]

        img = self.resize_image(img, [im_scale_x, im_scale_y])
        resize_h, resize_w = img.shape[:2]
        ratio *= np.array([resize_h / img_shape[0], resize_w / img_shape[1]])

        if len(gt_bbox) > 0:
            gt_bbox = self.resize_bbox(gt_bbox, [im_scale_x, im_scale_y], [resize_w, resize_h])

        if self.consider_poly:
            gt_poly = self.resize_poly(gt_poly, [im_scale_x, im_scale_y])
            return img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class, gt_poly
        else:
            return img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class
