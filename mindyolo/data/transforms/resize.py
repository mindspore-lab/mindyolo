import sys
import numpy as np
import cv2

sys.path.append('../')
from general import normalize_shape

__all__ = ['Resize', 'BatchRandomResize']


class Resize:
    def __init__(self, target_size=[640, 640], keep_ratio=False, interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True,
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        super(Resize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def resize_image(self, image, scale):
        im_scale_x, im_scale_y = scale

        return cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

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

    def __call__(self, img, w, h, gt_bbox, gt_class, *gt_poly):
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

            resize_h = im_scale * float(img_shape[0])
            resize_w = im_scale * float(img_shape[1])

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = np.asarray(self.target_size)
            im_scale_y = resize_h / img_shape[0]
            im_scale_x = resize_w / img_shape[1]

        img = self.resize_image(img, [im_scale_x, im_scale_y])
        resize_w = resize_w.astype(np.float32)
        resize_h = resize_h.astype(np.float32)

        if len(gt_bbox) > 0:
            gt_bbox = self.resize_bbox(gt_bbox, [im_scale_x, im_scale_y], [resize_w, resize_h])

        if gt_poly:
            gt_poly = self.resize_poly(gt_poly, [im_scale_x, im_scale_y])
            return img, resize_w, resize_h, gt_bbox, gt_class, gt_poly
        else:
            return img, resize_w, resize_h, gt_bbox, gt_class


class BatchRandomResize:
    """
    Resize image to target size randomly. random target_size and interpolation method
    Args:
        target_size (int, list, tuple): image target size, if random size is True, must be list or tuple
        keep_ratio (bool): whether keep_raio or not, default true
        interp (int): the interpolation method
        random_range (bool): whether random select target size of image, the target_size must be
            a [[min_short_edge, long_edge], [max_short_edge, long_edge]]
        random_size (bool): whether random select target size of image
        random_interp (bool): whether random select interpolation method
    """
    def __init__(self,
                 target_size=[640, 640],
                 keep_ratio=True,
                 interp=cv2.INTER_LINEAR,
                 random_range=False,
                 random_size=True,
                 random_interp=False):
        self.keep_ratio = keep_ratio
        self.interp = interp
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]

        if (random_range or random_size) and isinstance(target_size, int):
            raise TypeError("Type of target_size is invalid when random_size or random_range is True. Must be List or "
                            "Tuple, now is int.")
        if random_range and not len(target_size) == 2:
            raise TypeError("target_size must be two list as [[min_short_edge, long_edge], [max_short_edge, "
                            "long_edge]] when random_range is True.")
        self.target_size = target_size
        self.random_range = random_range
        self.random_size = random_size
        self.random_interp = random_interp

    def __call__(self, images, ws, hs, gt_bboxes, gt_classes, batch_info):
        """
        Resize the image numpy.
        """
        bs = len(images)
        if self.random_range:
            img_scale_long = [max(s) for s in self.target_size]
            img_scale_short = [min(s) for s in self.target_size]
            long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
            short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
            target_size = [short_edge, long_edge]
        else:
            if self.random_size:
                target_size = np.random.choice(self.target_size)
            else:
                target_size = self.target_size

        if self.random_interp:
            interp = np.random.choice(self.interps)
        else:
            interp = self.interp

        resizer = Resize(target_size, self.keep_ratio, interp)
        for i in range(bs):
            images[i], ws[i], hs[i], gt_bboxes[i], gt_classes[i] = resizer(images[i], ws[i], hs[i], gt_bboxes[i], gt_classes[i])
        images, ws, hs, gt_bboxes, gt_classes = normalize_shape(images, ws, hs, gt_bboxes, gt_classes, batch_info)
        return images, ws, hs, gt_bboxes, gt_classes
