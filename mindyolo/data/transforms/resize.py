import numpy as np
import cv2

from ..general import normalize_shape

__all__ = ['Resize', 'BatchRandomResize', 'BatchLabelsPadding']


class Resize:
    def __init__(self, target_size=[640, 640], keep_ratio=False, interp=None):
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

    def __call__(self, img, im_file, ori_shape, pad, ratio, gt_bbox, gt_class, gt_poly=None):
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

        if gt_poly:
            gt_poly = self.resize_poly(gt_poly, [im_scale_x, im_scale_y])
            return img, im_file, ori_shape, pad, ratio, gt_bbox, gt_class, gt_poly
        else:
            return img, im_file, ori_shape, pad, ratio, gt_bbox, gt_class


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

    def __call__(self, images, im_files, ori_shapes, pads, ratios, gt_bboxes, gt_classes, batch_info):
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
            images[i], im_files[i], ori_shapes[i], gt_bboxes[i], gt_classes[i] = \
                resizer(images[i], im_files[i], ori_shapes[i], gt_bboxes[i], gt_classes[i])
        images, im_files, ori_shapes, gt_bboxes, gt_classes, batch_idx = \
            normalize_shape(images, im_files, ori_shapes, pads, ratios, gt_bboxes, gt_classes, batch_info)
        return images, im_files, ori_shapes, gt_bboxes, gt_classes, batch_idx


class BatchLabelsPadding:
    """
    Padding the targets of each batch
    Args:
        padding_size (int): samples target padding to this size, if targets size greater than padding_size, crop to this size.
    """
    def __init__(self, padding_size, padding_value=-1):
        self.padding_size = padding_size
        self.padding_value = padding_value

    def __call__(self, images, im_files, ori_shapes, pads, ratios, gt_bboxes, gt_classes, batch_info):
        """
        Padding the list of numpy labels.
        """
        gt_bboxes, gt_classes, batch_idx = self.padding(gt_bboxes, gt_classes)
        return np.stack(images, 0), im_files, np.stack(ori_shapes, 0), np.stack(pads, 0), np.stack(ratios, 0), \
               np.stack(gt_bboxes, 0), np.stack(gt_classes, 0), np.stack(batch_idx, 0)

    def padding(self, gt_bboxes, gt_classes):
        """
        Labels padding to target size.
        """
        ps = self.padding_size
        pv = self.padding_value

        batch_idx = []
        for i, (gt_bbox, gt_class) in enumerate(zip(gt_bboxes, gt_classes)):
            nL = gt_class.shape[0]
            nL = nL if nL < ps else ps
            gt_bboxes[i] = np.full((ps, 4), pv, dtype=np.float32)
            gt_bboxes[i][:nL, :] = gt_bbox[:nL, :]
            gt_classes[i] = np.full((ps, 1), pv, dtype=np.int32)
            gt_classes[i][:nL, :] = gt_class[:nL, :]
            batch_idx.append(np.full((ps, 1), i, dtype=np.int32))

        return gt_bboxes, gt_classes, batch_idx
