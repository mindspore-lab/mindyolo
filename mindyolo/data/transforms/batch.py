import numpy as np
import cv2

from .resize import Resize
from ..general import normalize_shape, normalize_shape_with_poly

__all__ = ['BatchRandomResize', 'BatchNormalizeShape', 'BatchLabelsPadding']


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
        consider_poly(bool): whether to consider the change of gt_poly
    """
    def __init__(self,
                 target_size=[640, 640],
                 keep_ratio=True,
                 interp=cv2.INTER_LINEAR,
                 random_range=False,
                 random_size=True,
                 random_interp=False,
                 consider_poly=False
                 ):
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        self.random_range = random_range
        self.random_size = random_size
        self.random_interp = random_interp
        self.consider_poly = consider_poly

        if (random_range or random_size) and isinstance(target_size, int):
            raise TypeError("Type of target_size is invalid when random_size or random_range is True. Must be List or "
                            "Tuple, now is int.")
        if random_range and not len(target_size) == 2:
            raise TypeError("target_size must be two list as [[min_short_edge, long_edge], [max_short_edge, "
                            "long_edge]] when random_range is True.")

    def __call__(self, images, im_files, im_ids, ori_shapes, pads, ratios, gt_bboxes, gt_classes, batch_info):
        """Resize the image numpy."""
        bs = len(images)
        if self.random_range:
            img_scale_long = [max(s) for s in self.target_size]
            img_scale_short = [min(s) for s in self.target_size]
            long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
            short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
            target_size = [short_edge, long_edge]
        else:
            if self.random_size:
                index = np.random.choice(len(self.target_size))
                target_size = self.target_size[index]
            else:
                target_size = self.target_size

        if self.random_interp:
            interp = np.random.choice(self.interps)
        else:
            interp = self.interp

        resizer = Resize(target_size, self.keep_ratio, interp, consider_poly=False)
        for i in range(bs):
            images[i], im_files[i], im_ids[i], ori_shapes[i], pads[i], ratios[i], gt_bboxes[i], gt_classes[i] = \
                resizer(images[i], im_files[i], im_ids[i], ori_shapes[i], pads[i], ratios[i], gt_bboxes[i], gt_classes[i])

        images, gt_bboxes, gt_classes, batch_idxes = \
            normalize_shape(images, gt_bboxes, gt_classes)

        return np.stack(images, 0), \
               np.stack(im_files, 0), np.stack(im_ids, 0), np.stack(ori_shapes, 0), np.stack(pads, 0), np.stack(ratios, 0), \
               np.stack(gt_bboxes, 0), np.stack(gt_classes, 0), np.stack(batch_idxes, 0)


class BatchNormalizeShape:
    def __call__(self, images, im_files, im_ids, ori_shapes, pads, ratios, gt_bboxes, gt_classes, batch_info):
        images, gt_bboxes, gt_classes, batch_idxes = \
            normalize_shape(images, gt_bboxes, gt_classes)
        return np.stack(images, 0), \
               np.stack(im_files, 0), np.stack(im_ids, 0), np.stack(ori_shapes, 0), np.stack(pads, 0), np.stack(ratios, 0), \
               np.stack(gt_bboxes, 0), np.stack(gt_classes, 0), np.stack(batch_idxes, 0)


class BatchLabelsPadding:
    """
    Padding the targets of each batch
    Args:
        padding_size (int): samples target padding to this size, if targets size greater than padding_size, crop to this size.
    """
    def __init__(self, padding_size, padding_value=-1):
        self.padding_size = padding_size
        self.padding_value = padding_value

    def __call__(self, images, im_files, im_ids, ori_shapes, pads, ratios, gt_bboxes, gt_classes, batch_info):
        """
        Padding the list of numpy labels.
        """
        gt_bboxes, gt_classes, batch_idxes = self.padding(gt_bboxes, gt_classes)
        return np.stack(images, 0), \
               np.stack(im_files, 0), np.stack(im_ids, 0), np.stack(ori_shapes, 0), np.stack(pads, 0), np.stack(ratios, 0), \
               np.stack(gt_bboxes, 0), np.stack(gt_classes, 0), np.stack(batch_idxes, 0)

    def padding(self, gt_bboxes, gt_classes):
        """
        Labels padding to target size.
        """
        ps = self.padding_size
        pv = self.padding_value

        batch_idxes = []
        for i, (gt_bbox, gt_class) in enumerate(zip(gt_bboxes, gt_classes)):
            nL = gt_class.shape[0]
            nL = nL if nL < ps else ps
            gt_bboxes[i] = np.full((ps, 4), pv, dtype=np.float32)
            gt_bboxes[i][:nL, :] = gt_bbox[:nL, :]
            gt_classes[i] = np.full((ps, 1), pv, dtype=np.int32)
            gt_classes[i][:nL, :] = gt_class[:nL, :]
            batch_idxes.append(np.full((ps, 1), i, dtype=np.int32))

        return gt_bboxes, gt_classes, batch_idxes
