import numpy as np
import cv2

from ..general import normalize_shape, normalize_shape_with_poly

__all__ = ['Resize', 'BatchRandomResize', 'BatchLabelsPadding']


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
            consider_poly(bool): whether to consider the change of gt_poly
        """
        super(Resize, self).__init__()
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.keep_ratio = keep_ratio
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

    def __call__(self, img, gt_bbox, gt_class, gt_poly=[]):
        """ Resize the image numpy."""
        # apply image
        shape = img.shape
        new_shape = self.target_size
        if self.keep_ratio:
            # Scale ratio (new / old)
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

            # Compute padding
            ratio = r, r  # height, width ratios
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

            dw /= 2  # divide padding into 2 sides
            dh /= 2

            if shape[::-1] != new_unpad:  # resize
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border

            h_ratio, w_ratio = ratio[0], ratio[1]
            gt_bbox[:, 0] = w_ratio * gt_bbox[:, 0] + dw
            gt_bbox[:, 1] = h_ratio * gt_bbox[:, 1] + dh
            gt_bbox[:, 2] = w_ratio * gt_bbox[:, 2] + dw
            gt_bbox[:, 3] = h_ratio * gt_bbox[:, 3] + dh

            if self.consider_poly and len(gt_poly):
                gt_poly[..., 0] = w_ratio * gt_poly[..., 0] + dw
                gt_poly[..., 1] = h_ratio * gt_poly[..., 1] + dh
        else:
            im_scale_y = self.target_size[1] / shape[0]
            im_scale_x = self.target_size[0] / shape[1]

            img = self.resize_image(img, [im_scale_x, im_scale_y])
            resize_h, resize_w = img.shape[:2]

            if len(gt_bbox) > 0:
                gt_bbox[:, 0::2] *= im_scale_x
                gt_bbox[:, 1::2] *= im_scale_y
                gt_bbox[:, 0::2] = np.clip(gt_bbox[:, 0::2], 0, resize_w)
                gt_bbox[:, 1::2] = np.clip(gt_bbox[:, 1::2], 0, resize_h)

            if self.consider_poly and len(gt_poly):
                gt_poly[..., 0] *= im_scale_x
                gt_poly[..., 1] *= im_scale_y

        if self.consider_poly:
            return img, gt_bbox, gt_class, gt_poly
        else:
            return img, gt_bbox, gt_class


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

    def __call__(self, images, gt_bboxes, gt_classes, batch_info):
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
            images[i], gt_bboxes[i], gt_classes[i] = \
                resizer(images[i], gt_bboxes[i], gt_classes[i])
        if self.consider_poly:
            images, gt_bboxes, gt_classes, batch_idx = \
                normalize_shape(images, gt_bboxes, gt_classes, batch_info)
        else:
            images, gt_bboxes, gt_classes, batch_idx = \
                normalize_shape(images, gt_bboxes, gt_classes, batch_info)
        return images, gt_bboxes, gt_classes, batch_idx


class BatchLabelsPadding:
    """
    Padding the targets of each batch
    Args:
        padding_size (int): samples target padding to this size, if targets size greater than padding_size, crop to this size.
    """
    def __init__(self, padding_size, padding_value=-1):
        self.padding_size = padding_size
        self.padding_value = padding_value

    def __call__(self, images, gt_bboxes, gt_classes, batch_info):
        """
        Padding the list of numpy labels.
        """
        gt_bboxes, gt_classes, batch_idx = self.padding(gt_bboxes, gt_classes)
        return np.stack(images, 0), np.stack(gt_bboxes, 0), np.stack(gt_classes, 0), np.stack(batch_idx, 0)

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
