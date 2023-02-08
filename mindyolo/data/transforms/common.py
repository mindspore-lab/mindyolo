import numpy as np
import cv2

__all__ = ['RandomFlip', 'RandomHSV', 'NormalizeImage']


class RandomFlip:
    """Random left_right flip
    Args:
        prob (float): the probability of flipping image
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, w, h, gt_bbox, gt_class):
        if np.random.random() < self.prob:
            img = np.fliplr(img)
            oldx1 = gt_bbox[:, 0].copy()
            oldx2 = gt_bbox[:, 2].copy()
            gt_bbox[:, 0] = w - oldx2
            gt_bbox[:, 2] = w - oldx1
        return img, w, h, gt_bbox, gt_class


class NormalizeImage:
    def __init__(self,
                 is_scale=True,
                 norm_type='mean_std',
                 mean=None,
                 std=None,
                 ):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
            is_scale (bool): scale the pixel to [0,1]
            norm_type (str): type in ['mean_std', 'none']
        """
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type = norm_type

        from functools import reduce
        if self.std and reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, img, w, h, gt_bbox, gt_class):
        """Normalize the image.
        Operators:
            1.(optional) Scale the pixel to [0,1]
            2.(optional) Each pixel minus mean and is divided by std
        """
        img = img.astype(np.float32, copy=False)

        if self.is_scale:
            scale = 1.0 / 255.0
            img *= scale

        if self.norm_type == 'mean_std':
            mean = self.mean or img.mean((0, 1))
            mean = np.array(mean)[np.newaxis, np.newaxis, :]
            std = self.std or img.var((0, 1))
            std = np.array(std)[np.newaxis, np.newaxis, :]
            img -= mean
            img /= std
        return img, w, h, gt_bbox, gt_class


class RandomHSV:
    """
    HSV color-space augmentation
    """
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.gains = [hgain, sgain, vgain]

    def __call__(self, img, w, h, gt_bbox, gt_class):
        r = np.random.uniform(-1, 1, 3) * self.gains + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        return img, w, h, gt_bbox, gt_class