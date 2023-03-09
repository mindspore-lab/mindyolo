import random
import cv2
import numpy as np

from ..general import bbox_ioa, check_img_size

__all__ = ['SimpleCopyPaste',
           'RandomFlip', 'RandomHSV',
           'NormalizeImage', 'TransposeImage',
           'NormalizeBox', 'LetterBox']


class RandomFlip:
    """Random left_right flip
    Args:
        prob (float): the probability of flipping image
        consider_poly(bool): whether to consider the change of gt_poly
    """

    def __init__(self, prob=0.5, consider_poly=False):
        self.prob = prob
        self.consider_poly = consider_poly
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self,
                 img,
                 im_file, im_id, ori_shape, pad, ratio,
                 gt_bbox, gt_class, gt_poly=[]):
        if np.random.random() < self.prob:
            img = cv2.flip(img, 1)
            h, w = img.shape[:2]

            oldx1 = gt_bbox[:, 0].copy()
            oldx2 = gt_bbox[:, 2].copy()
            gt_bbox[:, 0] = w - oldx2
            gt_bbox[:, 2] = w - oldx1  # x1 and x2 will be exchanged after flip

        if self.consider_poly and len(gt_poly):
            for poly in gt_poly:
                poly[:, 0] = w - poly[:, 0]
            return img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class, gt_poly
        else:
            return img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class


class RandomHSV:
    """
    HSV color-space augmentation
    """
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5, consider_poly=False):
        self.gains = [hgain, sgain, vgain]
        self.consider_poly = consider_poly

    def __call__(self,
                 img,
                 im_file, im_id, ori_shape, pad, ratio,
                 gt_bbox, gt_class, gt_poly=[]):
        r = np.random.uniform(-1, 1, 3) * self.gains + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        if self.consider_poly:
            return img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class, gt_poly
        else:
            return img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class


class SimpleCopyPaste:
    """
    Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177
    """
    def __init__(self, prob=0.5, consider_poly=True):
        self.prob = prob
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, img, gt_bbox, gt_class, gt_poly=[]):
        n = len(gt_poly)
        if self.prob and n:
            h, w, c = img.shape  # height, width, channels
            im_new = np.zeros(img.shape, np.uint8)
            for j in random.sample(range(n), k=round(self.prob * n)):
                b, p, c = gt_bbox[j], gt_poly[j], gt_class[j]
                box = [w - b[2], b[1], w - b[0], b[3]]
                ioa = bbox_ioa(box, gt_bbox)  # intersection over area
                if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                    gt_bbox = np.concatenate((gt_bbox, [box, ]), 0)
                    gt_class = np.concatenate((gt_class, [c, ]), 0)
                    gt_poly.append(np.concatenate((w - p[:, 0:1], p[:, 1:2]), 1))
                    cv2.drawContours(im_new, [gt_poly[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

            result = cv2.bitwise_and(src1=img, src2=im_new)
            result = cv2.flip(result, 1)  # augment segments (flip left-right)
            i = result > 0  # pixels to replace
            img[i] = result[i]

        return img, gt_bbox, gt_class, gt_poly


class LetterBox:
    """
    Resize and pad image while meeting stride-multiple constraints
    Args:
        target_size: the newshape after letterbox
        color: color of the padding edge
        scaleup(bool): whether scaleup is allowed
        stride(int): stride for the image, which means the (w, h) for image will be divisible by it
        consider_poly(bool): whether to consider the change of gt_poly
    """
    def __init__(self,
                 target_size=(640, 640),
                 color=(114, 114, 114),
                 auto=False,
                 scalefill=False,
                 scaleup=False,
                 stride=32,
                 consider_poly=False):
        self.new_shape = (target_size, target_size) if isinstance(target_size, int) else target_size
        self.color = color
        self.auto = auto
        self.scalefill = scalefill
        self.scaleup = scaleup
        self.stride = stride
        self.consider_poly = consider_poly

    def __call__(self, img,
                 im_file, im_id, ori_shape, pad, ratio,
                 gt_bbox, gt_class, gt_poly=[]):
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = [check_img_size(s, self.stride) for s in self.new_shape]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # height, width ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scalefill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[0] / shape[0], new_shape[1] / shape[1]  # height, width ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)  # add border

        h_ratio, w_ratio = ratio[0], ratio[1]
        gt_bbox[:, 0] = w_ratio * gt_bbox[:, 0] + dw
        gt_bbox[:, 1] = h_ratio * gt_bbox[:, 1] + dh
        gt_bbox[:, 2] = w_ratio * gt_bbox[:, 2] + dw
        gt_bbox[:, 3] = h_ratio * gt_bbox[:, 3] + dh

        pad += np.array([dh, dw])
        ratio *= np.array(ratio)

        if self.consider_poly and len(gt_poly):
            for poly in gt_poly:
                poly[:, 0] = w_ratio * poly[:, 0] + dw
                poly[:, 1] = h_ratio * poly[:, 1] + dh
            return img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class, gt_poly
        else:
            return img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class


class NormalizeBox:
    """Transform the bounding box's coornidates to [0,1]."""
    def __init__(self, xyxy2xywh=True, consider_poly=False):
        self.xyxy2xywh = xyxy2xywh
        self.consider_poly = consider_poly

    def __call__(self, img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class, gt_poly=[]):
        img, gt_bbox, gt_class = self.normalize_box(img, gt_bbox, gt_class)

        if self.consider_poly and len(gt_poly):
            return img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class, gt_poly
        else:
            return img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class

    def normalize_box(self, img, gt_bbox, gt_class):
        height, width, _ = img.shape
        gt_bbox[:, [0, 2]] /= width
        gt_bbox[:, [1, 3]] /= height

        if self.xyxy2xywh:
            gt_bbox = self.xyxy2xywh_fn(gt_bbox)

        return img, gt_bbox, gt_class

    @staticmethod
    def xyxy2xywh_fn(x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y


class NormalizeImage:
    """
    Args:
        mean (list): the pixel mean
        std (list): the pixel variance
        is_scale (bool): scale the pixel to [0,1]
        norm_type (str): type in ['mean_std', 'none']
    """
    def __init__(self, mean=None, std=None, is_scale=True, norm_type='mean_std', consider_poly=False):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type = norm_type
        self.consider_poly = consider_poly

        if not (isinstance(self.is_scale, bool) and self.norm_type in ['mean_std', 'none']):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if self.std and reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class, gt_poly=[]):
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

        if self.consider_poly:
            return img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class, gt_poly
        else:
            return img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class


class TransposeImage:
    """
    Args:
        bgr2rgb (bool): transpose image channel from BGR to RGB
        hwc2chw (bool): transpose image dim from (h, w, c) to (c, h, w)
    """
    def __init__(self, bgr2rgb=True, hwc2chw=True, consider_poly=False):

        self.bgr2rgb = bgr2rgb
        self.hwc2chw = hwc2chw
        self.consider_poly = consider_poly

        if not (isinstance(bgr2rgb, bool) and isinstance(hwc2chw, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class, gt_poly=[]):

        if self.bgr2rgb:
            img = img[:, :, ::-1]

        if self.hwc2chw:
            img = img.transpose(2, 0, 1)

        if self.consider_poly:
            return img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class, gt_poly
        else:
            return img, im_file, im_id, ori_shape, pad, ratio, gt_bbox, gt_class
