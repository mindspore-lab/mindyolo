import random
import cv2
import numpy as np

from .perspective import RandomPerspective
from .common import SimpleCopyPaste

__all__ = ['Mosaic']


class Mosaic:
    """
    Mosaic Data Augmentation and Perspective
    1. get mosaic image, get mosaic_labels
    2. copy_paste(optional)
    3. random_perspective augment
    Args:
        additional_imgs(int): number of additional_images needed
        mosaic_prob (float): probability of using Mosaic, 1.0 as default
        copy_paste_prob (float): probability of using SimpleCopyPaste, 0.0 as default
        translate (float): the translate range to apply, transform range is [-0.1, 0.1]
        scale (float): the scale range to apply, transform range is [0.1, 2]
        target_size: the newshape after letterbox
        consider_poly(bool): whether to consider the change of gt_poly
    """
    def __init__(self,
                 additional_imgs=8,
                 mosaic_prob=1.0,
                 copy_paste_prob=0.0,
                 translate=0.2,
                 scale=0.9,
                 target_size=640,
                 consider_poly=False):
        self.additional_imgs = additional_imgs
        self.mosaic_prob = mosaic_prob
        self.copy_paste_prob = copy_paste_prob
        self.mosaic_border = [-target_size // 2, -target_size // 2]
        self.target_size = target_size
        self.consider_poly = consider_poly

        self.simple_copy_paste = SimpleCopyPaste(prob=copy_paste_prob)
        self.random_perspective = RandomPerspective(translate=translate,
                                                    scale=scale,
                                                    border=self.mosaic_border,
                                                    consider_poly=consider_poly)

    def __call__(self, records_outs):
        if random.random() < self.mosaic_prob:
            if random.random() < 0.8:
                record_out = self.mosaic4(records_outs[:4])
            else:
                record_out = self.mosaic9(records_outs[:9])
        else:
            record_out = records_outs[0]

        return record_out

    def mosaic4(self, records_outs):
        # loads images in a 4-mosaic
        img_size = self.target_size
        gt_bboxes4, gt_polys4, gt_classes4 = [], [], []
        yc, xc = [int(random.uniform(-x, 2 * img_size + x)) for x in self.mosaic_border]  # mosaic center x, y
        for i, record_out in enumerate(records_outs):
            # Load image
            img, gt_bbox, gt_class = record_out['image'], record_out['gt_bbox'], record_out['gt_class']  # BGR
            if self.copy_paste_prob or self.consider_poly:
                gt_poly = record_out['gt_poly']

            h0, w0 = img.shape[:2]  # orig hw
            r = img_size / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
                gt_bbox *= r
                if self.copy_paste_prob or self.consider_poly:
                    for poly in gt_poly:
                        poly *= r

            h, w = img.shape[:2]  # hw_resized

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((img_size * 2, img_size * 2, img.shape[2]), 114,
                               dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, img_size * 2), min(img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padh = y1a - y1b  # relative y offset
            padw = x1a - x1b  # relative x offset

            # Labels
            if gt_bbox.size:
                gt_bbox[:, [0, 2]] += padw
                gt_bbox[:, [1, 3]] += padh
                if self.copy_paste_prob or self.consider_poly:
                    for poly in gt_poly:
                        poly[:, 0] += padw
                        poly[:, 1] += padh

            gt_bboxes4.append(gt_bbox)
            gt_classes4.append(gt_class)
            if self.copy_paste_prob or self.consider_poly:
                gt_polys4.extend(gt_poly)

        # Concat/clip labels
        gt_bboxes4 = np.concatenate(gt_bboxes4, 0)
        gt_classes4 = np.concatenate(gt_classes4, 0)
        if self.copy_paste_prob or self.consider_poly:
            for x in gt_polys4:
                np.clip(x, 0, 2 * img_size, out=x)  # clip when using random_perspective()

        np.clip(gt_bboxes4, 0, 2 * img_size, out=gt_bboxes4)  # clip when using random_perspective()

        # Augment
        img4, gt_bboxes4, gt_classes4, gt_polys4 = \
            self.simple_copy_paste(img4, gt_bboxes4, gt_classes4, gt_polys4)
        if self.consider_poly:
            img4, gt_bboxes4, gt_classes4, gt_polys4 = \
                self.random_perspective(img4, gt_bboxes4, gt_classes4, gt_polys4)
        else:
            img4, gt_bboxes4, gt_classes4 = \
                self.random_perspective(img4, gt_bboxes4, gt_classes4)

        record_out = records_outs[0]
        record_out['image'] = img4
        record_out['gt_class'] = gt_classes4
        record_out['gt_bbox'] = gt_bboxes4
        if self.consider_poly:
            record_out['gt_poly'] = gt_polys4

        return record_out

    def mosaic9(self, records_outs):
        # loads images in a 9-mosaic
        s = self.target_size
        gt_bboxes9, gt_classes9, gt_polys9,  = [], [], []
        for i, record_out in enumerate(records_outs):
            # Load image
            img, gt_bbox, gt_class = record_out['image'], record_out['gt_bbox'], record_out['gt_class']  # BGR
            if self.copy_paste_prob or self.consider_poly:
                gt_poly = record_out['gt_poly']
            h0, w0 = img.shape[:2]  # orig hw
            r = s / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
                gt_bbox *= r
                if self.copy_paste_prob or self.consider_poly:
                    for poly in gt_poly:
                        poly *= r

            h, w = img.shape[:2]  # hw_resized

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

            # Labels
            if gt_bbox.size:
                gt_bbox[:, [0, 2]] += padx
                gt_bbox[:, [1, 3]] += pady
                if self.copy_paste_prob or self.consider_poly:
                    for poly in gt_poly:
                        poly[:, 0] += padx
                        poly[:, 1] += pady

            gt_bboxes9.append(gt_bbox)
            gt_classes9.append(gt_class)
            if self.copy_paste_prob or self.consider_poly:
                gt_polys9.extend(gt_poly)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        gt_bboxes9 = np.concatenate(gt_bboxes9, 0)
        gt_classes9 = np.concatenate(gt_classes9, 0)

        gt_bboxes9[:, [0, 2]] -= xc
        gt_bboxes9[:, [1, 3]] -= yc
        np.clip(gt_bboxes9, 0, 2 * s, out=gt_bboxes9)  # clip when using random_perspective()

        if self.copy_paste_prob or self.consider_poly:
            for poly in gt_polys9:
                poly[:, 0] -= xc
                poly[:, 1] -= yc
                np.clip(poly, 0, 2 * s, out=poly)  # clip when using random_perspective()

        # Augment
        img9, gt_bboxes9, gt_classes9, gt_polys9 = \
            self.simple_copy_paste(img9, gt_bboxes9, gt_classes9, gt_polys9)
        if self.consider_poly:
            img9, gt_bboxes9, gt_classes9, gt_polys9 = \
                self.random_perspective(img9, gt_bboxes9, gt_classes9, gt_polys9)
        else:
            img9, gt_bboxes9, gt_classes9 = \
                self.random_perspective(img9, gt_bboxes9, gt_classes9)

        record_out = records_outs[0]
        record_out['image'] = img9
        record_out['gt_class'] = gt_classes9
        record_out['gt_bbox'] = gt_bboxes9
        if self.consider_poly:
            record_out['gt_poly'] = gt_polys9

        return record_out

    @staticmethod
    def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
        # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
        return y
