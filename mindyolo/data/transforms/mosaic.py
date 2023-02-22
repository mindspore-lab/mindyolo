import numpy as np
import cv2

from .perspective import RandomPerspective
from .common import SimpleCopyPaste

__all__ = ['Mosaic']


class Mosaic:
    """
    Mosaic Data Augmentation and Perspective
    1. get mosaic image, get mosaic_labels
    2. copy_paste
    3. random_perspective augment
    Args:
        mosaic_prob (float): probability of using Mosaic, 1.0 as default
        copy_paste_prob (float): probability of using SimpleCopyPaste, 0.0 as default
        degrees (float): the rotate range to apply, transform range is [-10, 10]
        translate (float): the translate range to apply, transform range is [-0.1, 0.1]
        scale (float): the scale range to apply, transform range is [0.1, 2]
        shear (float): the shear range to apply, transform range is [-2, 2]
        perspective (float): the perspective range to apply, transform range is [0, 0.001]
    """
    def __init__(self, mosaic_prob=1.0, copy_paste_prob=0.0, degrees=0.0, translate=0.2, scale=0.9, shear=0.0, perspective=0.0, target_size=640):

        self.mosaic_prob = mosaic_prob
        self.target_size = target_size
        self.mosaic_border = [-target_size // 2, -target_size // 2]
        self.simple_copy_paste = SimpleCopyPaste(prob=copy_paste_prob)
        self.random_perspective = RandomPerspective(degrees=degrees,
                                                    translate=translate,
                                                    scale=scale,
                                                    shear=shear,
                                                    perspective=perspective,
                                                    border=self.mosaic_border)

    def __call__(self, records_outs):
        if np.random.random() < self.mosaic_prob:
            # loads images in a 4-mosaic
            img_size = self.target_size
            gt_bboxes4, gt_polys4, gt_classes4 = [], [], []
            yc, xc = [int(np.random.uniform(-x, 2 * img_size + x)) for x in self.mosaic_border]  # mosaic center x, y
            for i, record_out in enumerate(records_outs):
                # Load image
                img = record_out['image']  # BGR
                gt_bbox = record_out['gt_bbox']
                gt_poly = record_out['gt_poly']
                h0, w0 = img.shape[:2]  # orig hw
                r = img_size / max(h0, w0)  # resize image to img_size
                if r != 1:  # always resize down, only resize up if training with augmentation
                    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
                    gt_bbox *= r
                    for poly in gt_poly:
                        poly *= r

                h, w = img.shape[:2]  # hw_resized

                # place img in img4
                if i == 0:  # top left
                    img4 = np.full((img_size * 2, img_size * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
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
                padw = x1a - x1b
                padh = y1a - y1b

                # Labels
                gt_poly, gt_class = record_out['gt_poly'], record_out['gt_class']
                if gt_bbox.size:
                    gt_bbox[:, [0, 2]] += padw
                    gt_bbox[:, [1, 3]] += padh
                    for x in gt_poly:
                        x[:, 0] += padw
                        x[:, 1] += padh

                gt_bboxes4.append(gt_bbox)
                gt_classes4.append(gt_class)
                gt_polys4.extend(gt_poly)

            # Concat/clip labels
            gt_classes4 = np.concatenate(gt_classes4, 0)
            gt_bboxes4 = np.concatenate(gt_bboxes4, 0)
            for x in (gt_bboxes4, *gt_polys4):
                np.clip(x, 0, 2 * img_size, out=x)  # clip when using random_perspective()

            # Augment
            img4, gt_bboxes4, gt_classes4, gt_polys4 = \
                self.simple_copy_paste(img4, gt_bboxes4, gt_classes4, gt_polys4)
            img4, gt_bboxes4, gt_classes4, gt_polys4 = \
                self.random_perspective(img4, gt_bboxes4, gt_classes4, gt_polys4)

            record_out = records_outs[0]
            record_out['image'] = img4
            record_out['gt_class'] = gt_classes4
            record_out['gt_bbox'] = gt_bboxes4
            record_out['gt_poly'] = gt_polys4

            return record_out
        else:
            return records_outs[0]
