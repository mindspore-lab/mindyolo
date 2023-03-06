import random
import numpy as np

from .mosaic import Mosaic

__all__ = ['MixUp']


class MixUp:
    def __init__(self, additional_imgs=9, prob=1.0, alpha=8.0, beta=8.0, mosaic_needed=False, consider_poly=False):
        """ Mixup image and gt_bbox/gt_class/gt_poly
        Args:
            additional_imgs(int): number of additional_images needed
            prob (float): the probability of mixup
            alpha (float): alpha parameter of beta distribute
            beta (float): beta parameter of beta distribute
            mosaic_needed(bool): whether apply mosaic for the second image
            consider_poly(bool): whether to consider the change of gt_poly
        """
        self.additional_imgs = additional_imgs
        self.prob = prob
        self.alpha = alpha
        self.beta = beta
        self.mosaic_needed = mosaic_needed
        self.consider_poly = consider_poly

        if self.mosaic_needed:
            self.mosaic = Mosaic(mosaic_prob=1.0, consider_poly=consider_poly)

        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def __call__(self, records_outs):
        if random.random() > self.prob:
            return records_outs[0]

        record_out = records_outs[0]
        if self.mosaic_needed:
            record_out2 = self.mosaic(records_outs[1:])
        else:
            record_out2 = records_outs[1]

        img, gt_bbox, gt_class = record_out['image'], record_out['gt_bbox'], record_out['gt_class']  # BGR
        img2, gt_bbox2, gt_class2 = record_out2['image'], record_out2['gt_bbox'], record_out2['gt_class']  # BGR

        if self.consider_poly:
            gt_poly = record_out['gt_poly']
            gt_poly2 = record_out2['gt_poly']

        r = np.random.beta(self.alpha, self.beta)  # mixup ratio, alpha=beta=8.0
        if img.shape[:2] == img2.shape[:2]:
            img = (img * r + img2 * (1 - r)).astype(np.uint8)
        else:
            h, w, c = max(img.shape[0], img2.shape[0]), max(img.shape[1], img2.shape[1]), img.shape[2]
            new = np.zeros((h, w, c), 'float32')
            new[:img.shape[0], :img.shape[1], :] = img.astype('float32') * r
            new[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1.0 - r)
            img = new.astype('uint8')

        gt_bbox = np.concatenate((gt_bbox, gt_bbox2), 0)
        gt_class = np.concatenate((gt_class, gt_class2), 0)

        record_out['image'] = img
        record_out['gt_bbox'] = gt_bbox
        record_out['gt_class'] = gt_class

        if self.consider_poly:
            gt_poly = np.concatenate((gt_poly, gt_poly2), 0)
            record_out['gt_poly'] = gt_poly

        return record_out
