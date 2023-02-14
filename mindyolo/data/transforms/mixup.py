import numpy as np

from .mosaic import Mosaic

__all__ = ['MixUp']


class MixUp:
    def __init__(self, prob=1.0, alpha=1.5, beta=1.5, mosaic_needed=False):
        """ Mixup image and gt_bbox/gt_class/gt_poly
        Args:
            alpha (float): alpha parameter of beta distribute
            beta (float): beta parameter of beta distribute
        """
        self.prob = prob
        self.alpha = alpha
        self.beta = beta
        self.mosaic_needed = mosaic_needed
        self.mosaic = Mosaic(mosaic_prob=1.0)
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def __call__(self, records_outs):
        if np.random.uniform(0., 1.) > self.prob:
            return records_outs[0]

        if self.mosaic_needed:
            record_out1 = self.mosaic(records_outs[:4])
            record_out2 = self.mosaic(records_outs[4:])
        else:
            record_out1 = records_outs[0]
            record_out2 = records_outs[1]

        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            return record_out1
        if factor <= 0.0:
            return record_out2

        img1 = record_out1['image']
        img2 = record_out2['image']
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += \
            img2.astype('float32') * (1.0 - factor)
        im = img.astype('uint8')

        # apply bbox and class
        gt_bbox1 = record_out1['gt_bbox']
        gt_bbox2 = record_out2['gt_bbox']
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = record_out1['gt_class']
        gt_class2 = record_out2['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
        gt_poly1 = record_out1['gt_poly']
        gt_poly2 = record_out2['gt_poly']
        gt_poly = gt_poly1 + gt_poly2

        record_out = records_outs[0]
        record_out['image'] = im
        record_out['gt_class'] = gt_class
        record_out['gt_bbox'] = gt_bbox
        record_out['gt_poly'] = gt_poly

        return record_out