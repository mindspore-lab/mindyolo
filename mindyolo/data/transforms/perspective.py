import random
import math
import cv2
import numpy as np

from ..general import resample_polys, poly2box

__all__ = ['RandomPerspective']


class RandomPerspective:
    """
    Args:
        degrees (float): the rotate range to apply, transform range is [-10, 10]
        translate (float): the translate range to apply, transform range is [-0.1, 0.1]
        scale (float): the scale range to apply, transform range is [0.1, 2]
        shear (float): the shear range to apply, transform range is [-2, 2]
        perspective (float): the perspective range to apply, transform range is [0, 0.001]
        border: border to remove
        consider_poly(bool): whether to consider the change of gt_poly
    """
    def __init__(self, degrees=0.0, translate=.2, scale=.9, shear=0.0, perspective=0.0, border=(0, 0), consider_poly=False):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border
        self.consider_poly = consider_poly

    def __call__(self, img, gt_bbox, gt_class, gt_poly=[]):
        height = img.shape[0] + self.border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + self.border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1.1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # Transform label coordinates
        n = len(gt_bbox)
        if n:
            use_segments = len(gt_poly) > 0
            new_bbox = np.zeros((n, 4))
            if self.consider_poly:
                new_poly = [np.zeros((1000, 2))] * n
            if use_segments:
                gt_poly = resample_polys(gt_poly)  # upsample
                for i, poly in enumerate(gt_poly):
                    xy = np.ones((len(poly), 3))
                    xy[:, :2] = poly
                    xy = xy @ M.T  # transform
                    xy = xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]  # perspective rescale or affine

                    # clip
                    if self.consider_poly:
                        new_bbox[i], new_poly[i] = poly2box(xy, width, height, self.consider_poly)
                    else:
                        new_bbox[i] = poly2box(xy, width, height, self.consider_poly)
            else:
                xy = np.ones((n * 4, 3))
                xy[:, :2] = gt_bbox[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = xy @ M.T  # transform
                xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                new_bbox = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # clip
                new_bbox[:, [0, 2]] = new_bbox[:, [0, 2]].clip(0, width)
                new_bbox[:, [1, 3]] = new_bbox[:, [1, 3]].clip(0, height)

            # filter candidates
            i = box_candidates(box1=gt_bbox.T * s, box2=new_bbox.T, area_thr=0.01 if use_segments else 0.10)
            gt_class = gt_class[i]
            gt_bbox = new_bbox[i]

            # filter candidates for poly
            if self.consider_poly:
                gt_poly = []
                for j, value in enumerate(i):
                    if value:
                        gt_poly.append(new_poly[j])
                return img, gt_bbox, gt_class, gt_poly
            else:
                return img, gt_bbox, gt_class


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
