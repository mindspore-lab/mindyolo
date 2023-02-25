import random
import cv2
import numpy as np

from ..general import bbox_ioa, sample_polys

__all__ = ['PasteIn']


class PasteIn:
    """
    Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    """
    def __init__(self, prob=0.15, target_size=640):
        self.prob = prob
        self.target_size = target_size
        self.mosaic_border = [-target_size // 2, -target_size // 2]

    def __call__(self, records_outs):
        # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
        if random.random() > self.prob:
            return records_outs[0]

        image = records_outs[0]['image']
        gt_bbox = records_outs[0]['gt_bbox']
        gt_class = records_outs[0]['gt_class']
        gt_poly = records_outs[0]['gt_poly']

        sample_classes, sample_images, sample_masks, sample_polys = [], [], [], []
        i = 0
        sample_records_outs = records_outs[1:]
        sample_records_len = len(records_outs[1:])
        while len(sample_classes) < 30:
            record_out = [sample_records_outs[(i + _i) % sample_records_len] for _i in range(4)]
            sample_classes_, sample_images_, sample_masks_, sample_polys_ = \
                self.load_samples(record_out)
            sample_classes += sample_classes
            sample_images += sample_images_
            sample_masks += sample_masks_
            sample_polys += sample_polys_
            if len(sample_classes) == 0:
                break

        image, gt_bbox, gt_class, gt_poly = \
            self.pastein(image, gt_bbox, gt_class, gt_poly, sample_classes, sample_images, sample_masks, sample_polys)

        record_out = records_outs[0]
        record_out['image'] = image
        record_out['gt_bbox'] = gt_bbox
        record_out['gt_class'] = gt_class
        record_out['gt_poly'] = gt_poly

        return record_out

    def load_samples(self, records_outs):
        # loads images in a 4-mosaic
        assert len(records_outs) == 4
        img_size = self.target_size
        gt_bboxes4, gt_classes4, gt_polys4,  = [], [], []
        yc, xc = [int(random.uniform(-x, 2 * img_size + x)) for x in self.mosaic_border]  # mosaic center x, y
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
                for j, poly in enumerate(gt_poly):
                    gt_poly[j] = poly * r

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
            np.clip(x, 0, 2 * img_size, out=x)

        # Augment
        sample_classes, sample_images, sample_masks, sample_poly = \
            sample_polys(img4, gt_bboxes4, gt_classes4, gt_polys4, probability=0.5)

        return sample_classes, sample_images, sample_masks, sample_poly

    def pastein(
            self,
            image,
            gt_bbox, gt_class, gt_poly,
            sample_classes, sample_images, sample_masks, sample_polys
    ):

        h, w = image.shape[:2]

        # create random masks
        scales = [0.75] * 2 + [0.5] * 4 + [0.25] * 4 + [0.125] * 4 + [0.0625] * 6  # image size fraction
        for s in scales:
            if np.random.random() < 0.2:
                continue
            mask_h = random.randint(1, int(h * s))
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            if gt_bbox.shape[0] > 0:
                ioa = bbox_ioa(box, gt_bbox)  # intersection over area
            else:
                ioa = np.zeros(1)

            if (ioa < 0.30).all() and len(sample_classes) and (xmax > xmin + 20) and (
                    ymax > ymin + 20):  # allow 30% obscuration of existing labels

                sel_ind = random.randint(0, len(sample_classes) - 1)
                hs, ws, cs = sample_images[sel_ind].shape
                r_scale = min((ymax - ymin) / hs, (xmax - xmin) / ws)
                r_w = int(ws * r_scale)
                r_h = int(hs * r_scale)

                if (r_w > 10) and (r_h > 10):
                    r_mask = cv2.resize(sample_masks[sel_ind], (r_w, r_h))
                    r_image = cv2.resize(sample_images[sel_ind], (r_w, r_h))
                    r_poly = sample_polys[sel_ind] * r_scale
                    r_poly[:, 0] += xmin
                    r_poly[:, 1] += ymin + r_h
                    temp_crop = image[ymin:ymin + r_h, xmin:xmin + r_w]
                    m_ind = r_mask > 0
                    if m_ind.astype(np.int).sum() > 60:
                        temp_crop[m_ind] = r_image[m_ind]
                        box = np.array([xmin, ymin, xmin + r_w, ymin + r_h], dtype=np.float32)

                        if gt_bbox.shape[0] > 0:
                            gt_bbox = np.concatenate((gt_bbox, [[*box]]), 0)
                            gt_class = np.concatenate((gt_class, [sample_classes[sel_ind]]), 0)
                            gt_poly.append(r_poly)
                        else:
                            gt_bbox = np.array([[*box]])
                            gt_class = np.array([sample_classes[sel_ind]])
                            gt_poly = [r_poly]

                        image[ymin:ymin + r_h, xmin:xmin + r_w] = temp_crop

        return image, gt_bbox, gt_class, gt_poly
