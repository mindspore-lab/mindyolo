import math
import random
import cv2
import numpy as np

__all__ = ['show_img_with_bbox', 'show_img_with_poly',
           'bbox_ioa', 'sample_polys', 'resample_polys',
           'poly2box', 'in_range', 'coco80_to_coco91_class',
           'normalize_shape', 'normalize_shape_with_poly']


def show_img_with_bbox(record, classes):
    """
    Image and bboxes visualization. If input multiple images, apply on the first image only.
    Args:
        record: related data of images
        classes: all categories of the whole dataset

    Returns: an image with detection boxes and categories
    """
    if isinstance(record, tuple):
        img = record[0]
        category_ids = record[7]
        bboxes = record[6]
    else:
        img = record['image'][0]
        category_ids = record['gt_class'][0]
        bboxes = record['gt_bbox'][0]
    categories = [classes[category_id[0]] for category_id in category_ids]
    bboxes = bboxes[category_ids[:, 0] >= 0]
    for bbox, category in zip(bboxes, categories):
        bbox = bbox.astype(np.int32)
        categories_size = cv2.getTextSize(category + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        color = ((np.random.random((3,)) * 0.6 + 0.4) * 255).astype(np.uint8)
        color = np.array(color).astype(np.int32).tolist()

        if bbox[1] - categories_size[1] - 3 < 0:
            cv2.rectangle(img,
                          (bbox[0], bbox[1] + 2),
                          (bbox[0] + categories_size[0], bbox[1] + categories_size[1] + 3),
                          color=color,
                          thickness=-1
                          )
            cv2.putText(img, category,
                        (bbox[0], bbox[1] + categories_size[1] + 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        thickness=1
                        )
        else:
            cv2.rectangle(img,
                          (bbox[0], bbox[1] - categories_size[1] - 3),
                          (bbox[0] + categories_size[0], bbox[1] - 3),
                          color,
                          thickness=-1
                          )
            cv2.putText(img, category,
                        (bbox[0], bbox[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        thickness=1
                        )
        cv2.rectangle(img,
                      (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]),
                      color,
                      thickness=2)
    return img


def show_img_with_poly(record):
    """
    Image and polygons visualization. If input multiple images, apply on the first image only.
    Args:
        record: related data of images

    Returns: an image with polygons
    """
    if isinstance(record, tuple):
        img = record[0]
        category_ids = record[7]
        polys = record[8]
    else:
        img = record['image'][0]
        category_ids = record['gt_class'][0]
        polys = record['gt_poly'][0]
    i = category_ids[:, 0] >= 0
    real_polys = []
    for j, value in enumerate(i):
        if value:
            real_polys.append(polys[j])
    for poly in real_polys:
        poly = poly.astype(np.int32)
        color = ((np.random.random((3,)) * 0.6 + 0.4) * 255).astype(np.uint8)
        color = np.array(color).astype(np.int32).tolist()
        img = cv2.drawContours(img, [poly], -1, color, 2)
    return img


def bbox_ioa(box1, box2):
    """
    Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
    """
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

    # Intersection over box2 area
    return inter_area / box2_area


def sample_polys(img, gt_bbox, gt_class, gt_poly, probability=0.5):
    """
    Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177
    """
    n = len(gt_poly)
    sample_images = []
    sample_masks = []
    sample_classes = []
    sample_polys = []
    if probability and n:
        h, w, c = img.shape  # height, width, channels
        for j in random.sample(range(n), k=round(probability * n)):
            b, c, g = gt_bbox[j], gt_class[j], gt_poly[j]
            box = b[0].astype(int).clip(0, w - 1), b[1].astype(int).clip(0, h - 1), \
                  b[2].astype(int).clip(0, w - 1), b[3].astype(int).clip(0, h - 1)

            # print(box)
            if (box[2] <= box[0]) or (box[3] <= box[1]):
                continue

            sample_classes.append(c)

            mask = np.zeros(img.shape, np.uint8)

            cv2.drawContours(mask, [gt_poly[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)
            sample_masks.append(mask[box[1]:box[3], box[0]:box[2], :])

            result = cv2.bitwise_and(src1=img, src2=mask)
            i = result > 0  # pixels to replace
            mask[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug
            sample_images.append(mask[box[1]:box[3], box[0]:box[2], :])

            new_poly = np.copy(g)
            relative_offset_w, relative_offset_h = box[0], box[1]
            new_poly[:, 0] -= relative_offset_w
            new_poly[:, 1] -= relative_offset_h

            sample_polys.append(new_poly)

    return sample_classes, sample_images, sample_masks, sample_polys


def resample_polys(polys, n=1000):
    """
    Up-sample an (n,2) segment
    """
    resample_result = []
    for i, s in enumerate(polys):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        resample_result.append(np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T)  # segment xy
    return resample_result


def poly2box(poly, width=640, height=640, consider_poly=False):
    """
    Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    """
    x, y = poly.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y = x[inside], y[inside]
    if consider_poly:
        poly[:, 0] = np.clip(poly[:, 0], 0, width)
        poly[:, 1] = np.clip(poly[:, 1], 0, width)
        return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4)), \
           poly if any(x) else np.zeros((1, 2))  # xyxy, poly
    else:
        return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))


def normalize_shape(images, gt_bboxes, gt_classes, batch_info):
    """
    Ensure labels have the same shape to avoid dynamics shapes
    """
    most_boxes_per_img = 0
    for gt_class in gt_classes:
        most_boxes_per_img = max(most_boxes_per_img, gt_class.shape[0])

    batch_idx = []
    for i, (gt_bbox, gt_class) in enumerate(zip(gt_bboxes, gt_classes)):
        nL = gt_class.shape[0]
        gt_bboxes[i] = np.full((most_boxes_per_img, 4), -1, dtype=np.float32)
        gt_classes[i] = np.full((most_boxes_per_img, 1), -1, dtype=np.int32)
        batch_idx.append(np.full((most_boxes_per_img, 1), i, dtype=np.int32))
        if nL:
            gt_bboxes[i][:nL, :] = gt_bbox[:nL, :]
            gt_classes[i][:nL, :] = gt_class[:nL, :]

    return np.stack(images, 0), np.stack(gt_bboxes, 0), np.stack(gt_classes, 0), np.stack(batch_idx, 0)


def normalize_shape_with_poly(images, gt_bboxes, gt_classes, gt_polys, batch_info):
    """
    Ensure labels have the same shape to avoid dynamics shapes
    """
    most_boxes_per_img = 0
    for gt_class in gt_classes:
        most_boxes_per_img = max(most_boxes_per_img, gt_class.shape[0])

    batch_idx = []
    for i, (gt_bbox, gt_class, gt_poly) in enumerate(zip(gt_bboxes, gt_classes, gt_polys)):
        nL = gt_class.shape[0]
        gt_bboxes[i] = np.full((most_boxes_per_img, 4), -1, dtype=np.float32)
        gt_bboxes[i][:nL, :] = gt_bbox[:nL, :]
        gt_classes[i] = np.full((most_boxes_per_img, 1), -1, dtype=np.int32)
        gt_classes[i][:nL, :] = gt_class[:nL, :]
        batch_idx.append(np.full((most_boxes_per_img, 1), i, dtype=np.int32))
        gt_polys[i] = np.full((most_boxes_per_img, 100, 2), -1, dtype=np.float32)
        gt_polys[i][:nL, :] = gt_poly[:nL]

    return np.stack(images, 0), np.stack(gt_bboxes, 0), np.stack(gt_classes, 0), np.stack(gt_polys, 0), np.stack(batch_idx, 0)


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def in_range(n, start, end=0):
    return start <= n <= end if end >= start else end <= n <= start


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size
