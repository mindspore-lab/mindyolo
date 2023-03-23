import cv2
import numpy as np


def show_img_with_bbox(data_dict, classes):
    """
    Image and bboxes visualization. If input multiple images, apply on the first image only.
    Args:
        record: related data of images
        classes: all categories of the whole dataset

    Returns: an image with detection boxes and categories
    """
    img, labels = data_dict['image'][0], data_dict['labels'][0]
    labels = labels[labels[:, 1] > 0]  # filter invalid label
    category_ids = labels[:, 1]
    bboxes = labels[:, 2:]

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


def show_img_with_poly(data_dict):
    """
    Image and polygons visualization. If input multiple images, apply on the first image only.
    Args:
        record: related data of images

    Returns: an image with polygons
    """
    img, labels, polys = data_dict['image'][0], data_dict['labels'][0], data_dict['segments'][0]
    labels = labels[labels[:, 1] > 0]  # filter invalid label
    category_ids = labels[:, 1]

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
