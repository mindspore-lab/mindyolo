import cv2
import numpy as np


def show_img_with_bbox(record, classes):
    """
    Image and labels visualization. If input multiple images, apply on the first image only.
    Args:
        record: related data of images
        classes: all categories of the whole dataset

    Returns: an image with detection boxes and categories
    """
    if isinstance(record, tuple):
        img = record[0]
        category_ids = record[4]
        bboxes = record[3]
    else:
        img = record['image'][0]
        category_ids = record['gt_class'][0]
        bboxes = record['gt_bbox'][0]
    categories = [classes[category_id[0]] for category_id in category_ids]
    bboxes = bboxes[category_ids[:, 0] >= 0]
    for bbox, category in zip(bboxes, categories):
        bbox = bbox.astype(np.int32)
        categories_size = cv2.getTextSize(category + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        color = ((np.random.random((3, )) * 0.6 + 0.4)*255).astype(np.uint8)
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