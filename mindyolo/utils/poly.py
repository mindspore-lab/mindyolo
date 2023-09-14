import cv2
import numpy as np

from mindyolo.data.utils import xywhn2xyxy


def show_img_with_bbox(data_dict, classes):
    """
    Image and bboxes visualization. If input multiple images, apply on the first image only.
    Args:
        record: related data of images
        classes: all categories of the whole dataset

    Returns: an image with detection boxes and categories
    """
    img, labels = data_dict["images"][0], data_dict["labels"][0]
    img = img.transpose(1, 2, 0)[:, :, ::-1] * 255.0
    img = np.ascontiguousarray(img, dtype=np.uint8)
    labels = labels[labels[:, 1] > 0]  # filter invalid label
    category_ids = labels[:, 1]
    bboxes = labels[:, 2:]

    categories = [classes[int(category_id)] for category_id in category_ids]
    bboxes = xywhn2xyxy(bboxes[category_ids >= 0])
    for bbox, category in zip(bboxes, categories):
        bbox = bbox.astype(np.int32)
        categories_size = cv2.getTextSize(category + "0", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        color = ((np.random.random((3,)) * 0.6 + 0.4) * 255).astype(np.uint8)
        color = np.array(color).astype(np.int32).tolist()

        if bbox[1] - categories_size[1] - 3 < 0:
            cv2.rectangle(
                img,
                (bbox[0], bbox[1] + 2),
                (bbox[0] + categories_size[0], bbox[1] + categories_size[1] + 3),
                color=color,
                thickness=-1,
            )
            cv2.putText(
                img,
                category,
                (bbox[0], bbox[1] + categories_size[1] + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                thickness=1,
            )
        else:
            cv2.rectangle(
                img,
                (bbox[0], bbox[1] - categories_size[1] - 3),
                (bbox[0] + categories_size[0], bbox[1] - 3),
                color,
                thickness=-1,
            )
            cv2.putText(img, category, (bbox[0], bbox[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness=2)
    return img
