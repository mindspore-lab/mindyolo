import cv2
import numpy as np

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


def show_img_with_label(img_and_record, is_coco=True):
    img, record = img_and_record
    category_ids = record['gt_class'][0]
    categories = [CLASSES[category_id[0]] for category_id in category_ids]
    bboxes = record['gt_bbox'][0]
    for bbox, category in zip(bboxes, categories):
        bbox = bbox.astype(np.int32)
        labelSize = cv2.getTextSize(category + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        color = ((np.random.random((3, )) * 0.6 + 0.4)*255).astype(np.uint8)
        color = np.array(color).astype(np.int32).tolist()

        if bbox[1] - labelSize[1] - 3 < 0:
            cv2.rectangle(img,
                          (bbox[0], bbox[1] + 2),
                          (bbox[0] + labelSize[0], bbox[1] + labelSize[1] + 3),
                          color=color,
                          thickness=-1
                          )
            cv2.putText(img, category,
                        (bbox[0], bbox[1] + labelSize[1] + 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        thickness=1
                        )
        else:
            cv2.rectangle(img,
                          (bbox[0], bbox[1] - labelSize[1] - 3),
                          (bbox[0] + labelSize[0], bbox[1] - 3),
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


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[2] = x[0] + x[2]
    y[3] = x[1] + x[3]
    return y