import sys

sys.path.append(".")
import os
import numpy as np
from download import download
import cv2

from mindspore import nn

from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh
from mindyolo.data import COCO80_TO_COCO91_CLASS
from deploy.infer_engine.lite import LiteModel
from mindyolo.utils.utils import draw_result
from deploy.predict import detect

def test_deploy_predict():
    image_url = 'https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/image_cat.zip'
    path = download(image_url, './', kind="zip", replace=True)
    image_path = ('./image_cat/jpg/000000039769.jpg')

    model_url = 'https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-9b16bd7b-bd03027b.mindir'
    model_path = download(model_url, './yolov5n.mindir', kind="file", replace=True)

    network = LiteModel(model_path)
    img = cv2.imread(image_path)
    result_dict = detect(
        network=network,
        img=img,
        conf_thres=0.1,
        iou_thres=0.65,
        conf_free=False,
        nms_time_limit=20,
        img_size=640,
        is_coco_dataset=True,
    )
    save_path = os.path.join('./', "detect_results")
    names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush' ]
    draw_result(image_path, result_dict, names, save_path=save_path)
    assert names[COCO80_TO_COCO91_CLASS.index(result_dict['category_id'][0])]=='cat'


if __name__ == '__main__':
    test_deploy_predict()
