import sys

sys.path.append(".")
import os
import pytest
from download import download
import cv2

from mindyolo.utils.config import load_config, Config
from mindyolo.data import COCO80_TO_COCO91_CLASS
from mindyolo.utils.utils import draw_result
from demo.predict import detect
from mindyolo.models.model_factory import create_model


@pytest.mark.parametrize("yaml_name", ['yolov5n.yaml'])
def test_demo_predict(yaml_name):
    image_url = 'https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/image_cat.zip'
    path = download(image_url, './', kind="zip", replace=True)
    image_path = ('./image_cat/jpg/000000039769.jpg')

    model_url = 'https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-9b16bd7b.ckpt'
    weight = download(model_url, './yolov5n.ckpt', kind="file", replace=True)
    parent_dir = yaml_name[:6]
    yaml_path = os.path.join('./configs', parent_dir, yaml_name)
    cfg, _, _ = load_config(yaml_path)
    cfg = Config(cfg)

    network = create_model(
        model_name=cfg.network.model_name,
        model_cfg=cfg.network,
        num_classes=cfg.data.nc,
        sync_bn=False,
        checkpoint_path=weight,
    )
    network.set_train(False)
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
    test_demo_predict('yolov5n.yaml')
