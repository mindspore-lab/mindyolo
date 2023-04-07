# MindYOLO Model Zoo and Baselines

| Name   | Scale     | Context  | ImageSize | Dataset      | Box mAP (%) | Params  | FLOPs  | Recipe                                                                                        | Download                                                                                            |
|--------|-----------|----------|-----------|--------------|-------------|---------|--------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| YOLOv7 | Tiny      | D910x8-G | 640       | MS COCO 2017 | 37.5        | 6.2M    | 13.8G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7-tiny.yaml) | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-Tiny_300e_mAP375.ckpt)      |
| YOLOv7 | L         | D910x8-G | 640       | MS COCO 2017 | 50.8        | 36.9M   | 104.7G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7.yaml)      | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-L_300e_mAP508.ckpt)         |
| YOLOv7 | X         | D910x8-G | 640       | MS COCO 2017 | 52.4        | 71.3M   | 189.9G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7-x.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-X_300e_mAP524.ckpt)         |
| YOLOv5 | N         | D910x8-G | 640       | MS COCO 2017 | 27.3        | 1.9M    | 4.5G   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5n.yaml)     | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5-N_300e_mAP273.ckpt)         |
| YOLOv5 | S         | D910x8-G | 640       | MS COCO 2017 | 37.6        | 7.2M    | 16.5G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5s.yaml)     | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5-S_300e_mAP376.ckpt)         |
| YOLOv5 | M         | D910x8-G | 640       | MS COCO 2017 | 44.9        | 21.2M   | 49.0G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5m.yaml)     | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5-M_300e_mAP449.ckpt)         |
| YOLOv5 | L         | D910x8-G | 640       | MS COCO 2017 | 48.5        | 46.5M   | 109.1G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5l.yaml)     | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5-L_300e_mAP485.ckpt)         |
| YOLOv3 | Darknet53 | D910x8-G | 640       | MS COCO 2017 | 43.8        | 61.9M   | 156.4G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov3/yolov3.yaml)      | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov3/yolov3-Darknet53_300e_mAP438.ckpt) |

<br>

#### Notes
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Box mAP: Accuracy reported on the validation set.
