---
hide:
  - toc
---

# YOLOv3

> [YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)

## Abstract
We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that's pretty swell. It's a little bigger than last time but more accurate. It's still fast though, don't worry. At 320x320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 mAP@50 in 51 ms on a Titan X, compared to 57.5 mAP@50 in 198 ms by RetinaNet, similar performance but 3.8x faster. 

<div align=center>
<img src="https://raw.githubusercontent.com/zhanghuiyao/pics/main/mindyolo202304071143644.png"/>
</div>

## Results

<details open markdown>
<summary><b>performance tested on Ascend 910(8p) with graph mode</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---: |                :---:                         |        :---:       |
| YOLOv3 | Darknet53          |  16 * 8   |    640    | MS COCO 2017 |    45.5     | 61.9M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov3/yolov3.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-adfb27af.ckpt)         |
</details>

<details open markdown>
<summary><b>performance tested on Ascend 910*(8p)</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | ms/step | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---:  |  :---: |                :---:                         |        :---:       |
| YOLOv3 | Darknet53          |  16 * 8   |    640    | MS COCO 2017 |     46.6    | 396.60  | 61.9M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov3/yolov3.yaml)         | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-81895f09-910v2.ckpt)         |
</details>

<br>

#### Notes

- Box mAP: Accuracy reported on the validation set.
- We referred to a commonly used third-party [YOLOv3](https://github.com/ultralytics/yolov3) implementation.

## Quick Start

Please refer to the [QUICK START](../tutorials/quick_start.md) in MindYOLO for details.

### Training


#### - Pretraining Model

You can get the pre-training model from [here](https://pjreddie.com/media/files/darknet53.conv.74).

To convert it to a loadable ckpt file for mindyolo, please put it in the root directory then run it
```shell
python mindyolo/utils/convert_weight_darknet53.py
```

#### - Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run
```shell
# distributed training on multiple Ascend devices
msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov3_log python train.py --config ./configs/yolov3/yolov3.yaml --device_target Ascend --is_parallel True
```

**Note:** For more information about msrun configuration, please refer to [here](https://www.mindspore.cn/docs/en/r2.5.0/model_train/parallel/msrun_launcher.html).

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

#### - Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/Ascend device
python train.py --config ./configs/yolov3/yolov3.yaml --device_target Ascend
```



### Validation and Test

To validate the accuracy of the trained model, you can use `test.py` and parse the checkpoint path with `--weight`.

```
python test.py --config ./configs/yolov3/yolov3.yaml --device_target Ascend --weight /PATH/TO/WEIGHT.ckpt
```

### Deployment

See [here](../tutorials/deployment.md).

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Jocher Glenn. YOLOv3 release v9.1. https://github.com/ultralytics/yolov3/releases/tag/v9.1, 2021.
[2] Joseph Redmon and Ali Farhadi. YOLOv3: An incremental improvement. arXiv preprint arXiv:1804.02767, 2018.
