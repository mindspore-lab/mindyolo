# YOLOv3

> [YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)

## Abstract
We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that's pretty swell. It's a little bigger than last time but more accurate. It's still fast though, don't worry. At 320x320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 mAP@50 in 51 ms on a Titan X, compared to 57.5 mAP@50 in 198 ms by RetinaNet, similar performance but 3.8x faster. 

<div align=center>
<img src="https://raw.githubusercontent.com/zhanghuiyao/pics/main/mindyolo202304071143644.png"/>
</div>

## Results

<div align="center">

| Name   | Scale     | Context  | ImageSize | Dataset      | Box mAP (%) | Params  | FLOPs  | Recipe                                                                                        | Download                                                                                                      |
|--------|-----------|----------|-----------|--------------|-------------|---------|--------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| YOLOv3 | Darknet53 | D910x8-G | 640       | MS COCO 2017 | 45.5        | 61.9M   | 156.4G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov3/yolov3.yaml)      | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-adfb27af.ckpt)  |

</div>
<br>

#### Notes

- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Box mAP: Accuracy reported on the validation set.
- We referred to a commonly used third-party [YOLOv3](https://github.com/ultralytics/yolov3) implementation.

## Quick Start

Please refer to the [GETTING_STARTED](https://github.com/mindspore-lab/mindyolo/blob/master/GETTING_STARTED.md) in MindYOLO for details.

### Training

<details open>

#### - Pretraining Model

You can get the pre-training model from [here](https://pjreddie.com/media/files/darknet53.conv.74).

To convert it to a loadable ckpt file for mindyolo, please put it in the root directory then run it
```shell
python mindyolo/utils/convert_weight_darknet53.py
```

#### - Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run
```shell
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config ./configs/yolov3/yolov3.yaml --device_target Ascend --is_parallel True
```
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above mpirun command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

#### - Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config ./configs/yolov3/yolov3.yaml --device_target Ascend
```

</details>

### Validation and Test

To validate the accuracy of the trained model, you can use `test.py` and parse the checkpoint path with `--weight`.

```
python test.py --config ./configs/yolov3/yolov3.yaml --device_target Ascend --weight /PATH/TO/WEIGHT.ckpt
```

### Deployment

See [here](../../deploy/README.md).

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Jocher Glenn. YOLOv3 release v9.1. https://github.com/ultralytics/yolov3/releases/tag/v9.1, 2021.
[2] Joseph Redmon and Ali Farhadi. YOLOv3: An incremental improvement. arXiv preprint arXiv:1804.02767, 2018.
