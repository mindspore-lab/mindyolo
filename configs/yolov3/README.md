# YOLOv3

> [YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)

## Abstract
We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that's pretty swell. It's a little bigger than last time but more accurate. It's still fast though, don't worry. At 320x320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 mAP@50 in 51 ms on a Titan X, compared to 57.5 mAP@50 in 198 ms by RetinaNet, similar performance but 3.8x faster. 

<div align=center>
<img src="https://raw.githubusercontent.com/zhanghuiyao/pics/main/mindyolo202304071143644.png"/>
</div>

## Requirements

| mindspore | ascend driver | firmware     | cann toolkit/kernel |
| :-------: | :-----------: | :----------: |:-------------------:|
| 2.3.1     | 24.1.RC2      | 7.3.0.1.231  |   8.0.RC2.beta1     |

## Quick Start

Please refer to the [GETTING_STARTED](https://github.com/mindspore-lab/mindyolo/blob/master/GETTING_STARTED.md) in MindYOLO for details.

### Training

<details open markdown>
<summary><b>View More</b></summary>

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
msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov3_log python train.py --config ./configs/yolov3/yolov3.yaml --device_target Ascend --is_parallel True
```

Similarly, you can train the model on multiple GPU devices with the above msrun command.
**Note:** For more information about msrun configuration, please refer to [here](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.1/parallel/msrun_launcher.html)

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

## Performance

Experiments are tested on Ascend 910* with mindspore 2.3.1 graph mode.

| model name | cards | batch size | resolution | jit level | graph compile | ms/step | img/s  |  map  |        recipe         |                                                           weight                                                           |
| :--------: | :---: | :--------: | :--------: | :-------: | :-----------: | :-----: | :----: | :---: | :-------------------: | :------------------------------------------------------------------------------------------------------------------------: |
|   YOLOv3   |   8   |     16     |  640x640   |    O2     |    274.32s    | 383.68  | 333.61 | 46.6% | [yaml](./yolov3.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-81895f09-910v2.ckpt) |


Experiments are tested on Ascend 910 with mindspore 2.3.1 graph mode.

| model name | cards | batch size | resolution | jit level | graph compile | ms/step | img/s  |  map  |        recipe         |                                                    weight                                                    |
| :--------: | :---: | :--------: | :--------: | :-------: | :-----------: | :-----: | :----: | :---: | :-------------------: | :----------------------------------------------------------------------------------------------------------: |
|   YOLOv3   |   8   |     16     |  640x640   |    O2     |    160.80s    | 409.66  | 312.45 | 45.5% | [yaml](./yolov3.yaml) | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-adfb27af.ckpt) |


<br>

### Notes

- map: Accuracy reported on the validation set.
- We referred to a commonly used third-party [YOLOv3](https://github.com/ultralytics/yolov3) implementation.


## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Jocher Glenn. YOLOv3 release v9.1. https://github.com/ultralytics/yolov3/releases/tag/v9.1, 2021.
[2] Joseph Redmon and Ali Farhadi. YOLOv3: An incremental improvement. arXiv preprint arXiv:1804.02767, 2018.
