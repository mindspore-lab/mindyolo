# YOLOv5

## Abstract
YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset, representing Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.
<div align=center>
<img src="https://raw.githubusercontent.com/zhanghuiyao/pics/main/mindyolo20230407113509.png"/>
</div>

## Results

<details open markdown>
<summary><b>performance tested on Ascend 910(8p) with graph mode</b></summary>

|  model name  |  scale  | cards  | batch size | resolution |  jit level  | graph compile | Box mAP (%) |           recipe           | weight                                                                       |
|  :--------:  |  :---:  |  :---: |    :---:   |   :---:    |    :---:    |     :---:     |    :---:    |           :---:            |        :---:       |
|   YOLOv5     |    N    |    8   |    32      |  640x640   |     O2      |    3~5 mins     |    27.3     |   [yaml](./yolov5n.yaml)   | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-9b16bd7b.ckpt)                  |
|   YOLOv5     |    S    |    8   |    32      |  640x640   |     O2      |    3~5 mins     |    37.6     |   [yaml](./yolov5s.yaml)   | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-860bcf3b.ckpt)                  |
|   YOLOv5     |    M    |    8   |    32      |  640x640   |     O2      |    4~6 mins     |    44.9     |   [yaml](./yolov5m.yaml)   | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5m_300e_mAP449-e7bbf695.ckpt)                  |
|   YOLOv5     |    L    |    8   |    32      |  640x640   |     O2      |    5~7 mins     |    48.5     |   [yaml](./yolov5l.yaml)   | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5l_300e_mAP485-a28bce73.ckpt)                  |
|   YOLOv5     |    X    |    8   |    16      |  640x640   |     O2      |    8~10 mins    |    50.5     |   [yaml](./yolov5x.yaml)   | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5x_300e_mAP505-97d36ddc.ckpt)                  |
</details>

<details open markdown>
<summary><b>performance tested on Ascend 910*(8p) with graph mode</b></summary>

| model name   |  scale  | cards  | batch size | resolution |  jit level  | graph compile | Box mAP (%) | ms/step |           recipe           | weight                                                                                                |
|  :--------:  |  :---:  |  :---: |   :---:    |   :---:    |    :---:    |     :---:     |    :---:    |  :---:  |           :---:            |        :---:       |
|   YOLOv5     |    N    |    8   |    32      |  640x640   |     O2      |    3~5 mins     |     27.4    | 736.08  |   [yaml](./yolov5n.yaml)   | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-bedf9a93-910v2.ckpt)       |
|   YOLOv5     |    S    |    8   |    32      |  640x640   |     O2      |    3~5 mins     |     37.6    | 787.34  |   [yaml](./yolov5s.yaml)   | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-df4a45b6-910v2.ckpt)       |
</details>

<br>

#### Notes

- Box mAP: Accuracy reported on the validation set.
- We refer to the official [YOLOV5](https://github.com/ultralytics/yolov5) to reproduce the P5 series model, and the differences are as follows:
  1. We use 8x NPU(Ascend910) for training, and the single-NPU batch size is 32. This is different from the official code.

## Quick Start

Please refer to the [GETTING_STARTED](https://github.com/mindspore-lab/mindyolo/blob/master/GETTING_STARTED.md) in MindYOLO for details.

### Requirements

| mindspore | ascend driver | firmware     | cann toolkit/kernel
| :-------: | :-----------: | :----------: | :----------------:
| 2.3.1     | 24.1.RC2      | 7.3.0.1.231  | 8.0.RC2.beta1

### Training

<details open>
<summary><b>View More</b></summary>

#### - Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run
```shell
# distributed training on multiple GPU/Ascend devices
msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov5_log  python train.py --config ./configs/yolov5/yolov5n.yaml --device_target Ascend --is_parallel True
```

Similarly, you can train the model on multiple GPU devices with the above msrun command.
**Note:** For more information about msrun configuration, please refer to [here](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.1/parallel/msrun_launcher.html)

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

#### - Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config ./configs/yolov5/yolov5n.yaml --device_target Ascend
```

</details>

### Validation and Test

To validate the accuracy of the trained model, you can use `test.py` and parse the checkpoint path with `--weight`.

```
python test.py --config ./configs/yolov5/yolov5n.yaml --device_target Ascend --weight /PATH/TO/WEIGHT.ckpt
```

### Deployment

See [here](../../deploy/README.md).

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Jocher Glenn. YOLOv5 release v6.1. https://github.com/ultralytics/yolov5/releases/tag/v6.1, 2022.
