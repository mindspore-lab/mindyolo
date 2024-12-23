# YOLOv8

## Abstract
Ultralytics YOLOv8, developed by Ultralytics, is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection, image segmentation and image classification tasks.

<div align=center>
<img src="https://raw.githubusercontent.com/zhanghuiyao/pics/main/mindyolomindyolo-yolov8-comparison-plots.png"/>
</div>

## Requirements

| mindspore | ascend driver | firmware     | cann toolkit/kernel |
| :-------: | :-----------: | :----------: |:-------------------:|
| 2.3.1     | 24.1.RC2      | 7.3.0.1.231  |   8.0.RC2.beta1     |

## Quick Start

Please refer to the [GETTING_STARTED](https://github.com/mindspore-lab/mindyolo/blob/master/GETTING_STARTED.md) in MindYOLO for details.

### Training

<details open>
<summary><b>View More</b></summary>

#### - Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run
```shell
# distributed training on multiple Ascend devices
msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov8_log python train.py --config ./configs/yolov8/yolov8n.yaml --device_target Ascend --is_parallel True
```

**Note:** For more information about msrun configuration, please refer to [here](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.1/parallel/msrun_launcher.html).

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

#### - Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/Ascend device
python train.py --config ./configs/yolov8/yolov8n.yaml --device_target Ascend
```

</details>

### Validation and Test

To validate the accuracy of the trained model, you can use `test.py` and parse the checkpoint path with `--weight`.

```
python test.py --config ./configs/yolov8/yolov8n.yaml --device_target Ascend --weight /PATH/TO/WEIGHT.ckpt
```

## Performance


### Detection


Experiments are tested on Ascend 910* with mindspore 2.3.1 graph mode.

|  model name  |  scale  | cards  | batch size | resolution |  jit level  | graph compile | ms/step | img/s  |  map  |          recipe              |                                                       weight                                                       |
|  :--------:  |  :---:  |  :---: |   :---:    |   :---:    |    :---:    |     :---:     |  :---:  |  :---: |:-----:|          :---:               |:------------------------------------------------------------------------------------------------------------------:|
|    YOLOv8    |    N    |    8   |     16     |  640x640   |     O2      |    145.89s    | 252.79  | 506.35 | 37.3% |    [yaml](./yolov8n.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov8/yolov8-n_500e_mAP372-0e737186-910v2.ckpt) |
|    YOLOv8    |    S    |    8   |     16     |  640x640   |     O2      |    172.22s    | 251.30  | 509.35 | 44.7% |    [yaml](./yolov8s.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov8/yolov8-s_500e_mAP446-fae4983f-910v2.ckpt) |


Experiments are tested on Ascend 910 with mindspore 2.3.1 graph mode.

|  model name  |  scale  | cards  | batch size | resolution |  jit level  | graph compile | ms/step | img/s |  map  |            recipe            |                                                weight                                                |
|  :--------:  |  :---:  |  :---: |   :---:    |   :---:    |    :---:    |     :---:     |  :---: | :---:  |:-----:|            :---:             |:----------------------------------------------------------------------------------------------------:|
|    YOLOv8    |    N    |    8   |     16     |  640x640   |     O2      |    195.63s    | 265.13 | 482.78 | 37.2% |    [yaml](./yolov8n.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-n_500e_mAP372-cc07f5bd.ckpt) |
|    YOLOv8    |    S    |    8   |     16     |  640x640   |     O2      |    115.60s    | 292.68 | 437.34 | 44.6% |    [yaml](./yolov8s.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-s_500e_mAP446-3086f0c9.ckpt) |
|    YOLOv8    |    M    |    8   |     16     |  640x640   |     O2      |    185.25s    | 383.72 | 333.58 | 50.5% |    [yaml](./yolov8m.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-m_500e_mAP505-8ff7a728.ckpt) |
|    YOLOv8    |    L    |    8   |     16     |  640x640   |     O2      |    175.08s    | 429.02 | 298.35 | 52.8% |    [yaml](./yolov8l.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-l_500e_mAP528-6e96d6bb.ckpt) |
|    YOLOv8    |    X    |    8   |     16     |  640x640   |     O2      |    183.68s    | 521.97 | 245.22 | 53.7% |    [yaml](./yolov8x.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-x_500e_mAP537-b958e1c7.ckpt) |




### Segmentation


Experiments are tested on Ascend 910 with mindspore 2.3.1 graph mode.

*coming soon*

Experiments are tested on Ascend 910 with mindspore 2.3.1 graph mode.

|  model Name  |  scale  | cards  | batch size | resolution |  jit level  | graph compile | ms/step | img/s  |  map  | mask map |              recipe                  |                                                     weight                                                     |
|  :--------:  |  :---:  |  :---: |   :---:    |   :---:    |    :---:    |     :---:     |  :---:  |  :---: |:-----:|:--------:|              :---:                   |:--------------------------------------------------------------------------------------------------------------:|
|  YOLOv8-seg  |    X    |    8   |     16     |  640x640   |     O2      |    183.68s    | 641.25  | 199.61 | 52.5% |  42.9%   |    [yaml](./seg/yolov8x-seg.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-x-seg_300e_mAP_mask_429-b4920557.ckpt) |

### Notes

- map: Accuracy reported on the validation set.
- We refer to the official [YOLOV8](https://github.com/ultralytics/ultralytics) to reproduce the P5 series model.

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Jocher Glenn. Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics, 2023.
