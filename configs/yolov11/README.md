# YOLOv11

## Abstract
Ultralytics YOLO11 is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLO11 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and tracking, instance segmentation, image classification and pose estimation tasks.

<div align=center>
<img src="https://github.com/user-attachments/assets/10b2a1f7-b75c-40fe-8cc2-59e21c2d4d08"/>
</div>

## Requirements

| mindspore | ascend driver | firmware     | cann toolkit/kernel |
| :-------: | :-----------: | :----------: |:-------------------:|
| 2.5.0     | 24.1.0      | 7.5.0.3.220  |   8.0.0.beta1     |

## Quick Start

Please refer to the [GETTING_STARTED](https://github.com/mindspore-lab/mindyolo/blob/master/GETTING_STARTED.md) in MindYOLO for details.

### Training

<details open>
<summary><b>View More</b></summary>

#### - Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run
```shell
# distributed training on multiple Ascend devices
msrun --worker_num=3 --local_worker_num=3 --bind_core=True --log_dir=./yolov11_log python train.py --config ./configs/yolov11/yolov11-n.yaml --device_target Ascend --is_parallel True
```

**Note:** For more information about msrun configuration, please refer to [here](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.1/parallel/msrun_launcher.html).

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

#### - Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/Ascend device
python train.py --config ./configs/yolov11/yolov11-n.yaml --device_target Ascend
```

</details>

### Validation and Test

To validate the accuracy of the trained model, you can use `test.py` and parse the checkpoint path with `--weight`.

```
python test.py --config ./configs/yolov11/yolov11-n.yaml --device_target Ascend --weight /PATH/TO/WEIGHT.ckpt
```

## Performance


### Detection

Experiments are tested on Ascend 910* with mindspore 2.5.0 graph mode.

|  model name  |  scale  | cards  | batch size | resolution |  jit level  | ms/step | img/s |  map  |            recipe            |                                                weight                                                |
|  :--------:  |  :---:  |  :---: |   :---:    |   :---:    |    :---:    | :---: | :---:  |:-----:|            :---:             |:----------------------------------------------------------------------------------------------------:|
|    YOLOv11    |    N    |    1   |     128     |  640x640   |     O2      | 383.78 | 333.52 | 39.2% |    [yaml](./yolov11-n.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov11/yolov11n_600e_MAP392-78fd292c.ckpt) |
|    YOLOv11    |    S    |    1   |     128     |  640x640   |     O2      | 488.65 | 261.95 | 46.4% |    [yaml](./yolov11-s.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov11/yolov11s_600e_MAP464-26f6efa4.ckpt) |
|    YOLOv11    |    M    |    1   |     108     |  640x640   |     O2      | 721.72 | 149.64 | 51.1% |    [yaml](./yolov11-m.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov11/yolov11m_600e_MAP511-94a7cf04.ckpt) |
|    YOLOv11    |    L    |    2   |     64     |  640x640   |     O2      | 637.84 | 200.68 | 52.6% |    [yaml](./yolov11-l.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov11/yolov11l_600e_MAP526-48494760.ckpt) |
|    YOLOv11    |    X    |    3   |     43     |  640x640   |     O2      | 622.68 | 207.17 | 54.2% |    [yaml](./yolov11-x.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov11/yolov11x_600e_MAP542-19131881.ckpt) |

### Notes

- map: Accuracy reported on the validation set.
- When using 8 cards and 16 batch size for training, the total training time will be significantly reduced, but the accuracy may slightly decrease. Based on testing, the accuracy for both the n and x specifications has dropped by 0.3%.
- We refer to the official [YOLOV11](https://github.com/ultralytics/ultralytics) to reproduce the P5 series model.

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Jocher Glenn. Ultralytics YOLOv11. https://github.com/ultralytics/ultralytics, 2024.
