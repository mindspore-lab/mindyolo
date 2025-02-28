---
hide:
  - toc
---

# YOLOv7

> [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/pdf/2207.02696.pdf)

## 摘要
YOLOv7在5FPS到 160 FPS 范围内的速度和准确度都超过了所有已知的物体检测器，YOLOv7 在 5 FPS 到 160 FPS 范围内的速度和准确度都超过了所有已知的目标检测器，并且在 GPU V100 上 30 FPS 或更高的所有已知实时目标检测器中具有最高的准确度 56.8% AP。YOLOv7-E6 目标检测器（56 FPS V100，55.9% AP）比基于transformer-based的检测器 SWINL Cascade-Mask R-CNN（9.2 FPS A100，53.9% AP）的速度和准确度分别高出 509% 和 2%，以及基于卷积的检测器 ConvNeXt-XL Cascade-Mask R-CNN (8.6 FPS A100, 55.2% AP) 速度提高 551%，准确率提高 0.7%，以及 YOLOv7 的表现优于：YOLOR、YOLOX、Scaled-YOLOv4、YOLOv5、DETR、Deformable DETR  , DINO-5scale-R50, ViT-Adapter-B 和许多其他物体探测器在速度和准确度上。 此外，我们只在 MS COCO 数据集上从头开始训练 YOLOv7，而不使用任何其他数据集或预训练的权重。

<div align=center>
<img src="https://raw.githubusercontent.com/zhanghuiyao/pics/main/mindyolo1680834261686.jpg"/>
</div>

## 结果

<details open markdown>
<summary><b>使用图模式在 Ascend 910(8p) 上测试的表现</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---: |                :---:                         |        :---:       |
| YOLOv7 | Tiny               |  16 * 8   |    640    | MS COCO 2017 |    37.5     | 6.2M   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7-tiny.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-d8972c94.ckpt)              |
| YOLOv7 | L                  |  16 * 8   |    640    | MS COCO 2017 |    50.8     | 36.9M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7_300e_mAP508-734ac919.ckpt)                   |
| YOLOv7 | X                  |  12 * 8   |    640    | MS COCO 2017 |    52.4     | 71.3M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7-x.yaml)       | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-x_300e_mAP524-e2f58741.ckpt)                 |
</details>

<details open markdown>
<summary><b>在Ascend 910*(8p)上测试的表现</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | ms/step | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---:  |  :---: |                :---:                         |        :---:       |
| YOLOv7 | Tiny               |  16 * 8   |    640    | MS COCO 2017 |     37.5    | 496.21  | 6.2M   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7-tiny.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-1d2ddf4b-910v2.ckpt)              |
</details>

<br>

#### 说明

- Context：训练上下文，表示为{设备}x{设备数}-{mindspore模式}，其中mindspore模式可以是G-图模式或F-pynative模式。例如，D910x8-G用于在8块Ascend 910 NPU上使用graph模式进行训练。
- Box mAP：验证集上测试出的准确度。
- 我们参考了常用的第三方 [YOLOV7](https://github.com/WongKinYiu/yolov7) 重现了P5（大目标）系列模型，并做出了如下改动：与官方代码有所不同，我们使用了8x NPU(Ascend910)进行训练，tiny/l/x单NPU的batch size分别为16/16/12。

## 快速入门

详情请参阅 MindYOLO 中的 [快速入门](../tutorials/quick_start.md)。

### 训练

#### - 分布式训练

使用预置的训练配方可以轻松重现报告的结果。如需在多台Ascend 910设备上进行分布式训练，请运行
```shell
# 在多台Ascend设备上进行分布式训练
msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov7_log python train.py --config ./configs/yolov7/yolov7.yaml --device_target Ascend --is_parallel True
```

**注意:** 更多关于msrun配置的信息，请参考[这里](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/msrun_launcher.html)。

有关所有超参数的详细说明，请参阅[config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py)。

**注意：** 由于全局batch size（batch_size x 设备数）是一个重要的超参数，建议保持全局batch size不变进行复制，或者将学习率线性调整为新的全局batch size。

#### - 单卡训练

如果您想在较小的数据集上训练或微调模型而不进行分布式训练，请运行：

```shell
# 在 CPU/Ascend 设备上进行单卡训练
python train.py --config ./configs/yolov7/yolov7.yaml --device_target Ascend
```


### 验证和测试

要验证训练模型的准确性，您可以使用 `test.py` 并使用 `--weight` 传入权重路径。

```
python test.py --config ./configs/yolov7/yolov7.yaml --device_target Ascend --weight /PATH/TO/WEIGHT.ckpt
```

### 部署

详见 [部署](../tutorials/deployment.md)。

## 引用

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Chien-Yao Wang, Alexey Bochkovskiy, and HongYuan Mark Liao. Yolov7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv preprint arXiv:2207.02696, 2022.
