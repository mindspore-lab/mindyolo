---
hide:
  - toc
---

# YOLOv3

> [YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)

## 摘要
我们对YOLO进行了一系列更新！它包含一堆小设计，可以使系统的性能得到更新。我们也训练了一个新的、比较大的神经网络。虽然比上一版更大一些，但是精度也提高了。不用担心，它的速度依然很快。YOLOv3在320×320输入图像上运行时只需22ms，并能达到28.2mAP，其精度和SSD相当，但速度要快上3倍。使用之前0.5 IOU mAP的检测指标，YOLOv3的效果是相当不错。YOLOv3使用Titan X GPU，其耗时51ms检测精度达到57.9 AP50，与RetinaNet相比，其精度只有57.5 AP50，但却耗时198ms，相同性能的条件下YOLOv3速度比RetinaNet快3.8倍。

<div align=center>
<img src="https://raw.githubusercontent.com/zhanghuiyao/pics/main/mindyolo202304071143644.png"/>
</div>

## 结果

<details open markdown>
<summary><b>使用图模式在 Ascend 910(8p) 上测试的表现</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---: |                :---:                         |        :---:       |
| YOLOv3 | Darknet53          |  16 * 8   |    640    | MS COCO 2017 |    45.5     | 61.9M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov3/yolov3.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-adfb27af.ckpt)         |
</details>

<details open markdown>
<summary><b>在Ascend 910*(8p)上测试的表现</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | ms/step | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---:  |  :---: |                :---:                         |        :---:       |
| YOLOv3 | Darknet53          |  16 * 8   |    640    | MS COCO 2017 |     46.6    | 396.60  | 61.9M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov3/yolov3.yaml)         | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-81895f09-910v2.ckpt)         |
</details>

<br>

#### 说明

- Box mAP：验证集上测试出的准确度。
- 我们参考了常用的第三方 [YOLOv3](https://github.com/ultralytics/yolov3) 的实现。

## 快速入门

详情请参阅 MindYOLO 中的 [快速入门](../tutorials/quick_start.md)。

### 训练

#### - 预训练模型

您可以从 [此处](https://pjreddie.com/media/files/darknet53.conv.74) 获取预训练模型。

要将其转换为 mindyolo 可加载的 ckpt 文件，请将其放在根目录中，然后运行以下语句：
```shell
python mindyolo/utils/convert_weight_darknet53.py
```

#### - 分布式训练

使用预置的训练配方可以轻松重现报告的结果。如需在多台Ascend 910设备上进行分布式训练，请运行
```shell
# 在多台Ascend设备上进行分布式训练
msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov3_log python train.py --config ./configs/yolov3/yolov3.yaml --device_target Ascend --is_parallel True
```

**注意:** 更多关于msrun配置的信息，请参考[这里](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/msrun_launcher.html)。

有关所有超参数的详细说明，请参阅[config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py)。

**注意：** 由于全局batch size（batch_size x 设备数）是一个重要的超参数，建议保持全局batch size不变进行复制，或者将学习率线性调整为新的全局batch size。

#### - 单卡训练

如果您想在较小的数据集上训练或微调模型而不进行分布式训练，请运行：

```shell
# 在 CPU/Ascend 设备上进行单卡训练
python train.py --config ./configs/yolov3/yolov3.yaml --device_target Ascend
```

### 验证和测试

要验证训练模型的准确性，您可以使用 `test.py` 并使用 `--weight` 传入权重路径。

```
python test.py --config ./configs/yolov3/yolov3.yaml --device_target Ascend --weight /PATH/TO/WEIGHT.ckpt
```

### 部署

详见 [部署](../tutorials/deployment.md)。

## 引用

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Jocher Glenn. YOLOv3 release v9.1. https://github.com/ultralytics/yolov3/releases/tag/v9.1, 2021.
[2] Joseph Redmon and Ali Farhadi. YOLOv3: An incremental improvement. arXiv preprint arXiv:1804.02767, 2018.
