---
hide:
  - toc
---

# YOLOv5

## 摘要
YOLOv5 是在 COCO 数据集上预训练的一系列对象检测架构和模型，代表了 Ultralytics 对未来视觉 AI 方法的开源研究，融合了数千小时的研究和开发中积累的经验教训和最佳实践。
<div align=center>
<img src="https://raw.githubusercontent.com/zhanghuiyao/pics/main/mindyolo20230407113509.png"/>
</div>

## 结果

<details open markdown>
<summary><b>使用图模式在 Ascend 910(8p) 上测试的表现</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---: |                :---:                         |        :---:       |
| YOLOv5 | N                  |  32 * 8   |    640    | MS COCO 2017 |    27.3     | 1.9M   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5n.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-9b16bd7b.ckpt)                  |
| YOLOv5 | S                  |  32 * 8   |    640    | MS COCO 2017 |    37.6     | 7.2M   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5s.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-860bcf3b.ckpt)                  |
| YOLOv5 | M                  |  32 * 8   |    640    | MS COCO 2017 |    44.9     | 21.2M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5m.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5m_300e_mAP449-e7bbf695.ckpt)                  |
| YOLOv5 | L                  |  32 * 8   |    640    | MS COCO 2017 |    48.5     | 46.5M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5l.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5l_300e_mAP485-a28bce73.ckpt)                  |
| YOLOv5 | X                  |  16 * 8   |    640    | MS COCO 2017 |    50.5     | 86.7M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5x.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5x_300e_mAP505-97d36ddc.ckpt)                  |
</details>

<details open markdown>
<summary><b>在Ascend 910*(8p)上测试的表现</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | ms/step | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---:  |  :---: |                :---:                         |        :---:       |
| YOLOv5 | N                  |  32 * 8   |    640    | MS COCO 2017 |     27.4    | 736.08  | 1.9M   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5n.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-bedf9a93-910v2.ckpt)                  |
| YOLOv5 | S                  |  32 * 8   |    640    | MS COCO 2017 |     37.6    | 787.34  | 7.2M   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5s.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-df4a45b6-910v2.ckpt)                  |
</details>

<br>

#### 说明

- Box mAP：验证集上测试出的准确度。
- 我们参考了常用的第三方 [YOLOV5](https://github.com/ultralytics/yolov5) 重现了P5（大目标）系列模型，并做出了如下改动：与官方代码有所不同，我们使用了8x NPU(Ascend910)进行训练，单NPU的batch size为32。

## 快速入门

详情请参阅 MindYOLO 中的 [快速入门](../tutorials/quick_start.md)。

### 训练

#### - 分布式训练

使用预置的训练配方可以轻松重现报告的结果。如需在多台Ascend 910设备上进行分布式训练，请运行
```shell
# 在多台Ascend设备上进行分布式训练
msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov5_log python train.py --config ./configs/yolov5/yolov5n.yaml --device_target Ascend --is_parallel True
```

**注意:** 更多关于msrun配置的信息，请参考[这里](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/msrun_launcher.html)。

有关所有超参数的详细说明，请参阅[config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py)。

**注意：** 由于全局batch size（batch_size x 设备数）是一个重要的超参数，建议保持全局batch size不变进行复制，或者将学习率线性调整为新的全局batch size。

#### - 单卡训练

如果您想在较小的数据集上训练或微调模型而不进行分布式训练，请运行：

```shell
# 在 CPU/Ascend 设备上进行单卡训练
python train.py --config ./configs/yolov5/yolov5n.yaml --device_target Ascend
```



### 验证和测试

要验证训练模型的准确性，您可以使用 `test.py` 并使用 `--weight` 传入权重路径。

```
python test.py --config ./configs/yolov5/yolov5n.yaml --device_target Ascend --weight /PATH/TO/WEIGHT.ckpt
```

### 部署

详见 [部署](../tutorials/deployment.md)。

## 引用

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Jocher Glenn. YOLOv5 release v6.1. https://github.com/ultralytics/yolov5/releases/tag/v6.1, 2022.
