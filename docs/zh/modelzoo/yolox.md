---
hide:
  - toc
---

# YOLOX

## 摘要
YOLOX 是一款新型高性能检测模型，在 YOLO 系列的基础上进行了一些经验丰富的改进。我们将 YOLO 检测器改为无锚方式，并采用其他先进的检测技术，例如解耦头和领先的标签分配策略 SimOTA，以在大规模模型中实现最佳效果：对于只有 0.91M 参数和 1.08G FLOPs 的 YOLO-Nano，我们在 COCO 上获得了 25.3% 的 AP，比 NanoDet 高出 1.8% AP；对于业界使用最广泛的检测器之一 YOLOv3，我们将其在 COCO 上的 AP 提升到 47.3%，比目前的最佳实践高出 3.0% AP；对于参数量与 YOLOv4-CSP 大致相同的 YOLOX-L，YOLOv5-L 在 Tesla V100 上以 68.9 FPS 的速度在 COCO 上实现了 50.0% 的 AP，比 YOLOv5-L 高出 1.8% 的 AP。此外，我们使用单个 YOLOX-L 模型在流式感知挑战赛（CVPR 2021 自动驾驶研讨会）上获得了第一名。
<div align=center>
<img src="https://raw.githubusercontent.com/zhanghuiyao/pics/main/mindyoloyolox_baseline.png"/>
</div>

## 结果

<details open markdown>
<summary><b>使用图模式在 Ascend 910(8p) 上测试的表现</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---: |                :---:                         |        :---:       |
| YOLOX  | N                  |   8 * 8   |    416    | MS COCO 2017 |    24.1     | 0.9M   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-nano.yaml)      | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-n_300e_map241-ec9815e3.ckpt)                  |
| YOLOX  | Tiny               |   8 * 8   |    416    | MS COCO 2017 |    33.3     | 5.1M   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-tiny.yaml)      | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-tiny_300e_map333-e5ae3a2e.ckpt)               |
| YOLOX  | S                  |   8 * 8   |    640    | MS COCO 2017 |    40.7     | 9.0M   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-s.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-s_300e_map407-0983e07f.ckpt)                  |
| YOLOX  | M                  |   8 * 8   |    640    | MS COCO 2017 |    46.7     | 25.3M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-m.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-m_300e_map467-1db321ee.ckpt)                  |
| YOLOX  | L                  |   8 * 8   |    640    | MS COCO 2017 |    49.2     | 54.2M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-l.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-l_300e_map492-52a4ab80.ckpt)                  |
| YOLOX  | X                  |   8 * 8   |    640    | MS COCO 2017 |    51.6     | 99.1M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-x.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-x_300e_map516-52216d90.ckpt)                  |
| YOLOX  | Darknet53          |   8 * 8   |    640    | MS COCO 2017 |    47.7     | 63.7M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-darknet53.yaml) | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-darknet53_300e_map477-b5fcaba9.ckpt)          |
</details>

<details open markdown>
<summary><b>在Ascend 910*(8p)上测试的表现</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | ms/step | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---:  |  :---: |                :---:                         |        :---:       |
| YOLOX  | S                  |   8 * 8   |    640    | MS COCO 2017 |     41.0    | 242.15  | 9.0M   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-s.yaml)         | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolox/yolox-s_300e_map407-cebd0183-910v2.ckpt)                   |
</details>

<br>

#### 说明

- Box mAP: 验证集上测试出的准确度。
- 我们参考了官方的 [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) 来重现结果.

## 快速入门

详情请参阅 MindYOLO 中的 [快速入门](../tutorials/quick_start.md)。

### 训练

#### - 分布式训练

使用预置的训练配方可以轻松重现报告的结果。如需在多台Ascend 910设备上进行分布式训练，请运行
```shell
# 在多台Ascend设备上进行分布式训练
msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolox_log python train.py --config ./configs/yolox/yolox-s.yaml --device_target Ascend --is_parallel True
```

**注意:** 更多关于msrun配置的信息，请参考[这里](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/msrun_launcher.html)。

有关所有超参数的详细说明，请参阅[config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py)。

**注意：**由于全局batch size（batch_size x 设备数）是一个重要的超参数，建议保持全局batch size不变进行复制，或者将学习率线性调整为新的全局batch size。

#### - 单卡训练

如果您想在较小的数据集上训练或微调模型而不进行分布式训练，请运行：

```shell
# 在 CPU/Ascend 设备上进行单卡训练
python train.py --config ./configs/yolox/yolox-s.yaml --device_target Ascend
```


### 验证和测试

要验证训练模型的准确性，您可以使用 `test.py` 并使用 `--weight` 传入权重路径。

```
python test.py --config ./configs/yolox/yolox-s.yaml --device_target Ascend --weight /PATH/TO/WEIGHT.ckpt
```

### 部署

详见 [部署](../tutorials/deployment.md)。

## 引用

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Zheng Ge. YOLOX: Exceeding YOLO Series in 2021. https://arxiv.org/abs/2107.08430, 2021.
