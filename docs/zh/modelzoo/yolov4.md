---
hide:
  - toc
---

# YOLOv4

> [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)


## 摘要

目前有很多可以提高CNN准确性的算法。这些算法的组合在庞大数据集上进行测试、对实验结果进行理论验证都是非常必要的。
有些算法只在特定的模型上有效果，并且只对特定的问题有效，或者只对小规模的数据集有效；
然而有些算法，比如batch-normalization和residual-connections，对大多数的模型、任务和数据集都适用。
我们认为这样通用的算法包括：Weighted-Residual-Connections（WRC), Cross-Stage-Partial-connections（CSP）,
Cross mini-Batch Normalization（CmBN）, Self-adversarial-training（SAT）以及Mish-activation。
我们使用了新的算法：WRC, CSP, CmBN, SAT, Mish activation, Mosaic data augmentation, CmBN, Dropblock regularization 和CIoU loss以及它们的组合，
获得了最优的效果：在MS COCO数据集上的AP值为43.5%(65.7% AP50)，在Tesla V100上的实时推理速度为65FPS。

<div align=center>
<img src="https://github.com/yuedongli1/images/raw/master/mindyolo20230509.png"/>
</div>

## 结果

<details open markdown>
<summary><b>使用图模式在 Ascend 910(8p) 上测试的表现</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---: |                :---:                         |        :---:       |
| YOLOv4 | CSPDarknet53       |  16 * 8   |    608    | MS COCO 2017 |    45.4     | 27.6M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov4/yolov4.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_320e_map454-50172f93.ckpt)      |
| YOLOv4 | CSPDarknet53(silu) |  16 * 8   |    608    | MS COCO 2017 |    45.8     | 27.6M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov4/yolov4-silu.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_silu_320e_map458-bdfc3205.ckpt) |
</details>

<details open markdown>
<summary><b>在Ascend 910*(8p)上测试的表现</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | ms/step | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---:  |  :---: |                :---:                         |        :---:       |
| YOLOv4 | CSPDarknet53       |  16 * 8   |    608    | MS COCO 2017 |     46.1    | 337.25  | 27.6M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov4/yolov4.yaml)         | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_320e_map454-64b8506f-910v2.ckpt)      |
</details>

<br>

#### 说明

- Box mAP: 验证集上测试出的准确度。

## 快速入门

详情请参阅 MindYOLO 中的 [快速入门](../tutorials/quick_start.md)。

### 训练

#### - 预训练模型

您可以从 [此处](https://download.mindspore.cn/model_zoo/r1.2/cspdarknet53_ascend_v120_imagenet2012_official_cv_bs64_top1acc7854_top5acc9428/cspdarknet53_ascend_v120_imagenet2012_official_cv_bs64_top1acc7854_top5acc9428.ckpt) 获取预训练模型。

要将其转换为 mindyolo 可加载的 ckpt 文件，请将其放在根目录中，然后运行以下语句：
```shell
python mindyolo/utils/convert_weight_cspdarknet53.py
```

#### - 分布式训练

使用预置的训练配方可以轻松重现报告的结果。如需在多台Ascend 910设备上进行分布式训练，请运行
```shell
# distributed training on multiple Ascend devices
msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov4_log python train.py --config ./configs/yolov4/yolov4-silu.yaml --device_target Ascend --is_parallel True --epochs 320
```

**注意:** 更多关于msrun配置的信息，请参考[这里](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/msrun_launcher.html)。

有关所有超参数的详细说明，请参阅[config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py)。

#### 说明
- 由于全局batch size（batch_size x 设备数）是一个重要的超参数，建议保持全局batch size不变进行复制，或者将学习率线性调整为新的全局batch size。
- 如果出现以下警告，可以通过设置环境变量 PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning' 来修复。
```shell
multiprocessing/semaphore_tracker.py: 144 UserWarning: semaphore_tracker: There appear to be 235 leaked semaphores to clean up at shutdown len(cache))
```

#### - 单卡训练

如果您想在较小的数据集上训练或微调模型而不进行分布式训练，请运行：

```shell
# 在 CPU/Ascend 设备上进行单卡训练
python train.py --config ./configs/yolov4/yolov4-silu.yaml --device_target Ascend --epochs 320
```


### 验证和测试

要验证训练模型的准确性，您可以使用 `test.py` 并使用 `--weight` 传入权重路径。

```
python test.py --config ./configs/yolov4/yolov4-silu.yaml --device_target Ascend --iou_thres 0.6 --weight /PATH/TO/WEIGHT.ckpt
```

### 部署

详见 [部署](../tutorials/deployment.md).

## 引用

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Alexey Bochkovskiy, Chien-Yao Wang and Ali Farhadi. YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv preprint arXiv:2004.10934, 2020.
