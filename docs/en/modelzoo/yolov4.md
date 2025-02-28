---
hide:
  - toc
---

# YOLOv4

> [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)

## Abstract
There are a huge number of features which are said to
improve Convolutional Neural Network (CNN) accuracy.
Practical testing of combinations of such features on large
datasets, and theoretical justification of the result, is required. Some features operate on certain models exclusively
and for certain problems exclusively, or only for small-scale
datasets; while some features, such as batch-normalization
and residual-connections, are applicable to the majority of
models, tasks, and datasets. We assume that such universal
features include Weighted-Residual-Connections (WRC),
Cross-Stage-Partial-connections (CSP), Cross mini-Batch
Normalization (CmBN), Self-adversarial-training (SAT)
and Mish-activation. We use new features: WRC, CSP,
CmBN, SAT, Mish activation, Mosaic data augmentation,
CmBN, DropBlock regularization, and CIoU loss, and combine some of them to achieve state-of-the-art results: 43.5%
AP (65.7% AP50) for the MS COCO dataset at a realtime speed of 65 FPS on Tesla V100.

<div align=center>
<img src="https://github.com/yuedongli1/images/raw/master/mindyolo20230509.png"/>
</div>

## Results

<details open markdown>
<summary><b>performance tested on Ascend 910(8p) with graph mode</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---: |                :---:                         |        :---:       |
| YOLOv4 | CSPDarknet53       |  16 * 8   |    608    | MS COCO 2017 |    45.4     | 27.6M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov4/yolov4.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_320e_map454-50172f93.ckpt)      |
| YOLOv4 | CSPDarknet53(silu) |  16 * 8   |    608    | MS COCO 2017 |    45.8     | 27.6M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov4/yolov4-silu.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_silu_320e_map458-bdfc3205.ckpt) |
</details>

<details open markdown>
<summary><b>performance tested on Ascend 910*(8p)</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | ms/step | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---:  |  :---: |                :---:                         |        :---:       |
| YOLOv4 | CSPDarknet53       |  16 * 8   |    608    | MS COCO 2017 |     46.1    | 337.25  | 27.6M  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov4/yolov4.yaml)         | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_320e_map454-64b8506f-910v2.ckpt)      |
</details>

<br>

#### Notes

- Box mAP: Accuracy reported on the validation set.

## Quick Start

Please refer to the [QUICK START](../tutorials/quick_start.md) in MindYOLO for details.

### Training


#### - Pretraining Model

You can get the pre-training model trained on ImageNet2012 from [here](https://download.mindspore.cn/model_zoo/r1.2/cspdarknet53_ascend_v120_imagenet2012_official_cv_bs64_top1acc7854_top5acc9428/cspdarknet53_ascend_v120_imagenet2012_official_cv_bs64_top1acc7854_top5acc9428.ckpt).

To convert it to a loadable ckpt file for mindyolo, please put it in the root directory then run it
```shell
python mindyolo/utils/convert_weight_cspdarknet53.py
```

#### - Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run
```shell
# distributed training on multiple Ascend devices
msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov4_log python train.py --config ./configs/yolov4/yolov4-silu.yaml --device_target Ascend --is_parallel True --epochs 320
```

**Note:** For more information about msrun configuration, please refer to [here](https://www.mindspore.cn/docs/en/r2.5.0/model_train/parallel/msrun_launcher.html).

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py).

#### Notes 
- As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.
- If the following warning occurs, setting the environment variable PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning' will fix it.
```shell
multiprocessing/semaphore_tracker.py: 144 UserWarning: semaphore_tracker: There appear to be 235 leaked semaphores to clean up at shutdown len(cache))
```

#### - Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/Ascend device
python train.py --config ./configs/yolov4/yolov4-silu.yaml --device_target Ascend --epochs 320
```


### Validation and Test

To validate the accuracy of the trained model, you can use `test.py` and parse the checkpoint path with `--weight`.

```
python test.py --config ./configs/yolov4/yolov4-silu.yaml --device_target Ascend --iou_thres 0.6 --weight /PATH/TO/WEIGHT.ckpt
```

### Deployment

See [here](../tutorials/deployment.md).

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Alexey Bochkovskiy, Chien-Yao Wang and Ali Farhadi. YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv preprint arXiv:2004.10934, 2020.
