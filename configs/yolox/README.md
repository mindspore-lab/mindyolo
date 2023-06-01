# YOLOX

## Abstract
YOLOX is a new high-performance detector with some experienced improvements to YOLO series. We switch the YOLO detector to an anchor-free manner and conduct other advanced detection techniques, i.e., a decoupled head and the leading label assignment strategy SimOTA to achieve state-of-the-art results across a large scale range of models: For YOLO-Nano with only 0.91M parameters and 1.08G FLOPs, we get 25.3% AP on COCO, surpassing NanoDet by 1.8% AP; for YOLOv3, one of the most widely used detectors in industry, we boost it to 47.3% AP on COCO, outperforming the current best practice by 3.0% AP; for YOLOX-L with roughly the same amount of parameters as YOLOv4-CSP, YOLOv5-L, we achieve 50.0% AP on COCO at a speed of 68.9 FPS on Tesla V100, exceeding YOLOv5-L by 1.8% AP. Further, we won the 1st Place on Streaming Perception Challenge (Workshop on Autonomous Driving at CVPR 2021) using a single YOLOX-L model.
<div align=center>
<img src="https://raw.githubusercontent.com/zhanghuiyao/pics/main/mindyoloyolox_baseline.png"/>
</div>

## Results

<div align="center">

| Name   | Scale     | Context  | ImageSize | Dataset      | Box mAP (%) | Params | FLOPs  | Recipe                                                                                       | Download                                                                                                             |
|--------|-----------|----------|-----------|--------------|-------------|--------|--------|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| YOLOX  | N         | D910x8-G | 416       | MS COCO 2017 | 24.1        | 0.9M   | 1.1G   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-nano.yaml)  | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-n_300e_map241-ec9815e3.ckpt)                |
| YOLOX  | Tiny      | D910x8-G | 416       | MS COCO 2017 | 33.3        | 5.1M   | 6.5G   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-tiny.yaml)  | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-tiny_300e_map333-e5ae3a2e.ckpt)                |
| YOLOX  | S         | D910x8-G | 640       | MS COCO 2017 | 40.7        | 9.0M   | 26.8G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-s.yaml)     | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-s_300e_map407-0983e07f.ckpt)                   |
| YOLOX  | M         | D910x8-G | 640       | MS COCO 2017 | 46.7        | 25.3M  | 73.8G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-m.yaml)     | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-m_300e_map467-1db321ee.ckpt)                   |
| YOLOX  | L         | D910x8-G | 640       | MS COCO 2017 | 49.2        | 54.2M  | 155.6G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-l.yaml)     | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-l_300e_map492-52a4ab80.ckpt)                   |
| YOLOX  | X         | D910x8-G | 640       | MS COCO 2017 | 51.6        | 99.1M  | 281.9G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-x.yaml)     | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-x_300e_map516-52216d90.ckpt)                   |
| YOLOX  | Darknet53 | D910x8-G | 640       | MS COCO 2017 | 47.7        | 63.7M  | 185.3G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-darknet53.yaml) | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-darknet53_300e_map477-b5fcaba9.ckpt)                   |

</div>
<br>

#### Notes

- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Box mAP: Accuracy reported on the validation set.
- We refer to the official [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) to reproduce the results.

## Quick Start

Please refer to the [GETTING_STARTED](https://github.com/mindspore-lab/mindyolo/blob/master/GETTING_STARTED.md) in MindYOLO for details.

### Training

<details open>

#### - Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run
```shell
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config ./configs/yolox/yolox-s.yaml --device_target Ascend --is_parallel True
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.


Similarly, you can train the model on multiple GPU devices with the above mpirun command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction.

#### - Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please firstly run:

```shell
# standalone 1st stage training on a CPU/GPU/Ascend device
python train.py --config ./configs/yolox/yolox-s.yaml --device_target Ascend
```

</details>

### Validation and Test

To validate the accuracy of the trained model, you can use `test.py` and parse the checkpoint path with `--weight`.

```
python test.py --config ./configs/yolox/yolox-s.yaml --device_target Ascend --weight /PATH/TO/WEIGHT.ckpt
```

### Deployment

See [here](./deploy/README.md).

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Zheng Ge. YOLOX: Exceeding YOLO Series in 2021. https://arxiv.org/abs/2107.08430, 2021.
