# YOLOv7

> [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/pdf/2207.02696.pdf)

## Abstract
YOLOv7 surpasses all known object detectors in both speed and accuracy in the range from 5 FPS to 160 FPS and has the highest accuracy 56.8% AP among all known real-time object detectors with 30 FPS or higher on GPU V100. YOLOv7-E6 object detector (56 FPS V100, 55.9% AP) outperforms both transformer-based detector SWIN-L Cascade-Mask R-CNN (9.2 FPS A100, 53.9% AP) by 509% in speed and 2% in accuracy, and convolutional-based detector ConvNeXt-XL Cascade-Mask R-CNN (8.6 FPS A100, 55.2% AP) by 551% in speed and 0.7% AP in accuracy, as well as YOLOv7 outperforms: YOLOR, YOLOX, Scaled-YOLOv4, YOLOv5, DETR, Deformable DETR, DINO-5scale-R50, ViT-Adapter-B and many other object detectors in speed and accuracy. Moreover, we train YOLOv7 only on MS COCO dataset from scratch without using any other datasets or pre-trained weights.

<div align=center>
<img src="https://raw.githubusercontent.com/zhanghuiyao/pics/main/mindyolo1680834261686.jpg"/>
</div>

## Results

<details open markdown>
<summary><b>performance tested on Ascend 910(8p) with graph mode</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---: |                :---:                         |        :---:       |
| YOLOv7 | Tiny               |  16 * 8   |    640    | MS COCO 2017 |    37.5     | 6.2M   | [yaml](./yolov7-tiny.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-d8972c94.ckpt)              |
| YOLOv7 | L                  |  16 * 8   |    640    | MS COCO 2017 |    50.8     | 36.9M  | [yaml](./yolov7.yaml)         | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7_300e_mAP508-734ac919.ckpt)                   |
| YOLOv7 | X                  |  12 * 8   |    640    | MS COCO 2017 |    52.4     | 71.3M  | [yaml](./yolov7-x.yaml)       | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-x_300e_mAP524-e2f58741.ckpt)                 |
</details>

<details open markdown>
<summary><b>performance tested on Ascend 910*(8p)</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | ms/step | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---:  |  :---: |                :---:                         |        :---:       |
| YOLOv7 | Tiny               |  16 * 8   |    640    | MS COCO 2017 |     37.5    | 496.21  | 6.2M   | [yaml](./yolov7-tiny.yaml)    | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-1d2ddf4b-910v2.ckpt)              |
</details>

<br>

#### Notes

- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Box mAP: Accuracy reported on the validation set.
- We refer to the official [YOLOV7](https://github.com/WongKinYiu/yolov7) to reproduce the P5 series model, and the differences are as follows:
  1. We use 8x NPU(Ascend910) for training, and the single-NPU batch size for tiny/l/x is 16/16/12. This is different from the official code.

## Quick Start

Please refer to the [GETTING_STARTED](https://github.com/mindspore-lab/mindyolo/blob/master/GETTING_STARTED.md) in MindYOLO for details.

### Training

<details open>

#### - Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run
```shell
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config ./configs/yolov7/yolov7.yaml --device_target Ascend --is_parallel True
```
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above mpirun command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

#### - Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config ./configs/yolov7/yolov7.yaml --device_target Ascend
```

</details>

### Validation and Test

To validate the accuracy of the trained model, you can use `test.py` and parse the checkpoint path with `--weight`.

```
python test.py --config ./configs/yolov7/yolov7.yaml --device_target Ascend --weight /PATH/TO/WEIGHT.ckpt
```

### Deployment

See [here](../../deploy/README.md).

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Chien-Yao Wang, Alexey Bochkovskiy, and HongYuan Mark Liao. Yolov7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv preprint arXiv:2207.02696, 2022.
