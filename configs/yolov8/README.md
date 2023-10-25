# YOLOv8

## Abstract
Ultralytics YOLOv8, developed by Ultralytics, is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection, image segmentation and image classification tasks.

<div align=center>
<img src="https://raw.githubusercontent.com/zhanghuiyao/pics/main/mindyolomindyolo-yolov8-comparison-plots.png"/>
</div>

## Results

### Detection

<div align="center">

| Name   | Scale | Arch | Context  | ImageSize | Dataset      | Box mAP (%) | Params | FLOPs  | Recipe                                                                                    | Download                                                                                              |
|--------|-------|------|----------|-----------|--------------|-------------|--------|--------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| YOLOv8 | N     | P5   | D910x8-G | 640       | MS COCO 2017 | 37.2        | 3.2M   | 8.7G   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/yolov8n.yaml) | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-n_500e_mAP372-cc07f5bd.ckpt)  |
| YOLOv8 | S     | P5   | D910x8-G | 640       | MS COCO 2017 | 44.6        | 11.2M  | 28.6G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/yolov8s.yaml) | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-s_500e_mAP446-3086f0c9.ckpt)  |
| YOLOv8 | M     | P5   | D910x8-G | 640       | MS COCO 2017 | 50.5        | 25.9M  | 78.9G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/yolov8m.yaml) | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-m_500e_mAP505-8ff7a728.ckpt)  |
| YOLOv8 | L     | P5   | D910x8-G | 640       | MS COCO 2017 | 52.8        | 43.7M  | 165.2G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/yolov8l.yaml) | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-l_500e_mAP528-6e96d6bb.ckpt)  |
| YOLOv8 | X     | P5   | D910x8-G | 640       | MS COCO 2017 | 53.7        | 68.2M  | 257.8G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/yolov8x.yaml) | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-x_500e_mAP537-b958e1c7.ckpt)  |

</div>

### Segmentation

<div align="center">

| Name       | Scale | Arch | Context  | ImageSize | Dataset      | Box mAP (%) | Mask mAP (%) | Params | FLOPs  | Recipe                                                                                            | Download                                                                                                       |
|------------|-------|------|----------|-----------|--------------|-------------|--------------|--------|--------|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| YOLOv8-seg | X     | P5   | D910x8-G | 640       | MS COCO 2017 | 52.5        | 42.9         | 71.8M  | 344.1G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/seg/yolov8x-seg.yaml) | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-x-seg_300e_mAP_mask_429-b4920557.ckpt) |

</div>

### Notes

- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Box mAP: Accuracy reported on the validation set.
- We refer to the official [YOLOV8](https://github.com/ultralytics/ultralytics) to reproduce the P5 series model.

## Quick Start

Please refer to the [GETTING_STARTED](https://github.com/mindspore-lab/mindyolo/blob/master/GETTING_STARTED.md) in MindYOLO for details.

### Training

<details open>

#### - Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run
```shell
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config ./configs/yolov8/yolov8n.yaml --device_target Ascend --is_parallel True
```
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above mpirun command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

#### - Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config ./configs/yolov8/yolov8n.yaml --device_target Ascend
```

</details>

### Validation and Test

To validate the accuracy of the trained model, you can use `test.py` and parse the checkpoint path with `--weight`.

```
python test.py --config ./configs/yolov8/yolov8n.yaml --device_target Ascend --weight /PATH/TO/WEIGHT.ckpt
```

### Deployment

See [here](../../deploy/README.md).

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Jocher Glenn. Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics, 2023.
