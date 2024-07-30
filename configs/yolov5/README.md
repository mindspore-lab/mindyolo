# YOLOv5

## Abstract
YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset, representing Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.
<div align=center>
<img src="https://raw.githubusercontent.com/zhanghuiyao/pics/main/mindyolo20230407113509.png"/>
</div>

## Results

<details open markdown>
<summary><b>performance tested on Ascend 910(8p) with graph mode</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---: |                :---:                         |        :---:       |
| YOLOv5 | N                  |  32 * 8   |    640    | MS COCO 2017 |    27.3     | 1.9M   | [yaml](./yolov5n.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-9b16bd7b.ckpt)                  |
| YOLOv5 | S                  |  32 * 8   |    640    | MS COCO 2017 |    37.6     | 7.2M   | [yaml](./yolov5s.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-860bcf3b.ckpt)                  |
| YOLOv5 | M                  |  32 * 8   |    640    | MS COCO 2017 |    44.9     | 21.2M  | [yaml](./yolov5m.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5m_300e_mAP449-e7bbf695.ckpt)                  |
| YOLOv5 | L                  |  32 * 8   |    640    | MS COCO 2017 |    48.5     | 46.5M  | [yaml](./yolov5l.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5l_300e_mAP485-a28bce73.ckpt)                  |
| YOLOv5 | X                  |  16 * 8   |    640    | MS COCO 2017 |    50.5     | 86.7M  | [yaml](./yolov5x.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5x_300e_mAP505-97d36ddc.ckpt)                  |
</details>

<details open markdown>
<summary><b>performance tested on Ascend 910*(8p)</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | ms/step | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---:  |  :---: |                :---:                         |        :---:       |
| YOLOv5 | N                  |  32 * 8   |    640    | MS COCO 2017 |     27.4    | 736.08  | 1.9M   | [yaml](./yolov5n.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-bedf9a93-910v2.ckpt)                  |
| YOLOv5 | S                  |  32 * 8   |    640    | MS COCO 2017 |     37.6    | 787.34  | 7.2M   | [yaml](./yolov5s.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-df4a45b6-910v2.ckpt)                  |
</details>

<br>

#### Notes

- Box mAP: Accuracy reported on the validation set.
- We refer to the official [YOLOV5](https://github.com/ultralytics/yolov5) to reproduce the P5 series model, and the differences are as follows:
  1. We use 8x NPU(Ascend910) for training, and the single-NPU batch size is 32. This is different from the official code.

## Quick Start

Please refer to the [GETTING_STARTED](https://github.com/mindspore-lab/mindyolo/blob/master/GETTING_STARTED.md) in MindYOLO for details.

### Training

<details open>

#### - Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run
```shell
# distributed training on multiple GPU/Ascend devices
msrun --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir="output_log" python train.py --config ./configs/yolov5/yolov5n.yaml --device_target Ascend --is_parallel True
```

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/utils/config.py).

#### Notes
- As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.
- The command above will run a 8-card training and save the log files into "output_log". `--master_port` specifies the scheduler binding port number.  `--worker_num` and `--local_worker_num` should be the same to the number of running devices, e.g., 8.
- In case of the following error:
```bash
RuntimtError: Failed to register the compute graph node: 0. Reason: Repeated registration node: 0
```
Please edit the `master_port` to a different port number in the range 1024 to 65535, and run the script again.

#### - Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config ./configs/yolov5/yolov5n.yaml --device_target Ascend
```

</details>

### Validation and Test

To validate the accuracy of the trained model, you can use `test.py` and parse the checkpoint path with `--weight`.

```
python test.py --config ./configs/yolov5/yolov5n.yaml --device_target Ascend --weight /PATH/TO/WEIGHT.ckpt
```

### Deployment

See [here](../../deploy/README.md).

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Jocher Glenn. YOLOv5 release v6.1. https://github.com/ultralytics/yolov5/releases/tag/v6.1, 2022.
