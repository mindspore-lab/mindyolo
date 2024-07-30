# YOLOv8

## Abstract
Ultralytics YOLOv8, developed by Ultralytics, is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection, image segmentation and image classification tasks.

<div align=center>
<img src="https://raw.githubusercontent.com/zhanghuiyao/pics/main/mindyolomindyolo-yolov8-comparison-plots.png"/>
</div>

## Results

### Detection

<details open markdown>
<summary><b>performance tested on Ascend 910(8p) with graph mode</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---: |                :---:                         |        :---:       |
| YOLOv8 | N                  |  16 * 8   |    640    | MS COCO 2017 |    37.2     | 3.2M   | [yaml](./yolov8n.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-n_500e_mAP372-cc07f5bd.ckpt)                 |
| YOLOv8 | S                  |  16 * 8   |    640    | MS COCO 2017 |    44.6     | 11.2M  | [yaml](./yolov8s.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-s_500e_mAP446-3086f0c9.ckpt)                 |
| YOLOv8 | M                  |  16 * 8   |    640    | MS COCO 2017 |    50.5     | 25.9M  | [yaml](./yolov8m.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-m_500e_mAP505-8ff7a728.ckpt)                 |
| YOLOv8 | L                  |  16 * 8   |    640    | MS COCO 2017 |    52.8     | 43.7M  | [yaml](./yolov8l.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-l_500e_mAP528-6e96d6bb.ckpt)                 |
| YOLOv8 | X                  |  16 * 8   |    640    | MS COCO 2017 |    53.7     | 68.2M  | [yaml](./yolov8x.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-x_500e_mAP537-b958e1c7.ckpt)                 |
</details>

<details open markdown>
<summary><b>performance tested on Ascend 910*(8p)</b></summary>

| Name   |        Scale       | BatchSize | ImageSize | Dataset      | Box mAP (%) | ms/step | Params |                Recipe                        | Download                                                                                                             |
|--------|        :---:       |   :---:   |   :---:   |--------------|    :---:    |  :---:  |  :---: |                :---:                         |        :---:       |
| YOLOv8 | N                  |  16 * 8   |    640    | MS COCO 2017 |     37.3    | 373.55  | 3.2M   | [yaml](./yolov8n.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov8/yolov8-n_500e_mAP372-0e737186-910v2.ckpt)                 |
| YOLOv8 | S                  |  16 * 8   |    640    | MS COCO 2017 |     44.7    | 365.53  | 11.2M  | [yaml](./yolov8s.yaml)        | [weights](https://download-mindspore.osinfra.cn/toolkits/mindyolo/yolov8/yolov8-s_500e_mAP446-fae4983f-910v2.ckpt)  |
</details>

### Segmentation

<details open markdown>
<summary><b>performance tested on Ascend 910(8p) with graph mode</b></summary>

| Name       | Scale | BatchSize | ImageSize | Dataset      | Box mAP (%) | Mask mAP (%) | Params |                Recipe                        | Download                                                                                                       |
|------------| :---: |   :---:   |   :---:   |--------------|    :---:    |     :---:    |  :---: |                :---:                         |        :---:       |
| YOLOv8-seg |   X   |  16 * 8   |    640    | MS COCO 2017 |     52.5    |     42.9     |  71.8M | [yaml](./seg/yolov8x-seg.yaml) | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-x-seg_300e_mAP_mask_429-b4920557.ckpt) |
</details>

### Notes

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
msrun --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir="output_log" python train.py --config ./configs/yolov8/yolov8n.yaml --device_target Ascend --is_parallel True
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
