

# Quick Start


## Getting Started with MindYOLO

This document provides a brief introduction to the usage of built-in command-line tools in MindYOLO.

### Inference Demo with Pre-trained Models

1. Pick a model and its config file from the [Model Zoo](../modelzoo/benchmark.md), such as, `./configs/yolov7/yolov7.yaml`.
2. Download the corresponding pre-trained checkpoint from the [Model Zoo](../modelzoo/benchmark.md) of each model.
3. To run YOLO object detection with the built-in configs, please run:

```shell
# Run with Ascend (By default)
python demo/predict.py --config ./configs/yolov7/yolov7.yaml --weight=/path_to_ckpt/WEIGHT.ckpt --image_path /path_to_image/IMAGE.jpg
```


For details of the command line arguments, see `demo/predict.py -h` or look at its [source code](https://github.com/mindspore-lab/mindyolo/blob/master/deploy/predict.py).
to understand their behavior. Some common arguments are:
* To run on cpu, modify device_target to CPU.
* The results will be saved in `./detect_results`

### Training & Evaluation in Command Line

* Prepare your dataset in YOLO format. If training with COCO (YOLO format), please prepare it from [yolov5](https://github.com/ultralytics/yolov5) or the darknet.
  
  <details onclose>

  ```text
    coco/
      {train,val}2017.txt
      annotations/
        instances_{train,val}2017.json
      images/
        {train,val}2017/
            00000001.jpg
            ...
            # image files that are mentioned in the corresponding train/val2017.txt
      labels/
        {train,val}2017/
            00000001.txt
            ...
            # label files that are mentioned in the corresponding train/val2017.txt
  ```
  </details>

* To train a model on 8 NPUs:
  ```shell
  msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov7_log python train.py --config ./configs/yolov7/yolov7.yaml  --is_parallel True
  ```

* To train a model on 1 NPU/CPU:
  ```shell
  python train.py --config ./configs/yolov7/yolov7.yaml 
  ```

* To evaluate a model's performance on 1 NPU/CPU:
  ```shell
  python test.py --config ./configs/yolov7/yolov7.yaml --weight /path_to_ckpt/WEIGHT.ckpt
  ```
* To evaluate a model's performance 8 NPUs:
  ```shell
  msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov7_log python test.py --config ./configs/yolov7/yolov7.yaml --weight /path_to_ckpt/WEIGHT.ckpt --is_parallel True
  ```
*Notes:*

*(1) The default hyper-parameter is used for 8-card training, and some parameters need to be adjusted in the case of a single card.*

*(2) The default device is Ascend, and you can modify it by specifying 'device_target' as Ascend/CPU, as these are currently supported.*

*(3) For more options, see `train/test.py -h`.*

*(4) To train on CloudBrain, see [here](modelarts.md)*


### Deployment

See [here](deployment.md).


### To use MindYOLO APIs in Your Code

Coming soon.

