## Getting Started with MindYOLO

This document provides a brief introduction to the usage of built-in command-line tools in MindYOLO.

### Inference Demo with Pre-trained Models

1. Pick a model and its config file from the
  [model zoo](MODEL_ZOO.md),
  such as, `./configs/yolov7/yolov7.yaml`.
2. Download the corresponding pre-trained checkpoint from the [model zoo](MODEL_ZOO.md) of each model.
3. To run YOLO object detection with the built-in configs, please run::

```
# Run with Ascend (By default)
python demo/predict.py --config ./configs/yolov7/yolov7.yaml --weight=/path_to_ckpt/WEIGHT.ckpt --image_path /path_to_image/IMAGE.jpg

# Run with GPU
python demo/predict.py --config ./configs/yolov7/yolov7.yaml --weight=/path_to_ckpt/WEIGHT.ckpt --image_path /path_to_image/IMAGE.jpg --device_target=GPU
```


For details of the command line arguments, see `demo/predict.py -h` or look at its [source code](https://github.com/mindspore-lab/mindyolo/blob/master/deploy/predict.py)
to understand their behavior. Some common arguments are:
* To run on cpu, modify device_target to CPU.
* The results will be saved in `./detect_results`

### Training & Evaluation in Command Line

* Prepare your dataset in YOLO format. If trained with COCO (YOLO format), prepare it from [yolov5](https://github.com/ultralytics/yolov5) or the darknet.
  
  <details onclose>

  ```
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

* To train a model on 8 NPUs/GPUs:
  ```
  mpirun --allow-run-as-root -n 8 python train.py --config ./configs/yolov7/yolov7.yaml  --is_parallel True
  ```

* To train a model on 1 NPU/GPU/CPU:
  ```
  python train.py --config ./configs/yolov7/yolov7.yaml 
  ```

* To evaluate a model's performance:
  ```
  python test.py --config ./configs/yolov7/yolov7.yaml --weight=/path_to_ckpt/WEIGHT.ckpt
  ```
*Notes: (1) The default hyper-parameter is used for 8-card training, and some parameters need to be adjusted in the case of a single card. (2) The default device is Ascend, and you can modify it by specifying 'device_target' as Ascend/GPU/CPU, as these are currently supported.*
* For more options, see `train/test.py -h`.


### To use MindYOLO APIs in Your Code

To be supplemented.
