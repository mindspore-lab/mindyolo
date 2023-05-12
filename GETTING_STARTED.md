## Getting Started with MindYOLO

This document provides a brief intro of the usage of builtin command-line tools in MindYOLO.

### Inference Demo with Pre-trained Models

1. Pick a model and its config file from
  [model zoo](MODEL_ZOO.md),
  for example, `./configs/yolov7/yolov7.yaml`.
2. Download the corresponding pretrain checkpoint from the link in the [model zoo](MODEL_ZOO.md) of each model.
3. We provide `demo/predict.py` that is able to demo builtin configs. Run it with:

```
python demo/predict.py --config ./configs/yolov7/yolov7.yaml --device_target=Ascend --weight=MODEL.WEIGHTS --image_path /PATH/TO/IMAGE.jpg
```

The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for inference.
This command will run the inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `demo/predict.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run on cpu, modify device_target to CPU.
* The results will be saved in `./detect_results`

### Training & Evaluation in Command Line

* Prepare your dataset in YOLO format. If train with COCO dataset, prepare it from yolov5 or darknet.
  
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

* To train a model:
  ```
  mpirun --allow-run-as-root -n 8 python train.py
  e.g.:
  mpirun --allow-run-as-root -n 8 python train.py --config ./configs/yolov7/yolov7.yaml --device_target Ascend --is_parallel True > log.txt 2>&1 &
  tail -f ./log.txt
  ```

* The configs are made for 8 NPU/GPU training.
To train on 1 NPU/GPU, you may need to change some parameters, e.g.:
  ```
  python train.py --config ./configs/yolov7/yolov7.yaml --device_target Ascend --per_batch_size 16
  ```

* To evaluate a model's performance, use
  ```
  python test.py --config ./configs/yolov7/yolov7.yaml --device_target Ascend --weight=MODEL.WEIGHTS
  ```

* For more options, see `train/test.py -h`.


### Use MindYOLO APIs in Your Code

To be supplemented.
