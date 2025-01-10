

# Configuration

[MindYOLO] supports parameter parsing from both yaml files and command lines. The parameters which are fixed, complex, highly related to model or with nested structure are placed in yaml files. While the simpler ones or parameters variants with actual cases could be passed in from the command line.


The following takes yolov3 as an example to explain how to configure the corresponding parameters.

[MindYOLO]: https://github.com/mindspore-lab/mindyolo

## Parameter Inheritance Relationship

As follows, the parameter priority is from high to low. When a parameter with the same name appears, the low-priority parameter will be overwritten by the high-priority parameter.

- Parameters inputted with user command lines
- Default parameters set in parser from .py files
- Parameters in yaml files specified by user command lines
- Parameters in yaml files set by `__BASE__` contained in yaml files specified by user command lines. Take yolov3 as an example, it contains:
```yaml
__BASE__: [
  '../coco.yaml',
  './hyp.scratch.yaml',
]
```

## Basic Parameters

### Parameter Description
   - device_target: device used, Ascend/CPU
   - save_dir: the path to save the running results, the default is ./runs
   - log_interval: step interval to print logs, the default is 100
   - is_parallel: whether to perform distributed training, the default is False
   - ms_mode: whether to use static graph mode (0) or dynamic graph mode (1), the default is 0.
   - config: yaml configuration file path
   - per_batch_size: batch size of each card, default is 32
   - epochs: number of training epochs, default is 300
   - ...


### Parse parameter settings
This part of the parameters is usually passed in from the command line. Examples are as follows:

  ```shell
  msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov7_log python train.py --config ./configs/yolov7/yolov7.yaml  --is_parallel True --log_interval 50
  ```

## Dataset

### Parameter Description
   - dataset_name: dataset name
   - train_set: the path where the training set is located
   - val_set: the path where the verification set is located
   - test_set: the path where the test set is located
   - nc: number of categories in the data set
   - names: category name
   -...


### Yaml file sample
This part of the parameters is defined in [configs/coco.yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/coco.yaml), and the data set path usually needs to be modified.

```yaml
data:
  dataset_name: coco

  train_set: ./coco/train2017.txt  # 118287 images
  val_set: ./coco/val2017.txt  # 5000 images
  test_set: ./coco/test-dev2017.txt  # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794

  nc: 80

  # class names
  names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush' ]
```

## Data Augmentation

### Parameter Description
   - num_parallel_workers: number of worker processes reading data
   - train_transformers: data enhancement during training process
   - test_transformers: verification process data enhancement
   -...

### Yaml file sample
This part of the parameters is defined in [configs/yolov3/hyp.scratch.yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov3/hyp.scratch.yaml), where train_transformers and test_transformers are lists composed of dictionaries, each dictionary contains data enhancement operations name, probability of occurrence and parameters related to the enhancement method

```yaml
data:
  num_parallel_workers: 8

  train_transforms:
    - { func_name: mosaic, prob: 1.0, mosaic9_prob: 0.0, translate: 0.1, scale: 0.9 }
    - { func_name: mixup, prob: 0.1, alpha: 8.0, beta: 8.0, needed_mosaic: True }
    - { func_name: hsv_augment, prob: 1.0, hgain: 0.015, sgain: 0.7, vgain: 0.4 }
    - { func_name: label_norm, xyxy2xywh_: True }
    - { func_name: albumentations }
    - { func_name: fliplr, prob: 0.5 }
    - { func_name: label_pad, padding_size: 160, padding_value: -1 }
    - { func_name: image_norm, scale: 255. }
    - { func_name: image_transpose, bgr2rgb: True, hwc2chw: True }
      
  test_transforms:
    - { func_name: letterbox, scaleup: False }
    - { func_name: label_norm, xyxy2xywh_: True }
    - { func_name: label_pad, padding_size: 160, padding_value: -1 }
    - { func_name: image_norm, scale: 255. }
    - { func_name: image_transpose, bgr2rgb: True, hwc2chw: True }
```
## Model

### Parameter Description

   - model_name: model name
   - depth_multiple: model depth factor
   - width_multiple: model width factor
   - stride: feature map downsampling multiple
   - anchors: default anchor box
   - backbone: model backbone network
   - head: model detection head

### Yaml file sample
This part of the parameters is defined in [configs/yolov3/yolov3.yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov3/yolov3.yaml). The network is constructed based on the backbone and head parameters. The parameters are presented in the form of a nested list, with each line representing a The layer module contains 4 parameters, namely the input layer number (-1 represents the previous layer), the number of module repetitions, the module name and the corresponding parameters of the module. Users can also define and register networks directly in py files without resorting to yaml files.
```yaml
network:
  model_name: yolov3

  depth_multiple: 1.0  # model depth multiple
  width_multiple: 1.0  # layer channel multiple
  stride: [8, 16, 32]
  anchors:
    - [10,13, 16,30, 33,23]  # P3/8
    - [30,61, 62,45, 59,119]  # P4/16
    - [116,90, 156,198, 373,326]  # P5/32

  # darknet53 backbone
  backbone:
    # [from, number, module, args]
    [[-1, 1, ConvNormAct, [32, 3, 1]],  # 0
     [-1, 1, ConvNormAct, [64, 3, 2]],  # 1-P1/2
     [-1, 1, Bottleneck, [64]],
     [-1, 1, ConvNormAct, [128, 3, 2]],  # 3-P2/4
     [-1, 2, Bottleneck, [128]],
     [-1, 1, ConvNormAct, [256, 3, 2]],  # 5-P3/8
     [-1, 8, Bottleneck, [256]],
     [-1, 1, ConvNormAct, [512, 3, 2]],  # 7-P4/16
     [-1, 8, Bottleneck, [512]],
     [-1, 1, ConvNormAct, [1024, 3, 2]],  # 9-P5/32
     [-1, 4, Bottleneck, [1024]],  # 10
    ]

  # YOLOv3 head
  head:
    [[-1, 1, Bottleneck, [1024, False]],
     [-1, 1, ConvNormAct, [512, 1, 1]],
     [-1, 1, ConvNormAct, [1024, 3, 1]],
     [-1, 1, ConvNormAct, [512, 1, 1]],
     [-1, 1, ConvNormAct, [1024, 3, 1]],  # 15 (P5/32-large)

     [-2, 1, ConvNormAct, [256, 1, 1]],
     [-1, 1, Upsample, [None, 2, 'nearest']],
     [[-1, 8], 1, Concat, [1]],  # cat backbone P4
     [-1, 1, Bottleneck, [512, False]],
     [-1, 1, Bottleneck, [512, False]],
     [-1, 1, ConvNormAct, [256, 1, 1]],
     [-1, 1, ConvNormAct, [512, 3, 1]],  # 22 (P4/16-medium)

     [-2, 1, ConvNormAct, [128, 1, 1]],
     [-1, 1, Upsample, [None, 2, 'nearest']],
     [[-1, 6], 1, Concat, [1]],  # cat backbone P3
     [-1, 1, Bottleneck, [256, False]],
     [-1, 2, Bottleneck, [256, False]],  # 27 (P3/8-small)

     [[27, 22, 15], 1, YOLOv3Head, [nc, anchors, stride]],   # Detect(P3, P4, P5)
    ]
```
## Loss function

### Parameter Description
   - name: loss function name
   - box: box loss weight
   - cls: class loss weight
   - cls_pw: class loss positive sample weight
   - obj: object loss weight
   - obj_pw: object loss positive sample weight
   - fl_gamma: focal loss gamma
   - anchor_t: anchor shape proportion threshold
   - label_smoothing: label smoothing value

### Yaml file sample
This part of the parameters is defined in [configs/yolov3/hyp.scratch.yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov3/hyp.scratch.yaml)

```yaml
loss:
  name: YOLOv7Loss
  box: 0.05  # box loss gain
  cls: 0.5  # cls loss gain
  cls_pw: 1.0  # cls BCELoss positive_weight
  obj: 1.0  # obj loss gain (scale with pixels)
  obj_pw: 1.0  # obj BCELoss positive_weight
  fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
  anchor_t: 4.0  # anchor-multiple threshold
  label_smoothing: 0.0 # label smoothing epsilon
```

## Optimizer

### Parameter Description
   - optimizer: optimizer name.
   - lr_init: initial value of learning rate
   - warmup_epochs: number of warmup epochs
   - warmup_momentum: initial value of warmup momentum
   - warmup_bias_lr: initial value of warmup bias learning rate
   - min_warmup_step: minimum number of warmup steps
   - group_param: parameter grouping strategy
   - gp_weight_decay: Group parameter weight decay coefficient
   - start_factor: initial learning rate factor
   - end_factor: end learning rate factor
   - momentum: momentum of the moving average
   - loss_scale: loss scaling coefficient
   - nesterov: Whether to use the Nesterov Accelerated Gradient (NAG) algorithm to update the gradient.

### Yaml file sample
This part of the parameters is defined in [configs/yolov3/hyp.scratch.yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov3/hyp.scratch.yaml). In the following example, the initial learning rate after the warmup stage is lr_init * start_factor = 0.01 * 1.0 = 0.01, the final learning rate is lr_init * end_factor = 0.01 * 0.01 = 0.0001

```yaml
optimizer:
  optimizer: momentum
  lr_init: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
  momentum: 0.937  # SGD momentum/Adam beta1
  nesterov: True # update gradients with NAG(Nesterov Accelerated Gradient) algorithm
  loss_scale: 1.0 # loss scale for optimizer
  warmup_epochs: 3  # warmup epochs (fractions ok)
  warmup_momentum: 0.8  # warmup initial momentum
  warmup_bias_lr: 0.1  # warmup initial bias lr
  min_warmup_step: 1000 # minimum warmup step
  group_param: yolov7 # group param strategy
  gp_weight_decay: 0.0005  # group param weight decay 5e-4
  start_factor: 1.0
  end_factor: 0.01
```
