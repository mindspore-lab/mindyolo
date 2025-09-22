# 配置

[MindYOLO]套件同时支持yaml文件参数和命令行参数解析，并将相对固定、与模型强相关、较为复杂或者含有嵌套结构的参数编写成yaml文件，需根据实际应用场景更改或者较为简单的参数则通过命令行传入。

下面以yolov3为例，解释如何配置相应的参数。

[MindYOLO]: https://github.com/mindspore-lab/mindyolo

## 参数继承关系

参数优先级由高到低如下，出现同名参数时，低优先级参数会被高优先级参数覆盖

- 用户命令行传入参数
- python执行py文件中parser的默认参数
- 命令行传入config参数对应的yaml文件参数
- 命令行传入config参数对应的yaml文件中__BASE__参数中包含的yaml文件参数，例如yolov3.yaml含有如下参数：
```yaml
__BASE__: [
  '../coco.yaml',
  './hyp.scratch.yaml',
]
```

## 基础参数

### 参数说明
   - device_target: 所用设备，Ascend/CPU
   - save_dir: 运行结果保存路径，默认为./runs
   - log_interval: 打印日志step间隔，默认为100
   - is_parallel: 是否分布式训练，默认为False
   - ms_mode: 使用静态图模式(0)或动态图模式(1)，默认为0。
   - config: yaml配置文件路径
   - per_batch_size: 每张卡batch size，默认为32
   - epochs: 训练epoch数，默认为300
   - ...

### parse参数设置
该部分参数通常由命令行传入，示例如下：

  ```shell
  msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov7_log python train.py --config ./configs/yolov7/yolov7.yaml  --is_parallel True --log_interval 50
  ```

## 数据集

### 参数说明
   - dataset_name: 数据集名称
   - train_set: 训练集所在路径
   - val_set: 验证集所在路径
   - test_set: 测试集所在路径
   - nc: 数据集类别数
   - names: 类别名称
   - ...

### yaml文件样例
该部分参数在[configs/coco.yaml](../configs/coco.yaml)中定义，通常需修改其中的数据集路径

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

## 数据增强

### 参数说明
   - num_parallel_workers: 读取数据的工作进程数
   - train_transformers: 训练过程数据增强
   - test_transformers: 验证过程数据增强
   - ...

### yaml文件样例
该部分参数在[configs/yolov3/hyp.scratch.yaml](../configs/yolov3/hyp.scratch.yaml)中定义，其中train_transformers和test_transformers均为由字典组成的列表，各字典包含数据增强操作名称、发生概率及该增强方法相关的参数

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

## 模型

### 参数说明

   - model_name: 模型名称
   - depth_multiple: 模型深度因子
   - width_multiple: 模型宽度因子
   - stride: 特征图下采样倍数
   - anchors: 预设锚框
   - backbone: 模型骨干网络
   - head: 模型检测头

### yaml文件样例
该部分参数在[configs/yolov3/yolov3.yaml](../configs/yolov3/yolov3.yaml)中定义，根据backbon和head参数进行网络构建，参数以嵌套列表的形式呈现，每行代表一层模块，包含4个参数，分别是 输入层编号(-1代表上一层)、模块重复次数、模块名称和模块相应参数。用户也可以不借助yaml文件而直接在py文件中定义和注册网络。
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

## 损失函数

### 参数说明
   - name: 损失函数名称
   - box: box损失权重
   - cls: class损失权重
   - cls_pw: class损失正样本权重
   - obj: object损失权重
   - obj_pw: object损失正样本权重
   - fl_gamma: focal loss gamma
   - anchor_t: anchor shape比例阈值
   - label_smoothing: 标签平滑值

### yaml文件样例
该部分参数在[configs/yolov3/hyp.scratch.yaml](../configs/yolov3/hyp.scratch.yaml)中定义

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

## 优化器

### 参数说明
   - optimizer: 优化器名称。
   - lr_init: 学习率初始值
   - warmup_epochs: warmup epoch数
   - warmup_momentum: warmup momentum初始值
   - warmup_bias_lr: warmup bias学习率初始值
   - min_warmup_step: 最小warmup step数
   - group_param: 参数分组策略
   - gp_weight_decay: 分组参数权重衰减系数
   - start_factor: 初始学习率因数
   - end_factor: 结束学习率因数
   - momentum：移动平均的动量
   - loss_scale：loss缩放系数
   - nesterov：是否使用Nesterov Accelerated Gradient (NAG)算法更新梯度。

### yaml文件样例
该部分参数在[configs/yolov3/hyp.scratch.yaml](../configs/yolov3/hyp.scratch.yaml)中定义，如下示例中经过warmup阶段后的初始学习率为lr_init * start_factor = 0.01 * 1.0 = 0.01, 最终学习率为lr_init * end_factor = 0.01 * 0.01 = 0.0001

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
