### 单类别数据集训练流程

本文以自制巧克力花生豆数据集为例，介绍单类别数据集使用MindYOLO进行训练的主要流程。

Requirements

| mindspore | ascend driver | firmware     | cann toolkit/kernel |
| :-------: | :-----------: | :----------: |:-------------------:|
|   2.5.0   |    24.1.0     | 7.5.0.3.220  |     8.0.0.beta1     |

#### 数据集格式转换

巧克力花生豆数据集采用voc格式的数据标注，其文件目录如下所示：
```
             ROOT_DIR
                ├── Annotations
                │        ├── 000000.xml
                │        └── 000002.xml
                ├── Images
                │       ├── 000000.jpg
                │       └── 000002.jpg
                └── Test_Images
                      ├── 000004.jpg
                      └── 000006.jpg
```
数据集格式转换分为以下步骤：

1. 训练集与验证集转换为yolo格式。可参考[voc2yolo.py](../finetune_car_detection/voc2yolo.py)，使用时需修改图片文件夹路径、标签文件夹路径与生成的txt标签文件夹路径，且对训练集和验证集依次完成该过程。
2. 验证集转换为coco格式。首先完成图片重命名，可参考[rename.py
](../finetune_car_detection/rename.py)，使用时需确保当前目录为数据集的根目录。然后生成json文件，可参考[crejson.py](../finetune_car_detection/crejson.py)，使用时需修改验证集图片文件夹路径，验证集txt标注文件路径以及生成的json文件路径。

#### 编写yaml配置文件
配置文件继承[yolov8n.yaml](../../configs/yolov8/yolov8n.yaml)，并且列出需要修改的参数，通常包括数据集相关参数以及学习率等超参，如下所示：
```
__BASE__: [
  '../../configs/yolov8/yolov8n.yaml',
]

data:
  dataset_name: seed
  train_set: ./seed/train.txt
  val_set: ./seed/val.txt
  nc: 1
  # class names
  names: [ 'seed' ]

optimizer:
  lr_init: 0.001  # initial learning rate
  warmup_bias_lr: 0.01 # warmup initial bias lr
  min_warmup_step: 10 # minmum warmup step
```
#### 模型训练
选用yolov8n模型进行训练。
* 在多卡NPU上进行分布式模型训练，以8卡为例:

  ```shell
  mpirun --allow-run-as-root -n 8 python train.py --config ./examples/finetune_single_class_dataset/yolov8n_single_class_dataset.yaml --is_parallel True
  ```

#### 可视化推理
使用/demo/predict.py即可用训练好的模型进行可视化推理，运行方式如下：

```shell
python demo/predict.py --config ./examples/finetune_single_class_dataset/yolov8n_single_class_dataset.yaml --weight=/path_to_ckpt/WEIGHT.ckpt --image_path /path_to_image/IMAGE.jpg
```
推理效果如下：
<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/16.jpg"/>
</div>