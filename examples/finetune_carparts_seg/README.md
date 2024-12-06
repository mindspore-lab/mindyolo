# 基于MindYOLO的汽车配件分割案例输出

## 数据集介绍
本次实验使用了Carparts数据集，需要进行相应的数据集格式转换。

[Carparts数据集](https://universe.roboflow.com/gianmarco-russo-vt9xr/car-seg-un1pm?ref=ultralytics)是专门为计算机视觉应用而设计的精选图像集合，特别关注与汽车零部件相关的细分任务。此数据集提供了一组多样化的多个角度捕获的视觉对象，为训练和验证提供了有价值的注释。

## 数据集格式转换
MindYOLO在train过程中支持yolo格式的数据集。在yolo格式中，标注以.txt文本文件的形式存储，每个图像一个文件。每行代表一个对象，格式为：class_id，x1，y1，x2，y2...，xn，yn，其中所有的值都是归一化到图像尺寸的。
图像和标签通常分别存储在images和labels目录下。还包括train.txt和val.txt用来保存每张图片的路径。

需要注意的是，由于MindYOLO在验证阶段选用图片名称作为image_id，因此图片名称只能为数值类型，而不能为字符串类型。所以需要对图片进行改名，此外，同样作为路径标注的每个txt文件也需要使用更改命名后的文件路径。文件名以及路径生产的脚本可以参考[rename.py](./rename.py)。

MindYOLO在后续的eval过程中需要用到coco格式中annotations的内容，所以需要在yolo数据集格式的基础上，根据labels来生成annotations里的json文件内容。可参考[create_anno.py](./create_anno.py)，生成验证集的instances_val2017.json内容。

最后得到的数据集格式为：
```
Carpart
├── test.txt
├── train.txt
├── valid.txt
├── annotations
│       ├── instances_val2017.json
├── test
│       ├── images
│       │     ├── 0001.jpg
│       │     ├── 0002.jpg
│       │     └── ...
│       └── labels
│             ├── 0001.txt
│             ├── 0002.txt
│             └── ...
└── train
│       ├── images
│       │     ├── 0001.jpg
│       │     ├── 0002.jpg
│       │     └── ...
│       └── labels
│             ├── 0001.txt
│             ├── 0002.txt
│             └── ...
└── valid
        ├── images
        │     ├── 0001.jpg
        │     ├── 0002.jpg
        │     └── ...
        └── labels
              ├── 0001.txt
              ├── 0002.txt
              └── ...
```

## 实验过程

### 下载模型预训练权重
选择模型yolov8x-seg在Carparts数据集上来完成实验并测试结果精度。预训练模型权重参数可以在[README](../../configs/yolov8/README.md)中下载得到。将下载好的模型预训练权重放置MindYOLO主项目文件夹下。

### 编写yaml配置文件

MindYOLO支持yaml文件继承机制，因此新编写的配置文件只需要继承MindYOLO提供的原生yaml文件。
用yolov8x-seg进行汽车配件分割的配置文件见[yolov8x-seg_carparts.yaml](./yolov8x-seg_carparts.yaml)。
其中对实验所使用的carparts数据集内容进行设定，strict_load设为False，初始学习率lr_init设置为0.001，其他用原生yaml中的默认配置。

### 训练过程

可以选择在终端用命令行进行训练：
* 在多卡NPU上进行分布式模型训练，以8卡为例:
  ```shell
  msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./finetune_yolov8x-seg_carparts_log python train.py --config ./examples/finetune_carparts_seg/yolov8x-seg_carparts.yaml --is_parallel True
  ```

* 在单卡NPU上训练模型：
  ```shell
  python train.py --config ./examples/finetune_carparts_seg/yolov8x-seg_carparts.yaml

### 微调汽车配件分割的最终精度：
保存训练得到的权重参数的ckpt文件，用来测试精度和推理。
* 在单卡NPU上评估模型的精度：

  ```shell
  python test.py --config ./examples/finetune_carparts_seg/yolov8x-seg_carparts.yaml --weight /path_to_ckpt/WEIGHT.ckpt --task segment
  ```

通过yolov8x-seg在Carparts数据集上150轮的训练，实现了汽车配件分割的效果。
这里使用单卡计算精度，其结果的整体的推理精度box mAP(IoU=0.50:0.95)达到0.586和mask mAP(IoU=0.50:0.95)达到0.545。

### 可视化推理
 使用内置配置进行推理，运行以下命令：
```shell
# NPU (默认)
python ./examples/finetune_carparts_seg/predict.py --config ./examples/finetune_carparts_seg/yolov8x-seg_carparts.yaml --weight=/path_to_ckpt/WEIGHT.ckpt --image_path /path_to_image/IMAGE.jpg --conf_thres 0.5
```

![demo](https://github.com/user-attachments/assets/5ad7f53d-5d86-4c46-b98f-d7ba29926aa7)

#### Notes

- conf_thres：置信度阈值。用于控制显示预测概率超过conf_thres的预测结果。
