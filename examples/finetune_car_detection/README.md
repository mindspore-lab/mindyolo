# 基于MindYOLO的车辆检测案例输出

## 数据集介绍
本次实验使用了BDD100K和UA-DETRAC两个数据集，需要进行相应的格式转换和数据集整合。

[BDD100K数据集](http://bdd-data.berkeley.edu)是加州大学伯克利分校（UC Berkeley）的Berkeley Artificial Intelligence Research (BAIR) Lab发布的一个大规模、多样化的自动驾驶视频数据集。它有包含道路目标边界框的10万张图片，其中训练集7万，测试集2万，验证集1万。这个数据集旨在促进自动驾驶技术和计算机视觉领域的发展，特别是为了应对在不同环境和条件下进行驾驶的挑战。

[UA-DETRAC车辆检测数据集](http://detrac-db.rit.albany.edu/)是一个具有挑战性的现实世界多目标检测和多目标跟踪基准。该数据集包含使用佳能 EOS 550D 相机在中国北京和天津的 24 个不同地点拍摄的 10 小时视频。视频以每秒 25 帧 (fps) 的速度录制，分辨率为 960×540 像素。UA-DETRAC 数据集中有超过 14 万帧和 8250 辆手动标注的车辆，总共有 121 万个标记的对象边界框，其中训练集约82085张图片，测试集约56167张图片。该数据集可用于多目标检测和多目标跟踪算法开发。



## 数据集格式转换
MindYOLO在train过程中支持yolo格式的数据集。在yolo格式中，标注以.txt文本文件的形式存储，每个图像一个文件。每行代表一个对象，格式为：class_id，x_center，y_center，width，height，其中所有的值都是归一化到图像尺寸的。图像和标签通常分别存储在images和labels目录下。还包括train.txt和val.txt用来保存每张图片的路径，路径生成脚本可以参考[valtxt.py](./valtxt.py)。

下载得到的VOC格式的数据集的标签是以XML文件标注的，需要转换为yolo格式中的txt标注形式，可以参考脚本[voc2yolo.py](./voc2yolo.py)

需要注意的是，由于MindYOLO在验证阶段选用图片名称作为image_id，因此图片名称只能为数值类型，而不能为字符串类型，还需要对图片进行改名，同样作为标注的每个txt文件也需要改名。改文件名脚本可以参考[rename.py](./rename.py)。

MindYOLO在后续的eval过程中需要用到coco格式中annotations的内容，所以需要在yolo数据集格式的基础上，根据labels来生成annotations里的json文件内容。可参考[crejson.py](./crejson.py)，生成验证集的json内容。

最后得到的数据集格式为：
```
bdd_ud
├── train.txt
├── val.txt
├── annotations
│       ├── instances_train2017.json
│       └── instances_val2017.json
├── images
│       ├── train
│       │     ├── 0001.jpg
│       │     ├── 0002.jpg
│       │     └── ...
│       └── val
│             ├── 0001.jpg
│             ├── 0002.jpg
│             └── ...
└── labels
        ├── train
        │     ├── 0001.txt
        │     ├── 0002.txt
        │     └── ...
        └── val
              ├── 0001.txt
              ├── 0002.txt
              └── ...
```
## 数据集整合
整合两个不同的车辆检测数据集时，并不是简单的合并就可以的，需要考虑多个方面以确保合并后的数据集仍然保持高质量和一致性，这对于后续的模型训练和评估至关重要。

首先，由于UA-DETRAC数据集中的测试数据不含标注且没有单独给出验证集，我们只能用从训练数据中划分验证集，bdd100k的验证集包含10000张图片，所以这里划分的时候也从UA-DETRAC的训练集中划分一万左右的图片作为验证集，剩下七万多张图片用作训练集。

本次实验整合的类别是：'rider', 'pedestrian', 'trailer', 'train','bus','car','truck','traffic sign','traffic light','other person','motorcycle','bicycle','van'。其中12个类别属于bdd100k中的标注，然后增加了一个UA-DETRAC数据集中的类别"van"，两个数据集重合的类别是"bus"和"car"。在处理标签的时候，需要在丢弃UA-DETRAC数据集中原本带有的"others"类别，因为可能和bdd100k中的其他类别重合，影响模型的识别结果。

在整合数据集的过程中，进行充分的测试和验证是非常重要的，以确保合并的数据集不会引入新的偏见或错误。需要注意的是多次检查图片数据和标签数据的数量是否对的上，整合时需要统一改名为数字类型，如果数量对不上就会造成标签数据和图片也对不上，所以最好是改名之前就检查到位。

## 实验过程

选择模型时，尝试着先用较小的模型yolov7-tiny在普通的UA-DETRAC数据集上来完成实验并测试结果精度，之后再尝试更复杂的模型如yolov7l在整合后的数据集上来进行实验，识别更多的类别达到更好的效果。
预训练模型权重参数可以在[MODEL_ZOO.md](../../MODEL_ZOO.md)中下载得到。
## yolov7-tiny实验过程
### 编写yaml配置文件

MindYOLO支持yaml文件继承机制，因此新编写的配置文件只需要继承MindYOLO提供的原生yaml文件现有配置文件，
用yolov7-tiny进行车辆检测的配置文件见[yolov7-tiny_ud.yaml](./yolov7-tiny_ud.yaml)。
其中对实验所使用的UA-DETRAC数据集内容进行设定，strict_load设为False，初始学习率lr_init设置为0.001，其他用原生yaml中的默认配置。
### 训练过程
本次实验在启智云平台上进行，创建云脑任务中的训练任务，传入模型和数据集，操作步骤可以参考[openai_CN.md](../../tutorials/cloud/openi_CN.md)
。这里要注意的是云平台的模型存放路径较以前有变化，若是找不到路径可以利用云平台提供的C2Net库得到预训练模型路径，在train.py中修改weight和ckpt_url参数的值。

也可以选择在终端用命令行进行训练：
* 在多卡NPU/GPU上进行分布式模型训练，以8卡为例:
  ```shell
  mpirun --allow-run-as-root -n 8 python train.py --config ./yolov7-tiny_ud.yaml --is_parallel True
  ```

* 在单卡NPU/GPU/CPU上训练模型：
  ```shell
  python train.py --config ./yolov7-tiny_ud.yaml
  ```

### yolov7-tiny的最终精度：
保存训练得到的权重参数的ckpt文件，用来测试精度和推理。
* 在单卡NPU/GPU/CPU上评估模型的精度：

  ```shell
  python test.py --config ./yolov7-tiny_ud.yaml --weight /path_to_ckpt/WEIGHT.ckpt
  ```
* 在多卡NPU/GPU上进行分布式评估模型的精度：

  ```shell
  mpirun --allow-run-as-root -n 8 python test.py --config ./yolov7-tiny_ud.yaml --weight /path_to_ckpt/WEIGHT.ckpt --is_parallel True
  ```
  通过yolov7-tiny在UA-DETRAC数据集上300轮的训练，完整实现了车辆检测的效果。但由于该网络参数较少，能提取到的抽象特征不够全面和深层，对结果的精度存在一定的限制，
其结果的整体的推理精度AP(IoU=0.50:0.95)只达到0.266。所以后续尝试用更大的网络模型yolo7l在整合后的数据集上来训练，以达到更好的检测效果。

## 优化策略
* 数据集优化：训练时使用更完善的整合后的数据集，扩大识别的类别数量，提高检测效果。
* 模型选择：采用更深层、参数更庞大的网络，提取和表达特征的能力更强。
* 超参数优化：选择合适的初始学习率和学习率调度策略；根据NPU内存调整Batch Size，更大的批量可以提高训练稳定性。


## yolo7l在整合数据集上的实验过程
### 编写yaml配置文件
同样利用文件继承机制，编写yolov7在UA-DETRAC数据集上的配置文件，对比上一个实验主要修改了模型识别的类别和数据集地址的部分，详见[bdd_ud.yaml](./bdd_ud.yaml)
### 训练过程
训练方法与上文中yolov7-tiny相似，但由于这次用的网络参数更加庞大，所以整个训练时间会更长，可以尝试使用多卡训练。

### 可视化推理
 使用内置配置进行推理，运行以下命令：
```shell
# NPU (默认)
python demo/predict.py --config ./yolov7l_ud.yaml --weight=/path_to_ckpt/WEIGHT.ckpt --image_path /path_to_image/IMAGE.jpg

# GPU
python demo/predict.py --config ./yolov7l_ud.yaml --weight=/path_to_ckpt/WEIGHT.ckpt --image_path /path_to_image/IMAGE.jpg --device_target=GPU
```
<img src="./pic/predict.png" alt="predict" style="zoom:70%;" />

### yolov7l的最终精度
整合数据集后，模型可以识别更多个类别，但由于验证集的标签限制，验证精度时只对两个数据集的共有标签部分进行验证，发现改用yolov7l在整合数据上的时候可以达到更好的效果，模型的训练精度得到显著提升，达到了AP(IoU=0.50:0.95) = 0.551，更多结果数据如下图所示：
<img src="./pic/test.png" alt="test" style="zoom:70%;" />