

# 微调

## 自定义数据集finetune流程

本文以安全帽佩戴检测数据集(SHWD)为例，介绍自定义数据集在MindYOLO上进行finetune的主要流程。

### 数据集格式转换

[SHWD数据集](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset/tree/master)采用voc格式的数据标注，其文件目录如下所示：
```
             ROOT_DIR
                ├── Annotations
                │        ├── 000000.xml
                │        └── 000002.xml
                ├── ImageSets
                │       └── Main
                │             ├── test.txt
                │             ├── train.txt
                │             ├── trainval.txt
                │             └── val.txt
                └── JPEGImages
                        ├── 000000.jpg
                        └── 000002.jpg
```
Annotations文件夹下的xml文件为每张图片的标注信息，主要内容如下：
```
<annotation>
  <folder>JPEGImages</folder>
  <filename>000377.jpg</filename>
  <path>F:\baidu\VOC2028\JPEGImages\000377.jpg</path>
  <source>
    <database>Unknown</database>
  </source>
  <size>
    <width>750</width>
    <height>558</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>hat</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>142</xmin>
      <ymin>388</ymin>
      <xmax>177</xmax>
      <ymax>426</ymax>
    </bndbox>
  </object>
```
其中包含多个object, object中的name为类别名称，xmin, ymin, xmax, ymax则为检测框左上角和右下角的坐标。

MindYOLO支持的数据集格式为YOLO格式，详情可参考[数据准备](../how_to_guides/data_preparation.md)

由于MindYOLO在验证阶段选用图片名称作为image_id，因此图片名称只能为数值类型，而不能为字符串类型，还需要对图片进行改名。对SHWD数据集格式的转换包含如下步骤：
* 将图片复制到相应的路径下并改名
* 在根目录下相应的txt文件中写入该图片的相对路径
* 解析xml文件，在相应路径下生成对应的txt标注文件
* 验证集还需生成最终的json文件

详细实现可参考[convert_shwd2yolo.py](https://github.com/mindspore-lab/mindyolo/blob/master/examples/finetune_SHWD/convert_shwd2yolo.py)，运行方式如下：

  ```shell
  python examples/finetune_SHWD/convert_shwd2yolo.py --root_dir /path_to_shwd/SHWD
  ```
运行以上命令将在不改变原数据集的前提下，在同级目录生成yolo格式的SHWD数据集。

### 编写yaml配置文件
配置文件主要包含数据集、数据增强、loss、optimizer、模型结构涉及的相应参数，由于MindYOLO提供yaml文件继承机制，可只将需要调整的参数编写为yolov7-tiny_shwd.yaml，并继承MindYOLO提供的原生yaml文件即可，其内容如下：
```
__BASE__: [
  '../../configs/yolov7/yolov7-tiny.yaml',
]

per_batch_size: 16 # 单卡batchsize，总的batchsize=per_batch_size * device_num
img_size: 640 # image sizes
weight: ./yolov7-tiny_pretrain.ckpt
strict_load: False # 是否按严格加载ckpt内参数，默认True，若设成False，当分类数不一致，丢掉最后一层分类器的weight
log_interval: 10 # 每log_interval次迭代打印一次loss结果

data:
  dataset_name: shwd
  train_set: ./SHWD/train.txt # 实际训练数据路径
  val_set: ./SHWD/val.txt
  test_set: ./SHWD/val.txt
  nc: 2 # 分类数
  # class names
  names: [ 'person',  'hat' ] # 每一类的名字

optimizer:
  lr_init: 0.001  # initial learning rate
```
* ```__BASE__```为一个列表，表示继承的yaml文件所在路径，可以继承多个yaml文件
* per_batch_size和img_size分别表示单卡上的batch_size和数据处理图片采用的图片尺寸
* weight为上述提到的预训练模型的文件路径，strict_load表示丢弃shape不一致的参数
* log_interval表示日志打印间隔
* data字段下全部为数据集相关参数，其中dataset_name为自定义数据集名称，train_set、val_set、test_set分别为保存训练集、验证集、测试集图片路径的txt文件路径，nc为类别数量，names为类别名称
* optimizer字段下的lr_init为经过warm_up之后的初始化学习率，此处相比默认参数缩小了10倍

参数继承关系和参数说明可参考[configuration](../tutorials/configuration.md)。

### 下载预训练模型
可选用MindYOLO提供的[模型仓库](../modelzoo/benchmark.md)作为自定义数据集的预训练模型，预训练模型在COCO数据集上已经有较好的精度表现，相比从头训练，加载预训练模型一般会拥有更快的收敛速度以及更高的最终精度，并且大概率能避免初始化不当导致的梯度消失、梯度爆炸等问题。

自定义数据集类别数通常与COCO数据集不一致，MindYOLO中各模型的检测头head结构跟数据集类别数有关，直接将预训练模型导入可能会因为shape不一致而导入失败，可以在yaml配置文件中设置strict_load参数为False，MindYOLO将自动舍弃shape不一致的参数，并抛出该module参数并未导入的告警
### 模型微调(Finetune)
模型微调过程中，可首先按照默认配置进行训练，如效果不佳，可考虑调整以下参数：
* 学习率可调小一些，防止loss难以收敛
* per_batch_size可根据实际显存占用调整，通常per_batch_size越大，梯度计算越精确
* epochs可根据loss是否收敛进行调整
* anchor可根据实际物体大小进行调整

由于SHWD训练集只有约6000张图片，选用yolov7-tiny模型进行训练。
* 在多卡NPU上进行分布式模型训练，以8卡为例:

  ```shell
  msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov7-tiny_log python train.py --config ./examples/finetune_SHWD/yolov7-tiny_shwd.yaml --is_parallel True
  ```

* 在单卡NPU/CPU上训练模型：

  ```shell
  python train.py --config ./examples/finetune_SHWD/yolov7-tiny_shwd.yaml 
  ```
*注意：直接用yolov7-tiny默认参数在SHWD数据集上训练，可取得AP50 87.0的精度。将lr_init参数由0.01改为0.001，即可实现ap50为89.2的精度结果。*

### 可视化推理
使用/demo/predict.py即可用训练好的模型进行可视化推理，运行方式如下：

```shell
python demo/predict.py --config ./examples/finetune_SHWD/yolov7-tiny_shwd.yaml --weight=/path_to_ckpt/WEIGHT.ckpt --image_path /path_to_image/IMAGE.jpg
```
推理效果如下：
<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/00006630.jpg"/>
</div>