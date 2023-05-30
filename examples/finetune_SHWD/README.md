### 自定义数据集finetune流程

本文以安全帽佩戴检测数据集(SHWD)为例，介绍自定义数据集在MindYOLO上进行finetune的主要流程。

#### 数据集格式转换

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
其中，ImageSets/Main文件下的txt文件中每行代表相应子集中单张图片不含后缀的文件名，例如：
```
000002
000005
000019
000022
000027
000034
```

由于MindYOLO在验证阶段选用图片名称作为image_id，因此图片名称只能为数值类型，而不能为字符串类型，还需要对图片进行改名。对SHWD数据集格式的转换包含如下步骤：
* 将图片复制到相应的路径下并改名
* 在根目录下相应的txt文件中写入该图片的相对路径
* 解析xml文件，在相应路径下生成对应的txt标注文件
* 验证集还需生成最终的json文件

详细实现可参考[convert_shwd2yolo.py](./convert_shwd2yolo.py)。运行方式如下：

  ```shell
  python examples/finetune_SHWD/convert_shwd2yolo.py --root_dir /path_to_shwd/SHWD
  ```

运行以上命令将在不改变原数据集的前提下，在同级目录生成yolo格式的SHWD数据集。

#### 预训练模型文件转换

由于SHWD数据集只有7000+张图片，选择yolov7-tiny进行该数据集的训练，可下载MindYOLO提供的在coco数据集上训练好的[模型文件](https://github.com/mindspore-lab/mindyolo/blob/master/MODEL_ZOO.md)作为预训练模型。由于coco数据集含有80种物体类别，SHWD数据集只有两类，模型的最后一层head层输出与类别数nc有关，因此需将预训练模型文件的最后一层去掉， 可参考[convert_yolov7-tiny_pretrain_ckpt.py](./convert_yolov7-tiny_pretrain_ckpt.py)。运行方式如下：

  ```shell
  python examples/finetune_SHWD/convert_yolov7-tiny_pretrain_ckpt.py
  ```

#### 模型微调(Finetune)

简要的训练流程可参考[finetune_shwd.py](./finetune_shwd.py)

* 在多卡NPU/GPU上进行分布式模型训练，以8卡为例:

  ```shell
  mpirun --allow-run-as-root -n 8 python examples/finetune_SHWD/finetune_shwd.py --config ./examples/finetune_SHWD/yolov7-tiny.yaml --is_parallel True
  ```

* 在单卡NPU/GPU/CPU上训练模型：

  ```shell
  python examples/finetune_SHWD/finetune_shwd.py --config ./examples/finetune_SHWD/yolov7-tiny_shwd.yaml 
  ```

*注意：直接用yolov7-tiny默认coco参数在SHWD数据集上训练，可取得AP50 87.0的精度。将lr_init参数由0.01改为0.001，即可实现ap50为89.2的精度结果。*