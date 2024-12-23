
# Fine-tuning

## Custom Dataset Finetune Process

This article takes the Safety Hat Wearing Detection Dataset (SHWD) as an example to introduce the main process of finetune on MindYOLO with a custom data set.

### Dataset Conversion

[SHWD Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset/tree/master) uses data labels in voc format, and its file directory is as follows:
````
             Root directory
                ├── Comments
                │ ├── 000000.xml
                │ └── 000002.xml
                ├── Image Collection
                │ └── Main
                │ ├── test.txt
                │ ├── train.txt
                │ ├── trainval.txt
                │ └── val.txt
                └── JPEG image
                        ├── 000000.jpg
                        └── 000002.jpg
````
The xml file under the Annotations folder contains annotation information for each picture. The main contents are as follows:
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
It contains multiple objects. The name in object is the category name, and xmin, ymin, xmax, and ymax are the coordinates of the upper left corner and lower right corner of the detection frame.

The data set format supported by MindYOLO is YOLO format. For details, please refer to [Data Preparation](../how_to_guides/data_preparation.md)

Since MindYOLO selects the image name as image_id during the verification phase, the image name can only be of numeric type, not of string type, and the image needs to be renamed. Conversion to SHWD data set format includes the following steps:
* Copy the image to the corresponding path and rename it
* Write the relative path of the image in the corresponding txt file in the root directory
* Parse the xml file and generate the corresponding txt annotation file under the corresponding path
* The verification set also needs to generate the final json file

For detailed implementation, please refer to [convert_shwd2yolo.py](https://github.com/mindspore-lab/mindyolo/blob/master/examples/finetune_SHWD/convert_shwd2yolo.py). The operation method is as follows:

  ```shell
  python examples/finetune_SHWD/convert_shwd2yolo.py --root_dir /path_to_shwd/SHWD
  ```
Running the above command will generate a SHWD data set in yolo format in the same directory without changing the original data set.

### Write yaml configuration file
The configuration file mainly contains the corresponding parameters related to the data set, data enhancement, loss, optimizer, and model structure. Since MindYOLO provides a yaml file inheritance mechanism, you can only write the parameters that need to be adjusted as yolov7-tiny_shwd.yaml and inherit the native ones provided by MindYOLO. yaml file, its content is as follows:
```
__BASE__: [
  '../../configs/yolov7/yolov7-tiny.yaml',
]

per_batch_size: 16 # Single card batchsize, total batchsize=per_batch_size * device_num
img_size: 640 # image sizes
weight: ./yolov7-tiny_pretrain.ckpt
strict_load: False # Whether to strictly load the internal parameters of ckpt. The default is True. If set to False, when the number of classifications is inconsistent, the weight of the last layer of classifiers will be discarded.
log_interval: 10 #Print the loss result every log_interval iterations

data:
  dataset_name: shwd
  train_set: ./SHWD/train.txt # Actual training data path
  val_set: ./SHWD/val.txt
  test_set: ./SHWD/val.txt
  nc: 2 # Number of categories
  # class names
  names: [ 'person', 'hat' ] # The name of each category

optimizer:
  lr_init: 0.001 # initial learning rate
```
* ```__BASE__``` is a list, indicating the path of the inherited yaml file. Multiple yaml files can be inherited.
* per_batch_size and img_size respectively represent the batch_size on a single card and the image size used for data processing images.
* weight is the file path of the pre-trained model mentioned above, and strict_load means discarding parameters with inconsistent shapes.
* log_interval represents the log printing interval
* All parameters under the data field are data set related parameters, where dataset_name is the name of the custom data set, train_set, val_set, and test_set are the txt file paths that save the training set, validation set, and test set image paths respectively, nc is the number of categories, and names is classification name
* lr_init under the optimizer field is the initial learning rate after warm_up, which is 10 times smaller than the default parameters.

For parameter inheritance relationship and parameter description, please refer to [Configuration](../tutorials/configuration.md).

### Download pre-trained model
You can choose the [Model Warehouse](../modelzoo/benchmark.md) provided by MindYOLO as the pre-training model for the custom data set. The pre-training model already has better accuracy performance on the COCO data set. Compared with training from scratch, loading a pre-trained model will generally have faster convergence speed and higher final accuracy, and will most likely avoid problems such as gradient disappearance and gradient explosion caused by improper initialization.

The number of categories in the custom data set is usually inconsistent with the COCO data set. The detection head structure of each model in MindYOLO is related to the number of categories in the data set. Directly importing the pre-trained model may fail due to inconsistent shape. You can configure it in the yaml configuration file. Set the strict_load parameter to False, MindYOLO will automatically discard parameters with inconsistent shapes and throw a warning that the module parameter is not imported.
### Model fine-tuning (Finetune)
During the process of model fine-tuning, you can first train according to the default configuration. If the effect is not good, you can consider adjusting the following parameters:
* The learning rate can be adjusted smaller to prevent loss from being difficult to converge.
* per_batch_size can be adjusted according to the actual video memory usage. Generally, the larger per_batch_size is, the more accurate the gradient calculation will be.
* Epochs can be adjusted according to whether the loss converges
* Anchor can be adjusted according to the actual object size

Since the SHWD training set only has about 6,000 images, the yolov7-tiny model was selected for training.
* Distributed model training on multi-card NPU, taking 8 cards as an example:

  ```shell
  msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./yolov7-tiny_log python train.py --config ./examples/finetune_SHWD/yolov7-tiny_shwd.yaml --is_parallel True
  ```

* Train the model on a single card NPU/CPU:

  ```shell
  python train.py --config ./examples/finetune_SHWD/yolov7-tiny_shwd.yaml
  ```
*Note: Directly using the default parameters of yolov7-tiny to train on the SHWD data set can achieve an accuracy of AP50 87.0. Changing the lr_init parameter from 0.01 to 0.001 can achieve an accuracy result of ap50 of 89.2. *

### Visual reasoning
Use /demo/predict.py to use the trained model for visual reasoning. The operation method is as follows:

```shell
python demo/predict.py --config ./examples/finetune_SHWD/yolov7-tiny_shwd.yaml --weight=/path_to_ckpt/WEIGHT.ckpt --image_path /path_to_image/IMAGE.jpg
```
The reasoning effect is as follows:
<div align=center>
<img width='600' src="https://github.com/yuedongli1/images/raw/master/00006630.jpg"/>
</div>
