## MindYOLO 快速入门

本文简要介绍MindYOLO中内置的命令行工具的使用方法。

### 使用预训练模型进行推理

1. 从[model zoo](MODEL_ZOO.md)中选择一个模型及其配置文件，例如， `./configs/yolov7/yolov7.yaml`.
2. 从[model zoo](MODEL_ZOO.md)中下载相应的预训练模型权重文件。
3. 使用内置配置进行推理，请运行以下命令：

```shell
# NPU (默认)
python demo/predict.py --config ./configs/yolov7/yolov7.yaml --weight=/path_to_ckpt/WEIGHT.ckpt --image_path /path_to_image/IMAGE.jpg

# GPU
python demo/predict.py --config ./configs/yolov7/yolov7.yaml --weight=/path_to_ckpt/WEIGHT.ckpt --image_path /path_to_image/IMAGE.jpg --device_target=GPU
```

有关命令行参数的详细信息，请参阅`demo/predict.py -h`，或查看其[源代码](https://github.com/mindspore-lab/mindyolo/blob/master/deploy/predict.py)。

* 要在CPU上运行，请将device_target的值修改为CPU.
* 结果将保存在`./detect_results`目录下

### 使用命令行进行训练和评估

* 按照YOLO格式准备您的数据集。如果使用COCO数据集（YOLO格式）进行训练，请从[yolov5](https://github.com/ultralytics/yolov5)或darknet准备数据集.
  
  <details onclose>

  ```text
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

* 在多卡NPU/GPU上进行分布式模型训练，以8卡为例:

  ```shell
  mpirun --allow-run-as-root -n 8 python train.py --config ./configs/yolov7/yolov7.yaml  --is_parallel True
  ```

* 在单卡NPU/GPU/CPU上训练模型：

  ```shell
  python train.py --config ./configs/yolov7/yolov7.yaml 
  ```

* 评估模型的精度：

  ```shell
  python test.py --config ./configs/yolov7/yolov7.yaml --weight=/path_to_ckpt/WEIGHT.ckpt
  ```
*注意：默认超参为8卡训练，单卡情况需调整部分参数。 默认设备为Ascend，您可以指定'device_target'的值为Ascend/GPU/CPU。*
* 有关更多选项，请参阅 `train/test.py -h`.
* 在云脑上进行训练，请在[这里](./tutorials/modelarts_CN.md)查看

### 部署

请在[这里](./deploy/README.md)查看.


### 在代码中使用MindYOLO API


敬请期待



