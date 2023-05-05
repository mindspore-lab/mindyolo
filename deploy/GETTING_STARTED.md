# MindYOLO推理

## 概览

mindyolo支持mindx推理（mindspore 1.9）和 lite推理（mindspore2.0）

## mindx推理流程

1. 模型转换

   模型转换分为两步:
   1. ckpt模型转为air模型,使用mindspore.export 接口
      - 注意，此步骤暂不支持在Ascend310和CPU上运行（可以使用Ascend910）

   ```shell 
   python ./deploy/export.py --config configs.yaml --weight weight.ckpt --per_batch_size 1 --file_format AIR --device_target(optional) Ascend
   e.g.
   python ./deploy/export.py --config ./configs/yolov5/yolov5n.yaml --weight yolov5n_300e_mAP273-9b16bd7b.ckpt --per_batch_size 1 --file_format AIR --device_target Ascend
   ```

   2. air模型转为om模型，使用atc转换工具，此步骤需安装MindX环境，在310上运行。

   使用[ATC转换工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC1alpha002/infacldevg/atctool/atlasatc_16_0005.html)将air格式转换为om格式
   ```shell
   atc --model= ./weight.air --framework=1 --output=yolo  --soc_version=Ascend310
   ```

2. MindX Test
   ```shell
   python ./deploy/test.py --model_type MindX --model_path weight.om --config ./yolo.yaml
   e.g
   python ./deploy/test.py --model_type MindX --model_path yolov5n.om --config ./configs/yolov5/yolov5n.yaml
   ```

3. MindX Infer
   ```shell
   python ./deploy/infer.py --model_type MindX --model_path weight.om --config ./yolo.yaml --image_path image.jpg
   e.g
   python ./deploy/infer.py --model_type MindX --model_path yolov5n.om --config ./configs/yolov5/yolov5n.yaml --image_path ./coco/image/val2017/image.jpg
   ```

## lite推理流程

1. 模型转换

   ckpt模型转为mindir模型,使用mindspore.export
   - 注意，此步骤暂不支持在Ascend310上运行（可以使用Ascend910或CPU）

   ```shell
   python deploy/export.py --config configs.yaml --weight weight.ckpt --per_batch_size 1 --file_format MINDIR --device_target(optional) CPU
   ```

   mindir模型需要在Ascend310机器上再次进行转换，得到Lite版本的mindir文件
   - 此步骤会同时产生om格式的文件，如果遇到tvm相关报错，请卸载mindspore-ascend或安装mindspore lite2.0后重启环境再次尝试
   ```shell
   converter_lite --saveType=MINDIR --NoFusion=false --fmk=MINDIR  --device=Ascend --modelFile=weight.mindir  --outputFile=yolo --configFile=config.txt
   ```

2. Lite Test

    ```shell
   python deploy/test.py --model_type Lite --model_path weight.mindir --conifg conifg.yaml
   ```

3. Lite Infer
   ```shell
   python ./deploy/infer.py --model_type Lite --model_path weight.mindir --config ./yolo.yaml --image_path image.jpg
   ```