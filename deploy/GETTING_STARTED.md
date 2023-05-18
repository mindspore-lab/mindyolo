# MindYOLO推理

## 概览

MindYOLO支持MindX推理（mindspore1.9）和 Lite推理（mindspore2.0）

## mindx推理流程

### 模型转换
   1. ckpt模型转为air模型，此步骤需要在Ascend910上操作
   ```shell 
   python ./deploy/export.py --config ./path_to_config/model.yaml --weight ./path_to_ckpt/weight.ckpt --per_batch_size 1 --file_format AIR
   e.g.
   python ./deploy/export.py --config ./configs/yolov5/yolov5n.yaml --weight yolov5n_300e_mAP273-9b16bd7b.ckpt --per_batch_size 1 --file_format AIR
   ```
   * yolov7需要在2.0版本以上的Ascend910机器运行export

   2. air模型转为om模型，使用[atc转换工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC1alpha002/infacldevg/atctool/atlasatc_16_0005.html)，此步骤需安装MindX环境，在Ascend310上运行
   ```shell
   atc --model=./path_to_air/weight.air --framework=1 --output=yolo  --soc_version=Ascend310
   ```

### MindX Test
   对COCO数据推理：
   ```shell
   python ./deploy/test.py --model_type MindX --model_path ./path_to_om/weight.om --config ./path_to_config/yolo.yaml
   e.g.
   python ./deploy/test.py --model_type MindX --model_path ./yolov5n.om --config ./configs/yolov5/yolov5n.yaml
   ```
   
### MindX Predict
   对单张图片推理：
   ```shell
   python ./deploy/predict.py --model_type MindX --model_path ./path_to_om/weight.om --config ./path_to_config/yolo.yaml --image_path ./path_to_image/image.jpg
   e.g.
   python ./deploy/predict.py --model_type MindX --model_path ./yolov5n.om --config ./configs/yolov5/yolov5n.yaml --image_path ./coco/image/val2017/image.jpg
   ```

## lite推理流程

### 模型转换
   1. ckpt模型转为mindir模型，此步骤可在CPU/Ascend910上运行
   ```shell
   python ./deploy/export.py --config ./path_to_config/model.yaml --weight ./path_to_ckpt/weight.ckpt --per_batch_size 1 --file_format MINDIR --device_target [CPU/Ascend]
   e.g.
   # 在CPU上运行
   python ./deploy/export.py --config ./configs/yolov5/yolov5n.yaml --weight yolov5n_300e_mAP273-9b16bd7b.ckpt --per_batch_size 1 --file_format MINDIR --device_target CPU
   # 在Ascend上运行
   python ./deploy/export.py --config ./configs/yolov5/yolov5n.yaml --weight yolov5n_300e_mAP273-9b16bd7b.ckpt --per_batch_size 1 --file_format MINDIR --device_target Ascend
   ```

   2. 第一步得到的mindir模型需要在Ascend310机器上再次进行转换，得到Lite版本的mindir文件
   ```shell
   converter_lite --saveType=MINDIR --NoFusion=false --fmk=MINDIR  --device=Ascend --modelFile=./path_to_mindir/weight.mindir  --outputFile=yolo --configFile=config.txt
   ```
   *  conifg.txt文件需手动配置到./deploy文件夹下，bs,channel,height,weight对应export时的image的参数
   ```text
   [ascend_context]
   input_format=NCHW
   input_shape=x:[bs,channel,height,weight]
   ```
   * 注意：此步骤会同时产生om格式的文件，如果遇到tvm相关报错，请卸载mindspore-ascend或安装mindspore lite2.0后重启环境再次尝试

### Lite Test
   ```shell
   python deploy/test.py --model_type Lite --model_path ./path_to_mindir/weight.mindir --conifg ./path_to_config/yolo.yaml
   e.g.
   python deploy/test.py --model_type Lite --model_path ./yolov5n.mindir --conifg ./configs/yolov5/yolov5n.yaml
   ```

### Lite Predict
   ```shell
   python ./deploy/predict.py --model_type Lite --model_path ./path_to_mindir/weight.mindir --config ./path_to_conifg/yolo.yaml --image_path ./path_to_image/image.jpg
   e.g.
   python deploy/predict.py --model_type Lite --model_path ./yolov5n.mindir --conifg ./configs/yolov5/yolov5n.yaml --image_path ./coco/image/val2017/image.jpg
   ```