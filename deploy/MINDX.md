# MindX部署指南

## 环境配置
参考：[MindX环境准备](https://www.hiascend.com/document/detail/zh/mind-sdk/300/quickstart/visionquickstart/visionquickstart_0003.html) <br>
注意：MindX目前支持的python版本为3.9，请在安装MindX前，准备好python3.9的环境 <br>
1. 在MindX官网获取[环境安装包](https://www.hiascend.com/software/mindx-sdk/commercial)，目前支持3.0.0版本MindX推理
2. 跳转至[下载页面](https://support.huawei.com/enterprise/zh/ascend-computing/mindx-pid-252501207/software/255398987?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252501207)下载Ascend-mindxsdk-mxmanufacture_{version}_linux-{arch}.run
3. 将安装包放置于Ascend310机器目录中并解压
4. 如不是root用户，需增加对套件包的可执行权限：
```shell
chmod +x Ascend-mindxsdk-mxmanufacture_{version}_linux-{arch}.run
```
5. 进入开发套件包的上传路径，安装mxManufacture开发套件包。
```shell
./Ascend-mindxsdk-mxmanufacture_{version}_linux-{arch}.run --install
```
安装完成后，若出现如下回显，表示软件成功安装。
```text
The installation is successfully
```
安装完成后，mxManufacture软件目录结构如下所示：
```text
.
├── bin
├── config
├── filelist.txt
├── include
├── lib
├── opensource
├── operators
├── python
├── samples
├── set_env.sh
├── toolkit
└── version.info
```
6. 进入mxmanufacture的安装目录，运行以下命令，使MindX SDK环境变量生效。
```shell
source set_env.sh
```
7. 进入./mxVision-3.0.0/python/，安装mindx-3.0.0-py3-none-any.whl
```shell
pip install mindx-3.0.0-py3-none-any.whl
```

## 模型转换
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

## MindX Test
   对COCO数据推理：
   ```shell
   python ./deploy/test.py --model_type MindX --model_path ./path_to_om/weight.om --config ./path_to_config/yolo.yaml
   e.g.
   python ./deploy/test.py --model_type MindX --model_path ./yolov5n.om --config ./configs/yolov5/yolov5n.yaml
   ```
   
## MindX Predict
   对单张图片推理：
   ```shell
   python ./deploy/predict.py --model_type MindX --model_path ./path_to_om/weight.om --config ./path_to_config/yolo.yaml --image_path ./path_to_image/image.jpg
   e.g.
   python ./deploy/predict.py --model_type MindX --model_path ./yolov5n.om --config ./configs/yolov5/yolov5n.yaml --image_path ./coco/image/val2017/image.jpg
   ```