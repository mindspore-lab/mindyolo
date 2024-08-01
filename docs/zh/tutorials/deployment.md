# 部署

## 依赖
   ```shell
   pip install -r requirement.txt
   ```

## MindSpore Lite环境准备
   参考：[Lite环境配置](https://mindspore.cn/lite) <br>
   注意：MindSpore Lite适配的python环境为3.7，请在安装Lite前准备好python3.7的环境 <br>

   1. 根据环境，下载配套的tar.gz包和whl包

   2. 解压tar.gz包并安装对应版本的whl包
   ```shell
   tar -zxvf mindspore_lite-2.0.0a0-cp37-cp37m-{os}_{platform}_64.tar.gz
   pip install mindspore_lite-2.0.0a0-cp37-cp37m-{os}_{platform}_64.whl
   ```
   3. 配置Lite的环境变量
   LITE_HOME为tar.gz解压出的文件夹路径，推荐使用绝对路径
   ```shell
   export LITE_HOME=/path/to/mindspore-lite-{version}-{os}-{platform}
   export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
   export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
   ```

## 快速开始

### 模型转换
   ckpt模型转为mindir模型，此步骤可在CPU/Ascend910上运行
   ```shell
   python ./deploy/export.py --config ./path_to_config/model.yaml --weight ./path_to_ckpt/weight.ckpt --per_batch_size 1 --file_format MINDIR --device_target [CPU/Ascend]
   e.g.
   # 在CPU上运行
   python ./deploy/export.py --config ./configs/yolov5/yolov5n.yaml --weight yolov5n_300e_mAP273-9b16bd7b.ckpt --per_batch_size 1 --file_format MINDIR --device_target CPU
   # 在Ascend上运行
   python ./deploy/export.py --config ./configs/yolov5/yolov5n.yaml --weight yolov5n_300e_mAP273-9b16bd7b.ckpt --per_batch_size 1 --file_format MINDIR --device_target Ascend
   ```

### Lite Test
   ```shell
   python deploy/test.py --model_type Lite --model_path ./path_to_mindir/weight.mindir --config ./path_to_config/yolo.yaml
   e.g.
   python deploy/test.py --model_type Lite --model_path ./yolov5n.mindir --config ./configs/yolov5/yolov5n.yaml
   ```

### Lite Predict
   ```shell
   python ./deploy/predict.py --model_type Lite --model_path ./path_to_mindir/weight.mindir --config ./path_to_conifg/yolo.yaml --image_path ./path_to_image/image.jpg
   e.g.
   python deploy/predict.py --model_type Lite --model_path ./yolov5n.mindir --config ./configs/yolov5/yolov5n.yaml --image_path ./coco/image/val2017/image.jpg
   ```

## 脚本说明
   - predict.py 支持单张图片推理
   - test.py 支持COCO数据集推理
   - 注意：当前只支持在Ascend 310上推理

## MindX部署

### 环境配置
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

### 模型转换
   1. ckpt模型转为air模型，此步骤需要在Ascend910上操作
   ```shell 
   python ./deploy/export.py --config ./path_to_config/model.yaml --weight ./path_to_ckpt/weight.ckpt --per_batch_size 1 --file_format AIR
   e.g.
   python ./deploy/export.py --config ./configs/yolov5/yolov5n.yaml --weight yolov5n_300e_mAP273-9b16bd7b.ckpt --per_batch_size 1 --file_format AIR
   ```
   *yolov7需要在2.0版本以上的Ascend910机器运行export*

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

## MindIR部署

### 环境要求
mindspore>=2.1

### 注意事项
1. 当前仅支持Predict

2. 理论上也可在Ascend910上运行，未测试


### 模型转换
   ckpt模型转为mindir模型，此步骤可在CPU上运行
   ```shell
   python ./deploy/export.py --config ./path_to_config/model.yaml --weight ./path_to_ckpt/weight.ckpt --per_batch_size 1 --file_format MINDIR --device_target CPU
   e.g.
   # 在CPU上运行
   python ./deploy/export.py --config ./configs/yolov5/yolov5n.yaml --weight yolov5n_300e_mAP273-9b16bd7b.ckpt --per_batch_size 1 --file_format MINDIR --device_target CPU
   ```

### MindIR Test
   敬请期待
   
### MindIR Predict
   对单张图片推理：
   ```shell
   python ./deploy/predict.py --model_type MindIR --model_path ./path_to_mindir/weight.mindir --config ./path_to_conifg/yolo.yaml --image_path ./path_to_image/image.jpg
   e.g.
   python deploy/predict.py --model_type MindIR --model_path ./yolov5n.mindir --config ./configs/yolov5/yolov5n.yaml --image_path ./coco/image/val2017/image.jpg
   ```

## ONNX部署
**注意:** 仅部分模型支持导出ONNX并使用ONNXRuntime进行部署

### 环境配置
   ```shell
   pip install onnx>=1.9.0
   pip install onnxruntime>=1.8.0
   ```

### 注意事项
1. 当前并非所有mindyolo均支持ONNX导出和推理（仅以YoloV3为例）

2. 当前仅支持Predict功能

3. 导出ONNX需要调整nn.SiLU算子，采用sigmoid算子底层实现  

例如：添加如下自定义层并替换mindyolo中所有的nn.SiLU
```python
class EdgeSiLU(nn.Cell):
    """
    SiLU activation function: x * sigmoid(x). To support for onnx export with nn.SiLU.
    """

    def __init__(self):
        super().__init__()

    def construct(self, x):
        return x * ops.sigmoid(x)
```

### 模型转换
   ckpt模型转为ONNX模型，此步骤以及Test步骤均仅支持CPU上运行
   ```shell
   python ./deploy/export.py --config ./path_to_config/model.yaml --weight ./path_to_ckpt/weight.ckpt --per_batch_size 1 --file_format ONNX --device_target [CPU]
   e.g.
   # 在CPU上运行
   python ./deploy/export.py --config ./configs/yolov3/yolov3.yaml --weight yolov3-darknet53_300e_mAP455-adfb27af.ckpt --per_batch_size 1 --file_format ONNX --device_target CPU
   ```

### ONNX Test
   敬请期待
   
### ONNXRuntime Predict
   对单张图片推理：
   ```shell
   python ./deploy/predict.py --model_type ONNX --model_path ./path_to_onnx_model/model.onnx --config ./path_to_config/yolo.yaml --image_path ./path_to_image/image.jpg
   e.g.
   python ./deploy/predict.py --model_type ONNX --model_path ./yolov3.onnx --config ./configs/yolov3/yolov3.yaml --image_path ./coco/image/val2017/image.jpg
   ```


## 标准和支持的模型库

- [x] [YOLOv8](../modelzoo/yolov8.md)
- [x] [YOLOv7](../modelzoo/yolov7.md)
- [x] [YOLOX](../modelzoo/yolox.md)
- [x] [YOLOv5](../modelzoo/yolov5.md)
- [x] [YOLOv4](../modelzoo/yolov4.md)
- [x] [YOLOv3](../modelzoo/yolov3.md)

| Name   | Scale              | Context  | ImageSize | Dataset      | Box mAP (%) | Params | FLOPs  | Recipe                                                                                        | Download                                                                                                     |
|--------|--------------------|----------|-----------|--------------|-------------|--------|--------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| YOLOv8 | N                  | D310x1-G | 640       | MS COCO 2017 | 37.2        | 3.2M   | 8.7G   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/yolov8n.yaml)     | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-n_500e_mAP372-cc07f5bd.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-n_500e_mAP372-cc07f5bd-36a7ffec.mindir) |
| YOLOv8 | S            | D310x1-G | 640       | MS COCO 2017 | 44.6        | 11.2M  | 28.6G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/yolov8s.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-s_500e_mAP446-3086f0c9.ckpt)  <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-s_500e_mAP446-3086f0c9-137e9384.mindir) |
| YOLOv8 | M            | D310x1-G | 640       | MS COCO 2017 | 50.5        | 25.9M  | 78.9G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/yolov8m.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-m_500e_mAP505-8ff7a728.ckpt)  <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-m_500e_mAP505-8ff7a728-e21c252b.mindir) |
| YOLOv8 | L            | D310x1-G | 640       | MS COCO 2017 | 52.8        | 43.7M  | 165.2G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/yolov8l.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-l_500e_mAP528-6e96d6bb.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-l_500e_mAP528-6e96d6bb-55db59b4.mindir) |
| YOLOv8 | X            | D310x1-G | 640       | MS COCO 2017 | 53.7        | 68.2M  | 257.8G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/yolov8x.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-x_500e_mAP537-b958e1c7.ckpt)   <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-x_500e_mAP537-b958e1c7-2a034e2c.mindir)     |
| YOLOv7 | Tiny               | D310x1-G | 640       | MS COCO 2017 | 37.5        | 6.2M   | 13.8G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7-tiny.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-d8972c94.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-d8972c94-c550e241.mindir)      |
| YOLOv7 | L                  | D310x1-G | 640       | MS COCO 2017 | 50.8        | 36.9M  | 104.7G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7.yaml)      | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7_300e_mAP508-734ac919.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7_300e_mAP508-734ac919-6d65d27c.mindir)           |
| YOLOv7 | X                  | D310x1-G | 640       | MS COCO 2017 | 52.4        | 71.3M  | 189.9G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7-x.yaml)    | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-x_300e_mAP524-e2f58741.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-x_300e_mAP524-e2f58741-583e624b.mindir)        |
| YOLOv5 | N                  | D310x1-G | 640       | MS COCO 2017 | 27.3        | 1.9M   | 4.5G   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5n.yaml)     | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-9b16bd7b.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-9b16bd7b-bd03027b.mindir)         |
| YOLOv5 | S                  | D310x1-G | 640       | MS COCO 2017 | 37.6        | 7.2M   | 16.5G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5s.yaml)     | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-860bcf3b.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-860bcf3b-c105deb6.mindir)         |
| YOLOv5 | M                  | D310x1-G | 640       | MS COCO 2017 | 44.9        | 21.2M  | 49.0G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5m.yaml)     | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5m_300e_mAP449-e7bbf695.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5m_300e_mAP449-e7bbf695-b1525c76.mindir)         |
| YOLOv5 | L                  | D310x1-G | 640       | MS COCO 2017 | 48.5        | 46.5M  | 109.1G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5l.yaml)     | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5l_300e_mAP485-a28bce73.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5l_300e_mAP485-a28bce73-d4e437c2.mindir)         |
| YOLOv5 | X                  | D310x1-G | 640       | MS COCO 2017 | 50.5        | 86.7M  | 205.7G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5x.yaml)     | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5x_300e_mAP505-97d36ddc.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5x_300e_mAP505-97d36ddc-cae885cf.mindir)         |
| YOLOv4 | CSPDarknet53 | D310x1-G | 608             | MS COCO 2017 | 45.4        | 27.6M  | 52G    | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov4/yolov4.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_320e_map454-50172f93.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_320e_map454-50172f93-cf2b8452.mindir) |
| YOLOv4 | CSPDarknet53(silu) | D310x1-G | 640       | MS COCO 2017 | 45.8        | 27.6M  | 52G    | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov4/yolov4-silu.yaml)      | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_silu_320e_map458-bdfc3205.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_silu_320e_map458-bdfc3205-a0844d9f.mindir) |
| YOLOv3 | Darknet53          | D310x1-G | 640       | MS COCO 2017 | 45.5        | 61.9M  | 156.4G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov3/yolov3.yaml)      | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-adfb27af.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-adfb27af-335965fc.mindir)  |
| YOLOX  | N                  | D310x1-G | 416       | MS COCO 2017 | 24.1        | 0.9M   | 1.1G   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-nano.yaml)      | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-n_300e_map241-ec9815e3.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-n_300e_map241-ec9815e3-13b3ac7f.mindir)              |
| YOLOX  | Tiny               | D310x1-G | 416       | MS COCO 2017 | 33.3        | 5.1M   | 6.5G   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-tiny.yaml)      | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-tiny_300e_map333-e5ae3a2e.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-tiny_300e_map333-e5ae3a2e-ff08fe48.mindir)              |
| YOLOX  | S                  | D310x1-G | 640       | MS COCO 2017 | 40.7        | 9.0M   | 26.8G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-s.yaml)         | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-s_300e_map407-0983e07f.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-s_300e_map407-0983e07f-2f0f7762.mindir)                  |
| YOLOX  | M                  | D310x1-G | 640       | MS COCO 2017 | 46.7        | 25.3M  | 73.8G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-m.yaml)         | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-m_300e_map467-1db321ee.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-m_300e_map467-1db321ee-5a56d70e.mindir)                  |
| YOLOX  | L                  | D310x1-G | 640       | MS COCO 2017 | 49.2        | 54.2M  | 155.6G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-l.yaml)         | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-l_300e_map492-52a4ab80.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-l_300e_map492-52a4ab80-e1c4f344.mindir)                  |
| YOLOX  | X                  | D310x1-G | 640       | MS COCO 2017 | 51.6        | 99.1M  | 281.9G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-x.yaml)         | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-x_300e_map516-52216d90.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-x_300e_map516-52216d90-e5c397bc.mindir)                  |
| YOLOX  | Darknet53          | D310x1-G | 640       | MS COCO 2017 | 47.7        | 63.7M  | 185.3G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-darknet53.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-darknet53_300e_map477-b5fcaba9.ckpt) <br> [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-darknet53_300e_map477-b5fcaba9-d3380d02.mindir)                 |

<br>
