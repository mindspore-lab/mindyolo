# MindYOLO推理

## 环境配置

### 依赖
   ```shell
   pip install -r requirement.txt
   ```

### MindX
   参考：[MindX环境准备](https://www.hiascend.com/document/detail/zh/mind-sdk/300/quickstart/visionquickstart/visionquickstart_0003.html)
   注意：MindX目前支持的python版本为3.9，请在安装MindX前，准备好python3.9的环境
   1、在MindX官网获取[环境安装包](https://www.hiascend.com/software/mindx-sdk/commercial)，目前支持3.0.0版本MindX推理
   2、跳转至[下载页面](https://support.huawei.com/enterprise/zh/ascend-computing/mindx-pid-252501207/software/255398987?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252501207)下载Ascend-mindxsdk-mxmanufacture_{version}_linux-{arch}.run
   3、将安装包放置于Ascend310机器目录中并解压
   4、如不是root用户，需增加对套件包的可执行权限：
   ```shell
   chmod +x Ascend-mindxsdk-mxmanufacture_{version}_linux-{arch}.run
   ```
   5、进入开发套件包的上传路径，安装mxManufacture开发套件包。
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
   6、进入mxmanufacture的安装目录，运行以下命令，使MindX SDK环境变量生效。
   ```shell
   source set_env.sh
   ```
   7、进入./mxVision-3.0.0/python/，安装mindx-3.0.0-py3-none-any.whl
   ```shell
   pip install mindx-3.0.0-py3-none-any.whl
   ```


### MindSpore Lite
   参考：[Lite环境配置](https://mindspore.cn/lite)
   注意：MindSpore Lite适配的python环境为3.7，请在安装Lite前准备好python3.7的环境
   1、根据环境，下载配套的whl包和tar.gz包
   2、安装对应版本的whl包并解压tar.gz包
   ```shell
   pip install mindspore_lite-2.0.0a0-cp37-cp37m-{os}_{platform}_64.whl
   e.g.
   pip install mindspore_lite-2.0.0a0-cp37-cp37m-linux_x86_64.whl
   ```
   3、配置Lite的环境变量
   LITE_HOME为tar.gz解压的路径
   ```shell
   export LITE_HOME=/mindspore-lite-{version}-{os}-{platform}
   export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
   export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
   ```
   4、如果需要进行converter_lite操作，请添加config.txt


## 脚本说明
   - predict.py 支持单张图片推理
   - test.py 支持COCO数据集推理

## 标准和支持的模型库

- [x] [YOLOv7](configs/yolov7)
- [x] [YOLOv5](configs/yolov5)
- [x] [YOLOv3](configs/yolov3)
- [x] [YOLOv8](configs/yolov6)
- [ ] [YOLOv6](configs/yolov6)
- [x] [YOLOv4](configs/yolov6)
- [ ] [YOLOX](configs/yolox)

| Name   | Scale              | Context  | ImageSize | Dataset      | Box mAP (%) | Params | FLOPs  | Recipe                                                                                        | Download                                                                                                     |
|--------|--------------------|----------|-----------|--------------|-------------|--------|--------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| YOLOv8 | N                  | D310x1-G | 640       | MS COCO 2017 | 37.2        | 3.2M   | 8.7G   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/yolov8n.yaml)     | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-n_500e_mAP372-cc07f5bd.ckpt) <br> [air](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-n_500e_mAP372-cc07f5bd-28a67e76.air) <br> [om](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-n_500e_mAP372-cc07f5bd-e499a1b9.om)         |
| YOLOv7 | Tiny               | D310x1-G | 640       | MS COCO 2017 | 37.5        | 6.2M   | 13.8G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7-tiny.yaml) | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-d8972c94.ckpt) <br> [air](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-d8972c94-3f2faab0.air) <br> [om](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-d8972c94-8059c534.om)      |
| YOLOv7 | L                  | D310x1-G | 640       | MS COCO 2017 | 50.8        | 36.9M  | 104.7G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7.yaml)      | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7_300e_mAP508-734ac919.ckpt) <br> [air](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7_300e_mAP508-734ac919-75482314.air) <br> [om](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7_300e_mAP508-734ac919-4c470f0e.om)           |
| YOLOv7 | X                  | D310x1-G | 640       | MS COCO 2017 | 52.4        | 71.3M  | 189.9G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7-x.yaml)    | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-x_300e_mAP524-e2f58741.ckpt) <br> [air](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-x_300e_mAP524-e2f58741-c5974271.air) <br> [om](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-x_300e_mAP524-e2f58741-25371b39.om)        |
| YOLOv5 | N                  | D310x1-G | 640       | MS COCO 2017 | 27.3        | 1.9M   | 4.5G   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5n.yaml)     | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-9b16bd7b.ckpt) <br> [air](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-9b16bd7b-a3d12352.air) <br> [om](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-9b16bd7b-ecd3bb22.om)         |
| YOLOv5 | S                  | D310x1-G | 640       | MS COCO 2017 | 37.6        | 7.2M   | 16.5G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5s.yaml)     | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-860bcf3b.ckpt) <br> [air](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-860bcf3b-40f373c0.air) <br> [om](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-860bcf3b-813f9415.om)         |
| YOLOv5 | M                  | D310x1-G | 640       | MS COCO 2017 | 44.9        | 21.2M  | 49.0G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5m.yaml)     | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5m_300e_mAP449-e7bbf695.ckpt) <br> [air](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5m_300e_mAP449-e7bbf695-23d53dc1.air) <br> [om](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5m_300e_mAP449-e7bbf695-e8d66d96.om)         |
| YOLOv5 | L                  | D310x1-G | 640       | MS COCO 2017 | 48.5        | 46.5M  | 109.1G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5l.yaml)     | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5l_300e_mAP485-a28bce73.ckpt) <br> [air](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5l_300e_mAP485-a28bce73-8ff0309e.air) <br> [om](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5l_300e_mAP485-a28bce73-8de36034.om)         |
| YOLOv5 | X                  | D310x1-G | 640       | MS COCO 2017 | 50.5        | 86.7M  | 205.7G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5x.yaml)     | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5x_300e_mAP505-97d36ddc.ckpt) <br> [air](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5x_300e_mAP505-97d36ddc-267a16f8.air) <br> [om](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5x_300e_mAP505-97d36ddc-c127d745.om)         |
| YOLOv4 | CSPDarknet53(silu) | D310x1-G | 640       | MS COCO 2017 | 45.8        | 27.6M  | 52G    | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov4/yolov4-silu.yaml)      | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_silu_320e_map458-bdfc3205.ckpt) <br> [air](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_silu_320e_map458-bdfc3205-51ae0b66.air) <br> [om](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_silu_320e_map458-bdfc3205-4f605795.om) |
| YOLOv3 | Darknet53          | D310x1-G | 640       | MS COCO 2017 | 45.5        | 61.9M  | 156.4G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov3/yolov3.yaml)      | [ckpt](https://download.mindspore.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-adfb27af.ckpt)<br> [air](https://download.mindspore.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-adfb27af-4183d716.air) <br> [om](https://download.mindspore.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-adfb27af-7a732437.om)  |

<br>