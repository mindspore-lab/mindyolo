# MindYOLO推理

## 环境配置

1. MindX
   参考：[MindX环境准备](https://www.hiascend.com/document/detail/zh/mind-sdk/300/quickstart/visionquickstart/visionquickstart_0003.html)
   注意：MindX目前支持的python版本为3.9，请在安装MindX前，准备好python3.9的环境
   1、在MindX官网获取[环境安装包](https://www.hiascend.com/software/mindx-sdk/commercial)
   2、跳转至[下载页面](https://support.huawei.com/enterprise/zh/ascend-computing/mindx-pid-252501207/software/255398987?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252501207)下载Ascend-mindxsdk-mxmanufacture_{version}_linux-{arch}.run
   3、将安装包放置于Ascend310机器目录中并解压
   4、如不是root用户，需增加对套件包的可执行权限：
   '''shell
   chmod +x Ascend-mindxsdk-mxmanufacture_{version}_linux-{arch}.run
   '''
   5、进入开发套件包的上传路径，安装mxManufacture开发套件包。
   '''shell
   ./Ascend-mindxsdk-mxmanufacture_{version}_linux-{arch}.run --install
   '''
   安装完成后，若出现如下回显，表示软件成功安装。
   '''text
   The installation is successfully
   '''
   安装完成后，mxManufacture软件目录结构如下所示：
   '''text
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
   '''
   6、进入mxmanufacture的安装目录，运行以下命令，使MindX SDK环境变量生效。
   '''shell
   source . set_env.sh
   '''

2. MindSpore Lite
   参考：[Lite环境配置]https://mindspore.cn/lite
   注意，MindSpore Lite适配的python环境为3.7，请在安装Lite前准备好python3.7的环境
   1、根据环境，下载配套的whl包和tar.gz包
   2、安装对应版本的whl包并解压tar.gz包
   '''shell
   pip install mindspore_lite-2.0.0a0-cp37-cp37m-{os}_{platform}_64.whl
   e.g.
   pip install mindspore_lite-2.0.0a0-cp37-cp37m-linux_x86_64.whl
   '''
   3、配置Lite的环境变量
   LITE_HOME为tar.gz解压的路径
   '''shell
   export LITE_HOME=/mindspore-lite-{version}-{os}-{platform}
   export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
   export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
   '''
   4、如果需要进行converter_lite操作，请添加config.txt
   '''text
   [ascend_context]
   input_format=NCHW
   input_shape=x:[bs,channel,height,weight]
   '''
   bs,channel,height,weight需要对应export的image的参数

## 标准和支持的模型库

- [x] [YOLOv7](configs/yolov7)
- [x] [YOLOv5](configs/yolov5)
- [x] [YOLOv3](configs/yolov3)
- [ ] [YOLOv8](configs/yolov6)
- [ ] [YOLOv6](configs/yolov6)
- [ ] [YOLOv4](configs/yolov6)
- [ ] [YOLOX](configs/yolox)

<div align="center">

| Name   | Scale | Arch     | Context  | ImageSize | Dataset      | Box mAP (%) | Params | FLOPs  | Recipe                                                                                        | Download                                                                                            |
|--------|-------|----------|----------|-----------|--------------|-------------|--------|--------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| YOLOv3 | Darknet53 | D910x8-G | 640       | MS COCO 2017 | 43.8        | 61.9M   | 156.4G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov3/yolov3.yaml)      | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP438-4cddcb38.ckpt)  |
| YOLOv5 | N     | P5       | D910x8-G | 640       | MS COCO 2017 | 27.3        | 1.9M   | 4.5G   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5n.yaml)     | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-9b16bd7b.ckpt)         |
| YOLOv5 | S     | P5       | D910x8-G | 640       | MS COCO 2017 | 37.6        | 7.2M   | 16.5G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5s.yaml)     | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-860bcf3b.ckpt)         |
| YOLOv5 | M     | P5       | D910x8-G | 640       | MS COCO 2017 | 44.9        | 21.2M  | 49.0G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5m.yaml)     | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5m_300e_mAP449-e7bbf695.ckpt)         |
| YOLOv5 | L     | P5       | D910x8-G | 640       | MS COCO 2017 | 48.5        | 46.5M  | 109.1G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5l.yaml)     | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5l_300e_mAP485-a28bce73.ckpt)         |
| YOLOv7 | Tiny  | P5   | D910x8-G | 640       | MS COCO 2017 | 37.5        | 6.2M   | 13.8G  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7-tiny.yaml) | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-d8972c94.ckpt) |
| YOLOv7 | L     | P5   | D910x8-G | 640       | MS COCO 2017 | 50.8        | 36.9M  | 104.7G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7.yaml)      | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7_300e_mAP508-734ac919.ckpt)      |
| YOLOv7 | X     | P5   | D910x8-G | 640       | MS COCO 2017 | 52.4        | 71.3M  | 189.9G | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7-x.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-x_300e_mAP524-e2f58741.ckpt)    |

</div>