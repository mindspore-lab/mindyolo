##  MindYOLO推理

以下为yolo系列模型在ascend 310推理的步骤

### 1 安装依赖
   ```shell
   pip install -r requirement.txt
   ```

### 2 安装MindSpore Lite
   MindSpore Lite官方页面请查阅：[MindSpore Lite](https://mindspore.cn/lite) <br>
   - 下载tar.gz包并解压，同时配置环境变量LITE_HOME,LD_LIBRARY_PATH,PATH
   ```shell
   tar -zxvf mindspore_lite-[xxx].tar.gz
   export LITE_HOME=/[path_to_mindspore_lite_xxx]
   export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
   export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
   export Convert=$LITE_HOME/tools/converter/converter/converter_lite
   ```
   LITE_HOME为tar.gz解压出的文件夹路径，请设置绝对路径
   - 安装whl包
   ```shell
   pip install mindspore_lite-[xxx].whl
   ```
 - 验证过的MindSpore Lite版本为：2.2.14/2.3.0/2.3.1
 - 请安装相对应的ascend driver/firmware/ascend-toolkit
### 3 模型转换 ckpt -> mindir（可选）
   训练完成的模型ckpt权重转为mindir
   例如
   ```shell
   python ./deploy/export.py --config ./configs/yolov5/yolov5n.yaml --weight yolov5.ckpt --file_format MINDIR --device_target Ascend
   ```

### 4 单张图片推理

  - 以yolov5为例，工作目录为/work

   ```shell
   cd work
   git clone https://github.com/mindspore-lab/mindyolo.git
   cd mindyolo
   export PYTHONPATH="/work/mindyolo":$PYTHONPATH
   python ./deploy/mslite_predict.py --mindir_path yolov5n.mindir --config ./configs/yolov5/yolov5n.yaml --image_path test_img.jpg
   ```
  yolov5n.mindir 是已经从ckpt转好的mindir文件。可从mindir支持列表中下载

  - 如果想加快推理时加载模型的速度，可以把MindSpore mindir文件转换成MindSpore Lite mindir文件，直接使用lite mindir文件进行推理，例如：
  ```shell
  $Convert --fmk=MINDIR --modelFile=./yolov5n.mindir --outputFile=./yolov5n_lite  --saveType=MINDIR --optimize=ascend_oriented
  python ./deploy/mslite_predict.py --mindir_path yolov5n_lite.mindir --config ./configs/yolov5/yolov5n.yaml --image_path test_img.jpg
  ```
  modelFile为上面ckpt转好的mindir文件；outputFile为转换生成的MindSpore Lite mindir文件，默认会加扩展名mindir

## mindir支持列表

| model  | scale | img size | dataset | map| recipe | mindir|
|--------|:-----:|-----|--------|--------|--------|-------|
| YOLOv8 | N | 640 | MS COCO 2017 | 37.2 | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/yolov8n.yaml) | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-n_500e_mAP372-cc07f5bd-36a7ffec.mindir) |
| YOLOv8 | S  | 640  | MS COCO 2017 | 44.6 | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/yolov8s.yaml) | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-s_500e_mAP446-3086f0c9-137e9384.mindir) |
| YOLOv8 | M  | 640  | MS COCO 2017 | 50.5 | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/yolov8m.yaml) |[mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-m_500e_mAP505-8ff7a728-e21c252b.mindir) |
| YOLOv8 | L | 640 | MS COCO 2017 | 52.8  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/yolov8l.yaml) | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-l_500e_mAP528-6e96d6bb-55db59b4.mindir) |
| YOLOv8 | X  | 640 | MS COCO 2017 | 53.7 | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov8/yolov8x.yaml) | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov8/yolov8-x_500e_mAP537-b958e1c7-2a034e2c.mindir) |
| YOLOv7 | Tiny | 640 | MS COCO 2017 | 37.5  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7-tiny.yaml) |[mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-tiny_300e_mAP375-d8972c94-c550e241.mindir)      |
| YOLOv7 | L | 640  | MS COCO 2017 | 50.8    | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7.yaml)  | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7_300e_mAP508-734ac919-6d65d27c.mindir)           |
| YOLOv7 | X  | 640  | MS COCO 2017 | 52.4   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/yolov7-x.yaml) | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov7/yolov7-x_300e_mAP524-e2f58741-583e624b.mindir)        |
| YOLOv5 | N  | 640 | MS COCO 2017 | 27.3  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5n.yaml)     |[mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-9b16bd7b-bd03027b.mindir)         |
| YOLOv5 | S  | 640 | MS COCO 2017 | 37.6   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5s.yaml)    |[mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5s_300e_mAP376-860bcf3b-c105deb6.mindir)         |
| YOLOv5 | M   | 640 | MS COCO 2017 | 44.9  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5m.yaml)    | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5m_300e_mAP449-e7bbf695-b1525c76.mindir)         |
| YOLOv5 | L | 640 | MS COCO 2017 | 48.5   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5l.yaml)     | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5l_300e_mAP485-a28bce73-d4e437c2.mindir)         |
| YOLOv5 | X  | 640 | MS COCO 2017 | 50.5   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov5/yolov5x.yaml)     | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5x_300e_mAP505-97d36ddc-cae885cf.mindir)         |
| YOLOv4 | CSPDarknet53 | 608 | MS COCO 2017 | 45.4    | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov4/yolov4.yaml) | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_320e_map454-50172f93-cf2b8452.mindir) |
| YOLOv4 | CSPDarknet53(silu) | 640       | MS COCO 2017 | 45.8  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov4/yolov4-silu.yaml)      | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov4/yolov4-cspdarknet53_silu_320e_map458-bdfc3205-a0844d9f.mindir) |
| YOLOv3 | Darknet53 | 640       | MS COCO 2017 | 45.5   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov3/yolov3.yaml) | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolov3/yolov3-darknet53_300e_mAP455-adfb27af-335965fc.mindir)  |
| YOLOX  | N | 416       | MS COCO 2017 | 24.1   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-nano.yaml)  | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-n_300e_map241-ec9815e3-13b3ac7f.mindir)              |
| YOLOX  | Tiny| 416       | MS COCO 2017 | 33.3  | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-tiny.yaml) |  [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-tiny_300e_map333-e5ae3a2e-ff08fe48.mindir)              |
| YOLOX  | S   | 640  | MS COCO 2017 | 40.7   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-s.yaml)  | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-s_300e_map407-0983e07f-2f0f7762.mindir)                  |
| YOLOX  | M | 640  | MS COCO 2017 | 46.7   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-m.yaml)  | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-m_300e_map467-1db321ee-5a56d70e.mindir)                  |
| YOLOX  | L                  | 640       | MS COCO 2017 | 49.2 | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-l.yaml)         | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-l_300e_map492-52a4ab80-e1c4f344.mindir)    |
| YOLOX  | X                 | 640       | MS COCO 2017 | 51.6   | [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-x.yaml)         | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-x_300e_map516-52216d90-e5c397bc.mindir)   |
| YOLOX  | Darknet53           | 640       | MS COCO 2017 | 47.7| [yaml](https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolox/yolox-darknet53.yaml) | [mindir](https://download.mindspore.cn/toolkits/mindyolo/yolox/yolox-darknet53_300e_map477-b5fcaba9-d3380d02.mindir) 

<br>
