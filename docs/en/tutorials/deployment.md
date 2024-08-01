# Deployment

## Dependencies
   ```shell
   pip install -r requirement.txt
   ```

## MindSpore Lite environment preparation
   Reference: [Lite environment configuration](https://mindspore.cn/lite) <br>
   Note: The python environment that MindSpore Lite is adapted to is 3.7. Please prepare the python3.7 environment before installing Lite <br>

   1. Depending on the environment, download the matching tar.gz package and whl package.

   2. Unzip the tar.gz package and install the corresponding version of the whl package
   ```shell
   tar -zxvf mindspore_lite-2.0.0a0-cp37-cp37m-{os}_{platform}_64.tar.gz
   pip install mindspore_lite-2.0.0a0-cp37-cp37m-{os}_{platform}_64.whl
   ```
   3. Configure Lite environment variables
   LITE_HOME is the folder path extracted from tar.gz. It is recommended to use the absolute path.
   ```shell
   export LITE_HOME=/path/to/mindspore-lite-{version}-{os}-{platform}
   export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
   export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
   ```

## Quick Start

### Model conversion
   Convert ckpt model to mindir model, this step can be run on CPU/Ascend910
   ```shell
   python ./deploy/export.py --config ./path_to_config/model.yaml --weight ./path_to_ckpt/weight.ckpt --per_batch_size 1 --file_format MINDIR --device_target [CPU/Ascend]
   e.g.
   #Run on CPU
   python ./deploy/export.py --config ./configs/yolov5/yolov5n.yaml --weight yolov5n_300e_mAP273-9b16bd7b.ckpt --per_batch_size 1 --file_format MINDIR --device_target CPU
   # Run on Ascend
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

## Script description
   - predict.py supports single image inference
   - test.py supports COCO data set inference
   - Note: currently only supports inference on Ascend 310

## MindX Deployment

### Environment configuration
Reference: [MindX environment preparation](https://www.hiascend.com/document/detail/zh/mind-sdk/300/quickstart/visionquickstart/visionquickstart_0003.html) <br>
Note: MindX currently supports python version 3.9. Please prepare the python3.9 environment before installing MindX <br>

1. Obtain the [Environment Installation Package] (https://www.hiascend.com/software/mindx-sdk/commercial) from the MindX official website. Currently, version 3.0.0 of MindX infer is supported.

2. Jump to the [Download page](https://support.huawei.com/enterprise/zh/ascend-computing/mindx-pid-252501207/software/255398987?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252501207) Download Ascend-mindxsdk-mxmanufacture_{version}_linux-{arch}.run

3. Place the installation package in the Ascend310 machine directory and unzip it

4. If you are not a root user, you need to add executable permissions to the package:
```shell
chmod +x Ascend-mindxsdk-mxmanufacture_{version}_linux-{arch}.run
```
5. Enter the upload path of the development kit package and install the mxManufacture development kit package.
```shell
./Ascend-mindxsdk-mxmanufacture_{version}_linux-{arch}.run --install
```
After the installation is completed, if the following echo appears, it means that the software was successfully installed.
```text
The installation is successful
```
After the installation is complete, the mxManufacture software directory structure is as follows:
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
6. Enter the installation directory of mxmanufacture and run the following command to make the MindX SDK environment variables take effect.
```shell
source set_env.sh
```
7. Enter ./mxVision-3.0.0/python/ and install mindx-3.0.0-py3-none-any.whl
```shell
pip install mindx-3.0.0-py3-none-any.whl
```

### Model conversion
   1. Convert ckpt model to air model. This step needs to be performed on Ascend910.
   ```shell
   python ./deploy/export.py --config ./path_to_config/model.yaml --weight ./path_to_ckpt/weight.ckpt --per_batch_size 1 --file_format AIR
   e.g.
   python ./deploy/export.py --config ./configs/yolov5/yolov5n.yaml --weight yolov5n_300e_mAP273-9b16bd7b.ckpt --per_batch_size 1 --file_format AIR
   ```
   *yolov7 needs to run export on an Ascend910 machine with version 2.0 or above*

   2. To convert the air model to the om model, use [atc conversion tool](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC1alpha002/infacldevg/atctool/atlasatc_16_0005.html). This step requires the installation of MindX Environment, running on Ascend310
   ```shell
   atc --model=./path_to_air/weight.air --framework=1 --output=yolo --soc_version=Ascend310
   ```

### MindX Test
   Infer COCO data:
   ```shell
   python ./deploy/test.py --model_type MindX --model_path ./path_to_om/weight.om --config ./path_to_config/yolo.yaml
   e.g.
   python ./deploy/test.py --model_type MindX --model_path ./yolov5n.om --config ./configs/yolov5/yolov5n.yaml
   ```
   
### MindX Predict
   Infer a single image:
   ```shell
   python ./deploy/predict.py --model_type MindX --model_path ./path_to_om/weight.om --config ./path_to_config/yolo.yaml --image_path ./path_to_image/image.jpg
   e.g.
   python ./deploy/predict.py --model_type MindX --model_path ./yolov5n.om --config ./configs/yolov5/yolov5n.yaml --image_path ./coco/image/val2017/image.jpg
   ```

## MindIR Deployment

### Environmental requirements
mindspore>=2.1

### Precautions
1. Currently only supports Predict

2. Theoretically, it can also run on Ascend910, but it has not been tested.


### Model conversion
   Convert the ckpt model to the mindir model, this step can be run on the CPU
   ```shell
   python ./deploy/export.py --config ./path_to_config/model.yaml --weight ./path_to_ckpt/weight.ckpt --per_batch_size 1 --file_format MINDIR --device_target CPU
   e.g.
   #Run on CPU
   python ./deploy/export.py --config ./configs/yolov5/yolov5n.yaml --weight yolov5n_300e_mAP273-9b16bd7b.ckpt --per_batch_size 1 --file_format MINDIR --device_target CPU
   ```

### MindIR Test
   Coming soon
   
### MindIR Predict
   Infer a single image:
   ```shell
   python ./deploy/predict.py --model_type MindIR --model_path ./path_to_mindir/weight.mindir --config ./path_to_conifg/yolo.yaml --image_path ./path_to_image/image.jpg
   e.g.
   python deploy/predict.py --model_type MindIR --model_path ./yolov5n.mindir --config ./configs/yolov5/yolov5n.yaml --image_path ./coco/image/val2017/image.jpg
   ```

## ONNX deployment

### Environment configuration
   ```shell
   pip install onnx>=1.9.0
   pip install onnxruntime>=1.8.0
   ```

### Precautions
1. Currently not all mindyolo supports ONNX export and inference (only YoloV3 is used as an example)

2. Currently only supports the Predict function

3. Exporting ONNX requires adjusting the nn.SiLU operator and using the underlying implementation of the sigmoid operator.

For example: add the following custom layer and replace all nn.SiLU in mindyolo
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

### Model conversion
   Convert the ckpt model to an ONNX model. This step and the Test step can only be run on the CPU.
   ```shell
   python ./deploy/export.py --config ./path_to_config/model.yaml --weight ./path_to_ckpt/weight.ckpt --per_batch_size 1 --file_format ONNX --device_target [CPU]
   e.g.
   #Run on CPU
   python ./deploy/export.py --config ./configs/yolov3/yolov3.yaml --weight yolov3-darknet53_300e_mAP455-adfb27af.ckpt --per_batch_size 1 --file_format ONNX --device_target CPU
   ```

### ONNX Test
   Coming soon
   
### ONNXRuntime Predict
   Infer a single image:
   ```shell
   python ./deploy/predict.py --model_type ONNX --model_path ./path_to_onnx_model/model.onnx --config ./path_to_config/yolo.yaml --image_path ./path_to_image/image.jpg
   e.g.
   python ./deploy/predict.py --model_type ONNX --model_path ./yolov3.onnx --config ./configs/yolov3/yolov3.yaml --image_path ./coco/image/val2017/image.jpg
   ```


## Standard and supported model libraries

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