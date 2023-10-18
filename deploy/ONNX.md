# ONNX部署指南

## 环境配置
   ```shell
   pip install onnx>=1.9.0
   pip install onnxruntime>=1.8.0
   ```

## 注意事项
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

## 模型转换
   ckpt模型转为ONNX模型，此步骤以及Test步骤均仅支持CPU上运行
   ```shell
   python ./deploy/export.py --config ./path_to_config/model.yaml --weight ./path_to_ckpt/weight.ckpt --per_batch_size 1 --file_format ONNX --device_target [CPU]
   e.g.
   # 在CPU上运行
   python ./deploy/export.py --config ./configs/yolov3/yolov3.yaml --weight yolov3-darknet53_300e_mAP455-adfb27af.ckpt --per_batch_size 1 --file_format ONNX --device_target CPU
   ```

## ONNX Test
   TODO
   
## ONNXRuntime Predict
   对单张图片推理：
   ```shell
   python ./deploy/predict.py --model_type ONNX --model_path ./path_to_onnx_model/model.onnx --config ./path_to_config/yolo.yaml --image_path ./path_to_image/image.jpg
   e.g.
   python ./deploy/predict.py --model_type ONNX --model_path ./yolov3.onnx --config ./configs/yolov3/yolov3.yaml --image_path ./coco/image/val2017/image.jpg
   ```