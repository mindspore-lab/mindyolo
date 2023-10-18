# MINDIR部署指南

## 环境要求
mindspore>=2.1

## 注意事项
1. 当前仅支持Predict
2. 理论上也可在Ascend910上运行，未测试


## 模型转换
   ckpt模型转为mindir模型，此步骤可在CPU上运行
   ```shell
   python ./deploy/export.py --config ./path_to_config/model.yaml --weight ./path_to_ckpt/weight.ckpt --per_batch_size 1 --file_format MINDIR --device_target CPU
   e.g.
   # 在CPU上运行
   python ./deploy/export.py --config ./configs/yolov5/yolov5n.yaml --weight yolov5n_300e_mAP273-9b16bd7b.ckpt --per_batch_size 1 --file_format MINDIR --device_target CPU
   ```

## MindIR Test
   TODO
   
## MindIR Predict
   对单张图片推理：
   ```shell
   python ./deploy/predict.py --model_type MindIR --model_path ./path_to_mindir/weight.mindir --config ./path_to_conifg/yolo.yaml --image_path ./path_to_image/image.jpg
   e.g.
   python deploy/predict.py --model_type MindIR --model_path ./yolov5n.mindir --config ./configs/yolov5/yolov5n.yaml --image_path ./coco/image/val2017/image.jpg
   ```