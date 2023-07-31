# 数据增强

## 套件自带的数据增强方法清单

| 数据增强方法名            | 概要解释                | 
|--------------------|---------------------|
| mosaic             | 随机选择mosaic4和mosaic9 | 
| mosaic4            | 4分格拼接               | 
| mosaic9            | 9分格拼接               | 
| mixup              | 对两个图像进行线性混合         | 
| pastein            | 剪贴增强                | 
| random_perspective | 随机透视变换              | 
| hsv_augment        | 随机颜色变换              | 
| fliplr             | 水平翻转                | 
| flipud             | 垂直翻转                | 
| letterbox          | 缩放和填充               |
| label_norm         | 标签归一化 坐标归一化到0-1到范围  | 
| label_pad          | 将标签信息填充为固定大小的数组     | 
| image_norm         | 图像数据标准化             | 
| image_transpose    | 通道转置和维度转置           | 
| albumentations    | albumentations数据增强  | 

这些数据增强函数定义在 [mindyolo/data/dataset.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/data/dataset.py) 中。

## 使用方法
MindYOLO数据增强方法通过在yaml文件里配置。例如，训练过程添加一个数据增强，需要在yaml文件data.train_transforms字段下添加一个字典列表，数据增强方法自上而下依次罗列。

一个典型的数据增强方法配置字典里必须有func_name，表示应用的数据增强方法名，而后罗列该方法需要设置的参数，若没有在数据增强配置字典中配置参数项，则会选择该数据增强方法默认的数值。

数据增强通用配置字典：
```yaml
- {func_name: 数据增强方法名1, args11=x11, args12=x12, ..., args1n=x1n}
- {func_name: 数据增强方法名2, args21=x21, args22=x22, ..., args2n=x2n}
...
- {func_name: 数据增强方法名n, argsn1=xn1, argsn2=xn2, ..., argsnn=xnn}
```

以YOLOv7训练数据增强示例：
```yaml
# 文件目录：configs/yolov7/hyp.scratch.tiny.yaml (https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/hyp.scratch.tiny.yaml)
  train_transforms:
    - {func_name: mosaic, prob: 1.0, mosaic9_prob: 0.2, translate: 0.1, scale: 0.5}
    - {func_name: mixup, prob: 0.05, alpha: 8.0, beta: 8.0, needed_mosaic: True}
    - {func_name: hsv_augment, prob: 1.0, hgain: 0.015, sgain: 0.7, vgain: 0.4}
    - {func_name: pastein, prob: 0.05, num_sample: 30}
    - {func_name: label_norm, xyxy2xywh_: True}
    - {func_name: fliplr, prob: 0.5}
    - {func_name: label_pad, padding_size: 160, padding_value: -1}
    - {func_name: image_norm, scale: 255.}
    - {func_name: image_transpose, bgr2rgb: True, hwc2chw: True}
```
_注意：func_name表示数据增强方法名，prob，mosaic9_prob，translate，scale为该方法参数。 其中prob为所有方法均有的参数，表示该数据增强方法的执行概率，默认值为1_

上述yaml文件执行的具体操作如下：

- `mosaic`：以1.0的概率对输入的图片进行mosaic操作，即将4张不同的图片拼接成一张图片。mosaic9_prob表示使用9宫格方式进行拼接的概率，translate和scale分别表示随机平移和缩放的程度。
如图所示：
<img width='600' src="https://github.com/chenyang23333/images/raw/main/mosaic.png"/>

- `mixup`：以0.05的概率对输入的图片进行mixup操作，即将两张不同的图片进行混合。其中alpha和beta表示混合系数，needed_mosaic表示是否需要使用mosaic进行混合。

- `hsv_augment`: HSV增强, 以1.0的概率对输入的图片进行HSV颜色空间的调整，增加数据多样性。其中hgain、sgain和vgain分别表示对H、S、V通道的调整程度。

- `pastein `：以0.05的概率在输入的图片中随机贴入一些样本。其中num_sample表示随机贴入的样本数量。

- `label_norm`：将输入的标签从(x1, y1, x2, y2)的格式转换为(x, y, w, h)的格式。

- `fliplr`：以0.5的概率对输入的图片进行水平翻转，增加数据多样性。

- `label_pad`：对输入的标签进行填充，使得每个图片都有相同数量的标签。padding_size表示填充后标签的数量，padding_value表示填充的值。

- `image_norm`：将输入的图片像素值从[0, 255]范围内缩放到[0, 1]范围内。

- `image_transpose`：将输入的图片从BGR格式转换为RGB格式，并将图片的通道数从HWC格式转换为CHW格式。

测试数据增强需要用test_transforms字段标注，配置方法同训练。

## 自定义数据增强
编写指南：

- 在[mindyolo/data/dataset.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/data/dataset.py)文件COCODataset类中添加自定义数据增强方法   
- 数据增强方法的输入通常包含图片、标签和自定义参数。  
- 编写函数体内容，自定义输出

一个典型的数据增强方法：
```python
#在mindyolo/data/dataset.py COCODataset 添加子方法
    def data_trans_func(self, image, labels, args1=x1, args2=x2, ..., argsn=xn):
        # 数据增强逻辑
        ......
        return image, labels
```
自定义一个功能为旋转的数据增强函数
```python
#mindyolo/data/dataset.py
    def rotate(self, image, labels, angle):
        # rotate image
        image = np.rot90(image, angle // 90)
        if len(labels):
            if angle == 90:
                labels[:, 0], labels[:, 1] = 1 - labels[:, 1], labels[:, 0]
            elif angle == 180:
                labels[:, 0], labels[:, 1] = 1 - labels[:, 0], 1 - labels[:, 1]
            elif angle == 270:
                labels[:, 0], labels[:, 1] = labels[:, 1], 1 - labels[:, 0]
        return image, labels
```

使用指南：
- 在模型的yaml文件中，以字典的形式定义此数据增强方法。与上文所述用法一致
```yaml
    - {func_name: rotate, angle: 90}
```

效果展示：

&nbsp; &nbsp; &nbsp; <img width='600' src="https://github.com/chenyang23333/images/raw/main/rotate.png"/>

