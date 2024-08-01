
# Data Augmentation

## List of data enhancement methods that come with the package

| Data Enhancement Method Name | Summary Explanation                                            |
|------------------------------|----------------------------------------------------------------|
| mosaic                       | randomly select mosaic4 and mosaic9                            |
| mosaic4                      | 4-part splicing                                                |
| mosaic9                      | 9-point splicing                                               |
| mixup                        | linearly mix two images                                        |
| pastein                      | clipping enhancement                                           |
| random_perspective           | random perspective transformation                              |
| hsv_augment                  | random color transformation                                    |
| fliplr                       | flip horizontally                                              |
| flipud                       | vertical flip                                                  |
| letterbox                    | scale and fill                                                 |
| label_norm                   | label normalization and coordinates normalized to 0-1 to range |
| label_pad                    | fill label information into a fixed-size array                 |
| image_norm                   | image data normalization                                       |
| image_transpose              | channel transpose and dimension transpose                      |
| albumentations               | albumentations data enhancement                                |

These data augmentation functions are defined in [mindyolo/data/dataset.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/data/dataset.py).

## Instructions
The MindYOLO data enhancement method is configured in the yaml file. For example, to add a data enhancement during the training process, you need to add a dictionary list under the data.train_transforms field of the yaml file. The data enhancement methods are listed from top to bottom.

A typical data enhancement method configuration dictionary must have func_name, which represents the name of the applied data enhancement method, and then lists the parameters that need to be set for the method. If no parameter item is configured in the data enhancement configuration dictionary, the data enhancement method will be selected by default. value.

Data enhancement common configuration dictionary:
```yaml
- {func_name: data enhancement method name 1, args11=x11, args12=x12, ..., args1n=x1n}
- {func_name: data enhancement method name 2, args21=x21, args22=x22, ..., args2n=x2n}
...
- {func_name: data enhancement method name n, argsn1=xn1, argsn2=xn2, ..., argsnn=xnn}
```

Example enhanced with YOLOv7 training data:
```yaml
#File directory: configs/yolov7/hyp.scratch.tiny.yaml (https://github.com/mindspore-lab/mindyolo/blob/master/configs/yolov7/hyp.scratch.tiny.yaml)
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
_Note: func_name represents the name of the data enhancement method, prob, mosaic9_prob, translate, and scale are the method parameters. Among them, prob is a parameter common to all methods, indicating the execution probability of the data enhancement method. The default value is 1_

The specific operations performed by the above yaml file are as follows:

- `mosaic`: Perform mosaic operation on the input image with a probability of 1.0, that is, splicing 4 different images into one image. mosaic9_prob represents the probability of splicing using the 9-square grid method, and translate and scale represent the degree of random translation and scaling respectively.
as the picture shows:
<img width='600' src="https://github.com/ChongWei905/images/raw/main/mindyolo-mosaic-en.png"/>

- `mixup`: Perform a mixup operation on the input image with a probability of 0.05, that is, mix two different images. Among them, alpha and beta represent the mixing coefficient, and needed_mosaic represents whether mosaic needs to be used for mixing.

- `hsv_augment`: HSV enhancement, adjust the HSV color space of the input image with a probability of 1.0 to increase data diversity. Among them, hgain, sgain and vgain represent the degree of adjustment of H, S and V channels respectively.

- `pastein`: randomly paste some samples into the input image with a probability of 0.05. Among them, num_sample represents the number of randomly posted samples.

- `label_norm`: Convert the input label from the format of (x1, y1, x2, y2) to the format of (x, y, w, h).

- `fliplr`: Flip the input image horizontally with a probability of 0.5 to increase data diversity.

- `label_pad`: Pad the input labels so that each image has the same number of labels. padding_size represents the number of labels after padding, and padding_value represents the value of padding.

- `image_norm`: Scale the input image pixel value from the range [0, 255] to the range [0, 1].

- `image_transpose`: Convert the input image from BGR format to RGB format, and convert the number of channels of the image from HWC format to CHW format.

Test data enhancement needs to be marked with the test_transforms field, and the configuration method is the same as training.

## Custom data enhancement
Writing Guide:

- Add custom data enhancement methods to the COCODataset class in the [mindyolo/data/dataset.py](https://github.com/mindspore-lab/mindyolo/blob/master/mindyolo/data/dataset.py) file
- Inputs to data augmentation methods usually include images, labels, and custom parameters.  
-Write function body content and customize output

A typical data enhancement method:
```python
#Add submethods in mindyolo/data/dataset.py COCODataset
def data_trans_func(self, image, labels, args1=x1, args2=x2, ..., argsn=xn):
    # Data enhancement logic
    ...
    return image, labels
```
Customize a data enhancement function whose function is rotation
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

user's guidance:
- Define this data augmentation method in the form of a dictionary in the model's yaml file. Same usage as described above
```yaml
    - {func_name: rotate, angle: 90}
```

Show results:

&nbsp; &nbsp; &nbsp; <img width='600' src="https://github.com/ChongWei905/images/raw/main/mindyolo-rotate-en.png"/>

