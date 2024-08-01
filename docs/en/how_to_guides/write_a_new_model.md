# Model Writing Guide
This document provides a tutorial for writing custom models for MindYOLO. <br>
It is divided into three parts:

- Model definition: We can define a network directly or use a yaml file to define a network.
- Register model: Optional. After registration, you can use the file name in the create_model interface to create a custom model
- Verification: Verify whether the model is operational

## Model definition

### 1. Use python code directly to write the network

#### Module import
Import the nn module and ops module in the MindSpore framework to define the components and operations of the neural network.
```python
import mindspore.nn as nn
import mindspore.ops.operations as ops
```

#### Create a model
Define a model class MyModel that inherits from nn.Cell. In the constructor __init__, define the various components of the model:

```python
class MyModel(nn.Cell):
    def __init__(self):
        super(MyModel, self).__init__()
        #conv1是一个2D卷积层，输入通道数为3，输出通道数为16，卷积核大小为3x3，步长为1，填充为1。
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        #relu是一个ReLU激活函数操作。
        self.relu = ops.ReLU()
        #axpool是一个2D最大池化层，池化窗口大小为2x2，步长为2。
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        #conv2是另一个2D卷积层，输入通道数为16，输出通道数为32，卷积核大小为3x3，步长为1，填充为1。
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        #fc是一个全连接层，输入特征维度为32x8x8，输出特征维度为10。
        self.fc = nn.Dense(32 * 8 * 8, 10)

    #在construct方法中，定义了模型的前向传播过程。输入x经过卷积、激活函数、池化等操作后，通过展平操作将特征张量变为一维向量，然后通过全连接层得到最终的输出结果。    
    def construct(self, x): 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
```
#### Create a model instance
By instantiating the MyModel class, create a model instance model, which can be used for model training and reasoning later.
```python

model = MyModel()

```

### 2. Use yaml file to write network
Usually need the following three steps:

- Create a new mymodel.yaml file
- Create a corresponding mymodel.py file
- Introduce the model in the mindyolo/models/_init_.py file

Here is a detailed guide to writing mymodel.yaml file:<br>
Take writing a simple network as an example:
Write the necessary parameters in yaml format, and then use these parameters in the mymodel.py file.
The network part is the model network <br>
[[from, number, module, args], ...]: Each element represents the configuration of a network layer. <br>
```yaml
# The yaml in __BASE__ indicates the base configuration file for inheritance. Repeated parameters will be overwritten by the current file;
__BASE__:
-'../coco.yaml'
-'./hyp.scratch-high.yaml'

per_batch_size: 32
img_size: 640
sync_bn: False

network:
model_name: mymodel
depth_multiple: 1.0 # model depth multiple
width_multiple: 1.0 # layer channel multiple
stride: [ 8, 16, 32 ]

# Configuration of the backbone network. The meaning of each layer is
# [from, number, module, args]
# Take the first layer as an example, [-1, 1, ConvNormAct, [32, 3, 1]], which means the input comes from `-1` (the previous layer), the number of repetitions is 1, and the module name is ConvNormAct, module input parameters are [32, 3, 1];
backbone:
[[-1, 1, ConvNormAct, [32, 3, 1]], # 0
[-1, 1, ConvNormAct, [64, 3, 2]], # 1-P1/2
[-1, 1, Bottleneck, [64]],
[-1, 1, ConvNormAct, [128, 3, 2]], # 3-P2/4
[-1, 2, Bottleneck, [128]],
[-1, 1, ConvNormAct, [256, 3, 2]], # 5-P3/8
[-1, 8, Bottleneck, [256]],
]

#head part configuration
head:
[
[ -1, 1, ConvNormAct, [ 512, 3, 2 ] ], # 7-P4/16
[ -1, 8, Bottleneck, [ 512 ] ],
[ -1, 1, ConvNormAct, [ 1024, 3, 2 ] ], # 9-P5/32
[ -1, 4, Bottleneck, [ 1024 ] ], # 10
]
```

Write mymodel.py file:
#### Module import
It is necessary to import modules in the package. For example, `from .registry import register_model`, etc.

```python
import numpy as np

import mindspore as ms
from mindspore import Tensor, nn

from .initializer import initialize_defult #Used to initialize the default parameters of the model, including weight initialization method, BN layer parameters, etc.
from .model_factory import build_model_from_cfg #Used to build a target detection model according to the parameters in the YAML configuration file and return an instance of the model.
from .registry import register_model #Used to register a custom model in Mindyolo for use in the YAML configuration file.

#Visibility declaration
__all__ = ["MYmodel", "mymodel"]
```
#### Create a configuration dictionary
The _cfg function is an auxiliary function used to create a configuration dictionary. It accepts a url parameter and other keyword parameters and returns a dictionary containing the url and other parameters. <br>
default_cfgs is a dictionary used to store default configurations. Here, mymodel is used as the key to create a configuration dictionary using the _cfg function.
```python
def _cfg(url="", **kwargs):
    return {"url": url, **kwargs}

default_cfgs = {"mymodel": _cfg(url="")}
```
#### Create a model
In `MindSpore`, the model class inherits from `nn.Cell`. Generally, the following two functions need to be overloaded:

- In the `__init__` function, the module layer needed in the model should be defined.
- In the `construct` function, define the model forward logic. <br>

```python
class MYmodel(nn.Cell):
    
    def __init__(self, cfg, in_channels=3, num_classes=None, sync_bn=False):
        super(MYmodel, self).__init__()
        self.cfg = cfg
        self.stride = Tensor(np.array(cfg.stride), ms.int32)
        self.stride_max = int(max(self.cfg.stride))
        ch, nc = in_channels, num_classes

        self.nc = nc  # override yaml value
        self.model = build_model_from_cfg(model_cfg=cfg, in_channels=ch, num_classes=nc, sync_bn=sync_bn)
        self.names = [str(i) for i in range(nc)]  # default names
        
        initialize_defult()  # 可选，你可能需要initialize_defult方法以获得和pytorch一样的conv2d、dense层的初始化方式；

    def construct(self, x):
        return self.model(x)

```

## Register model (optional)
If you need to use the mindyolo interface to initialize a custom model, you need to first **register** and **import** the model

**Model registration** <br>
```python
@register_model #The registered model can be accessed by the create_model interface as a model name;
def mymodel(cfg, in_channels=3, num_classes=None, **kwargs) -> MYmodel:
    """Get GoogLeNet model.
    Refer to the base class `models.GoogLeNet` for more details."""
    model = MYmodel(cfg=cfg, in_channels=in_channels, num_classes=num_classes, **kwargs)
    return model
```
**Model import** <br>

```python
#Add the following code to the mindyolo/models/_init_.py file

from . import mymodel #mymodel.py files are usually placed in the mindyolo/models/directory
__all__.extend(mymodel.__all__)
from .mymodel import *
```

## Verify main

The initial writing phase should ensure that the model is runnable. Basic verification can be performed through the following code block:
First import the required modules and functions. Then, parse the configuration object.

```python
if __name__ == "__main__":
    from mindyolo.models.model_factory import create_model
    from mindyolo.utils.config import parse_config

    opt = parse_config()
```
Create a model and specify related parameters. Note: If you want to use the file name to create a custom model in create_model, you need to register it using the register @register_model first. Please refer to the above Register model (optional) section
```python
    model = create_model(
        model_name="mymodel",
        model_cfg=opt.net,
        num_classes=opt.data.nc,
        sync_bn=opt.sync_bn if hasattr(opt, "sync_bn") else False,
    )

```

Otherwise, please use import to introduce the model

```python
    from mindyolo.models.mymodel import MYmodel
    model = MYmodel(
        model_name="mymodel",
        model_cfg=opt.net,
        num_classes=opt.data.nc,
        sync_bn=opt.sync_bn if hasattr(opt, "sync_bn") else False,
    ) 
```
Finally, create an input tensor x and pass it to the model for forward computation.
```python
    x = Tensor(np.random.randn(1, 3, 640, 640), ms.float32)
    out = model(x)
    out = out[0] if isinstance(out, (list, tuple)) else out
    print(f"Output shape is {[o.shape for o in out]}")
```