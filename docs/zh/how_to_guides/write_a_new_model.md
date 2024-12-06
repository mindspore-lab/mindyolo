# 模型编写指南
本文档提供MindYOLO编写自定义模型的教程。<br>
分为三个部分：

- 模型定义：我们可以直接定义一个网络，也可以使用yaml文件方式定义一个网络。
- 注册模型：可选，注册之后可以在create_model接口中使用文件名创建自定义的模型
- 验证: 验证模型是否可运行

## 模型定义

### 1.直接使用python代码来编写网络

#### 模块导入
导入MindSpore框架中的nn模块和ops模块，用于定义神经网络的组件和操作。
```python
import mindspore.nn as nn
import mindspore.ops.operations as ops
```

#### 创建模型
定义了一个继承自nn.Cell的模型类MyModel。在构造函数__init__中，定义模型的各个组件：

```python
class MyModel(nn.Cell):
    def __init__(self):
        super(MyModel, self).__init__()
        #conv1是一个2D卷积层，输入通道数为3，输出通道数为16，卷积核大小为3x3，步长为1，填充为1。
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        #relu是一个ReLU激活函数操作。
        self.relu = ops.ReLU()
        #maxpool是一个2D最大池化层，池化窗口大小为2x2，步长为2。
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
#### 创建模型实例
通过实例化MyModel类，创建一个模型实例model，后续可以使用该实例进行模型的训练和推理。
```python

model = MyModel()

```

### 2.使用yaml文件编写网络
通常需要以下三个步骤：

- 新建一个mymodel.yaml文件
- 新建对应的mymodel.py文件 
- 在mindyolo/models/_init_.py文件中引入该模型

以下是编写mymodel.yaml文件的详细指导:<br>
以编写一个简单网络为例：
以yaml格式编写必要参数，后续在mymodel.py文件里面可以用到这些参数。
其中network部分为模型网络 <br>
[[from, number, module, args], ...]：每个元素代表一个网络层的配置。<br>
```yaml
# __BASE__中的yaml表示用于继承的基础配置文件，重复的参数会被当前文件覆盖；
__BASE__:
  - '../coco.yaml'
  - './hyp.scratch-high.yaml'

per_batch_size: 32
img_size: 640
sync_bn: False

network:
  model_name: mymodel
  depth_multiple: 1.0  # model depth multiple
  width_multiple: 1.0  # layer channel multiple
  stride: [ 8, 16, 32 ]

  # 骨干网络部分的配置，每层的元素含义为
  # [from, number, module, args]
  # 以第一层为例，[-1, 1, ConvNormAct, [32, 3, 1]], 表示输入来自 `-1`(上一层) ，重复次数为 1，模块名为 ConvNormAct，模块输入参数为 [32, 3, 1]；
  backbone: 
    [[-1, 1, ConvNormAct, [32, 3, 1]],  # 0
     [-1, 1, ConvNormAct, [64, 3, 2]],  # 1-P1/2
     [-1, 1, Bottleneck, [64]],
     [-1, 1, ConvNormAct, [128, 3, 2]],  # 3-P2/4
     [-1, 2, Bottleneck, [128]],
     [-1, 1, ConvNormAct, [256, 3, 2]],  # 5-P3/8
     [-1, 8, Bottleneck, [256]],
      ]
  
  #head部分的配置 
  head: 
    [
    [ -1, 1, ConvNormAct, [ 512, 3, 2 ] ],  # 7-P4/16
      [ -1, 8, Bottleneck, [ 512 ] ],
      [ -1, 1, ConvNormAct, [ 1024, 3, 2 ] ],  # 9-P5/32
      [ -1, 4, Bottleneck, [ 1024 ] ],  # 10
    ]
```

编写mymodel.py文件:
#### 模块导入
需要导入套件内的模块。 如`from .registry import register_model`等等

```python
import numpy as np

import mindspore as ms
from mindspore import Tensor, nn


from .initializer import initialize_defult #用于初始化模型的默认参数，包括权重初始化方式、BN 层参数等。
from .model_factory import build_model_from_cfg #用于根据 YAML 配置文件中的参数构建目标检测模型，并返回该模型的实例。
from .registry import register_model #用于将自定义的模型注册到 Mindyolo 中，以便在 YAML 配置文件中使用。

#可见性声明
__all__ = ["MYmodel", "mymodel"]
```
#### 创建配置字典
_cfg函数是一个辅助函数，用于创建配置字典。它接受一个url参数和其他关键字参数，并返回一个包含url和其他参数的字典。<br>
default_cfgs是一个字典，用于存储默认配置。在这里，mymodel作为键，使用_cfg函数创建了一个配置字典。
```python
def _cfg(url="", **kwargs):
    return {"url": url, **kwargs}

default_cfgs = {"mymodel": _cfg(url="")}
```
#### 创建模型
在`MindSpore`中，模型的类继承于`nn.Cell`，一般来说需要重载以下两个函数：

- 在`__init__`函数中，应当定义模型中需要用到的module层。
- 在`construct`函数中定义模型前向逻辑。 <br>

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

## 注册模型（可选）
如果需要使用mindyolo接口初始化自定义的模型，那么需要先对模型进行**注册**和**导入**

**模型注册** <br>
```python
@register_model #注册后的模型可以通过 create_model 接口以模型名的方式进行访问；
def mymodel(cfg, in_channels=3, num_classes=None, **kwargs) -> MYmodel:
    """Get GoogLeNet model.
    Refer to the base class `models.GoogLeNet` for more details."""
    model = MYmodel(cfg=cfg, in_channels=in_channels, num_classes=num_classes, **kwargs)
    return model
```
**模型导入** <br>

```python
#在mindyolo/models/_init_.py文件中添加以下代码

from . import mymodel #mymodel.py文件通常放在mindyolo/models/目录下
__all__.extend(mymodel.__all__)
from .mymodel import *
```

## 验证main

初始编写阶段应当保证模型是可运行的。可通过下述代码块进行基础验证：
首先导入所需的模块和函数。然后，通过解析配置对象。

```python
if __name__ == "__main__":
    from mindyolo.models.model_factory import create_model
    from mindyolo.utils.config import parse_config

    opt = parse_config()
```
创建模型并指定相关参数，注意：如果要在create_model中使用文件名创建自定义的模型，那么需要先使用注册器@register_model进行注册，请参见上文 注册模型（可选)部分内容
```python
    model = create_model(
        model_name="mymodel",
        model_cfg=opt.net,
        num_classes=opt.data.nc,
        sync_bn=opt.sync_bn if hasattr(opt, "sync_bn") else False,
    ) 
    
```

否则，请使用import的方式引入模型

```python
    from mindyolo.models.mymodel import MYmodel
    model = MYmodel(
        model_name="mymodel",
        model_cfg=opt.net,
        num_classes=opt.data.nc,
        sync_bn=opt.sync_bn if hasattr(opt, "sync_bn") else False,
    ) 
```
最后，创建一个输入张量x并将其传递给模型进行前向计算。
```python    
    x = Tensor(np.random.randn(1, 3, 640, 640), ms.float32)
    out = model(x)
    out = out[0] if isinstance(out, (list, tuple)) else out
    print(f"Output shape is {[o.shape for o in out]}")
```


