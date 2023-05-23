import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, nn


class ImplicitA(nn.Cell):
    """
    https://arxiv.org/pdf/2105.04206v1.pdf. Implicit knowledge in YOLOR combined with convolution
        feature map in addition and multiplication manner: Implicit knowledge in YOLOR can be simplified to a vector by
        pre-computing at the inference stage. This vector can be combined with the bias and weight of the previous or
        subsequent convolutional layer.
    """

    def __init__(self, channel, mean=0.0, std=0.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = Parameter(Tensor(np.random.normal(self.mean, self.std, (1, channel, 1, 1)), ms.float32))

    def construct(self, x):
        return self.implicit + x


class ImplicitM(nn.Cell):
    """
    https://arxiv.org/pdf/2105.04206v1.pdf. Implicit knowledge in YOLOR combined with convolution
        feature map in addition and multiplication manner: Implicit knowledge in YOLOR can be simplified to a vector by
        pre-computing at the inference stage. This vector can be combined with the bias and weight of the previous or
        subsequent convolutional layer.
    """

    def __init__(self, channel, mean=0.0, std=0.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = Parameter(Tensor(np.random.normal(self.mean, self.std, (1, channel, 1, 1)), ms.float32))

    def construct(self, x):
        return self.implicit * x
