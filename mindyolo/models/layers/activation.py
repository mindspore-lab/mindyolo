"""
Custom activation operators.
"""
from mindspore import nn, ops


class Swish(nn.Cell):
    """
    Swish activation function: x * sigmoid(βx). If beta equals 1, you can use nn.SiLU instead.
    """

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def construct(self, x):
        return x * ops.sigmoid(self.beta * x)
