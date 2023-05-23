from mindspore import nn


class MP(nn.Cell):
    """
    Use the same step size and kernel size for maxpool.
    """

    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def construct(self, x):
        return self.m(x)


class SP(nn.Cell):
    """
    Use autopad for maxpool.
    """

    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def construct(self, x):
        return self.m(x)


class MaxPool2d(nn.Cell):
    """
    Maxpool with pad.
    """

    def __init__(self, kernel_size, stride, padding=0):
        super(MaxPool2d, self).__init__()
        assert isinstance(padding, int)
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (padding, padding), (padding, padding)))
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def construct(self, x):
        x = self.pad(x)
        x = self.pool(x)
        return x
