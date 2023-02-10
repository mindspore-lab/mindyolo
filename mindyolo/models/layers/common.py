from mindspore import nn, ops

class Shortcut(nn.Cell):
    """
    Shortcut layer.
    """
    def construct(self, x):
        if isinstance(x, (tuple, list)) and len(x) == 2:
            return x[0] + x[1]
        return x


class Concat(nn.Cell):
    """
    Connect tensor in the specified axis.
    """
    def __init__(self, axis=1):
        super(Concat, self).__init__()
        self.axis = axis

    def construct(self, x):
        return ops.concat(x, self.axis)


class ReOrg(nn.Cell):
    """
    Reorganize the input Tensor (b, c, w, h) into a new shape (b, 4c, w/2, h/2).
    """
    def __init__(self):
        super(ReOrg, self).__init__()

    def construct(self, x):
        # in: (b,c,w,h) -> out: (b,4c,w/2,h/2)
        x1 = x[:, :, ::2, ::2]
        x2 = x[:, :, 1::2, ::2]
        x3 = x[:, :, ::2, 1::2]
        x4 = x[:, :, 1::2, 1::2]
        out = ops.concat((x1, x2, x3, x4), 1)
        return out


class Identity(nn.Cell):
    def construct(self, x):
        return x
