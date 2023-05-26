from mindspore import nn, ops

from .conv import ConvNormAct, DWConvNormAct


class Bottleneck(nn.Cell):
    # Standard bottleneck
    def __init__(
        self, c1, c2, shortcut=True, k=(1, 3), g=(1, 1), e=0.5, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, k[0], 1, g=g[0], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c_, c2, k[1], 1, g=g[1], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        if self.add:
            out = x + self.conv2(self.conv1(x))
        else:
            out = self.conv2(self.conv1(x))
        return out

    
class Residualblock(nn.Cell):
    def __init__(
        self, c1, c2, k=(1, 3), g=(1, 1), act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, kernels, groups, expand
        super().__init__()
        self.conv1 = ConvNormAct(c1, c2, k[0], 1, g=g[0], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c2, c2, k[1], 1, g=g[1], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)

    def construct(self, x):
        out = x + self.conv2(self.conv1(x))
        return out


class C3(nn.Cell):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False):
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv3 = ConvNormAct(2 * c_, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)  # act=FReLU(c2)
        self.m = nn.SequentialCell(
            [
                Bottleneck(c_, c_, shortcut, k=(1, 3), e=1.0, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.m(c1)
        c3 = self.conv2(x)
        c4 = self.concat((c2, c3))
        c5 = self.conv3(c4)

        return c5


class C2f(nn.Cell):
    # CSP Bottleneck with 2 convolutions
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        _c = int(c2 * e)  # hidden channels
        self.cv1 = ConvNormAct(c1, 2 * _c, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(
            (2 + n) * _c, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn
        )  # optional act=FReLU(c2)
        self.m = nn.CellList(
            [
                Bottleneck(_c, _c, shortcut, k=(3, 3), g=(1, g), e=1.0, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )

    def construct(self, x):
        y = ()
        x = self.cv1(x)
        x_tuple = ops.split(x, axis=1, output_num=2)
        y += x_tuple
        for i in range(len(self.m)):
            m = self.m[i]
            out = m(y[-1])
            y += (out,)

        return self.cv2(ops.concat(y, axis=1))


class DWBottleneck(nn.Cell):
    # depthwise bottleneck used in yolox nano scale
    def __init__(
        self, c1, c2, shortcut=True, k=(1, 3), e=0.5, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, k[0], 1, act=True, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = DWConvNormAct(c_, c2, k[1], 1, act=True, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        if self.add:
            out = x + self.conv2(self.conv1(x))
        else:
            out = self.conv2(self.conv1(x))
        return out


class DWC3(nn.Cell):
    # depthwise DwC3 used in yolox nano scale, similar as C3
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False):
        super(DWC3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv3 = ConvNormAct(2 * c_, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)  # act=FReLU(c2)
        self.m = nn.SequentialCell(
            [
                DWBottleneck(c_, c_, shortcut, k=(1, 3), e=1.0, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.m(c1)
        c3 = self.conv2(x)
        c4 = self.concat((c2, c3))
        c5 = self.conv3(c4)

        return c5
