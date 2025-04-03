import math

from mindspore import nn, ops

from .activation import SiLU
from .common import Identity
from .utils import autopad
from .pool import SP, MaxPool2d


class ConvNormAct(nn.Cell):
    """Conv2d + BN + Act

    Args:
        c1 (int): In channels, the channel number of the input tensor of the Conv2d layer.
        c2 (int): Out channels, the channel number of the output tensor of the Conv2d layer.
        k (Union[int, tuple[int]]): Kernel size, Specifies the height and width of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the height
            and width of the convolution kernel. A tuple of two integers represents the height
            and width of the convolution kernel respectively. Default: 1.
        s (Union[int, tuple[int]]): Stride, the movement stride of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the movement step size
            in both height and width directions. A tuple of two integers represents the movement step size in the height
            and width directions respectively. Default: 1.
        p (Union[None, int, tuple[int]]): Padding, the number of padding on the height and width directions of the input.
            The data type is None or an integer or a tuple of four integers. If `padding` is an None, then padding with autopad.
            If `padding` is an integer, then the top, bottom, left, and right padding are all equal to `padding`.
            If `padding` is a tuple of 4 integers, then the top, bottom, left, and right padding
            is equal to `padding[0]`, `padding[1]`, `padding[2]`, and `padding[3]` respectively.
            The value should be greater than or equal to 0. Default: None.
        g (int): Group, Splits filter into groups, `c1` and `c2` must be
            divisible by `group`. If the group is equal to `c1` and `c2`,
            this 2D convolution layer also can be called 2D depthwise convolution layer. Default: 1.
        d (Union[int, tuple[int]]): Dilation, Dilation size of 2D convolution kernel.
            The data type is an integer or a tuple of two integers. If :math:`k > 1`, the kernel is sampled
            every `k` elements. The value of `k` on the height and width directions is in range of [1, H]
            and [1, W] respectively. Default: 1.
        act (Union[bool, nn.Cell]): Activation. The data type is bool or nn.Cell. If `act` is True,
            then the activation function uses nn.SiLU. If `act` is False, do not use activation function.
            If 'act' is nn.Cell, use the object of this cell as the activation function. Default: True.
        sync_bn (bool): Whether the BN layer use nn.SyncBatchNorm. Default: False.
    """

    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, pad_mode="pad", padding=autopad(k, p, d), group=g, dilation=d, has_bias=False
        )

        if sync_bn:
            self.bn = nn.SyncBatchNorm(c2, momentum=momentum, eps=eps)
        else:
            self.bn = nn.BatchNorm2d(c2, momentum=momentum, eps=eps)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Cell) else Identity())

    def construct(self, x):
        return self.act(self.bn(self.conv(x)))


class RepConv(nn.Cell):
    """Represented convolution, https://arxiv.org/abs/2101.03697

    Args:
        c1 (int): In channels, the channel number of the input tensor of the Conv2d layer.
        c2 (int): Out channels, the channel number of the output tensor of the Conv2d layer.
        k (Union[int, tuple[int]]): Kernel size, Specifies the height and width of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the height
            and width of the convolution kernel. A tuple of two integers represents the height
            and width of the convolution kernel respectively. Default: 1.
        s (Union[int, tuple[int]]): Stride, the movement stride of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the movement step size
            in both height and width directions. A tuple of two integers represents the movement step size in the height
            and width directions respectively. Default: 1.
        p (Union[None, int, tuple[int]]): Padding, the number of padding on the height and width directions of the input.
            The data type is None or an integer or a tuple of four integers. If `padding` is an None, then padding with autopad.
            If `padding` is an integer, then the top, bottom, left, and right padding are all equal to `padding`.
            If `padding` is a tuple of 4 integers, then the top, bottom, left, and right padding
            is equal to `padding[0]`, `padding[1]`, `padding[2]`, and `padding[3]` respectively.
            The value should be greater than or equal to 0. Default: None.
        g (int): Group, Splits filter into groups, `c1` and `c2` must be
            divisible by `group`. If the group is equal to `c1` and `c2`,
            this 2D convolution layer also can be called 2D depthwise convolution layer. Default: 1.
        act (Union[bool, nn.Cell]): Activation. The data type is bool or nn.Cell. If `act` is True,
            then the activation function uses nn.SiLU. If `act` is False, do not use activation function.
            If 'act' is nn.Cell, use the object of this cell as the activation function. Default: True.
        sync_bn (bool): Whether the BN layer use nn.SyncBatchNorm. Default: False.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, momentum=0.97, eps=1e-3, bn=True, sync_bn=False):
        super(RepConv, self).__init__()

        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = SiLU() if act is True else (act if isinstance(act, nn.Cell) else Identity())

        if sync_bn:
            BatchNorm = nn.SyncBatchNorm
        else:
            BatchNorm = nn.BatchNorm2d

        self.rbr_identity = BatchNorm(num_features=c1, momentum=(1 - 0.03), eps=1e-3) if bn and c2 == c1 and s == 1 else None
        self.rbr_dense = nn.SequentialCell(
            [
                nn.Conv2d(c1, c2, k, s, pad_mode="pad", padding=autopad(k, p), group=g, has_bias=False),
                BatchNorm(num_features=c2, momentum=momentum, eps=eps),
            ]
        )
        self.rbr_1x1 = nn.SequentialCell(
            nn.Conv2d(c1, c2, 1, s, pad_mode="pad", padding=padding_11, group=g, has_bias=False),
            BatchNorm(num_features=c2, momentum=momentum, eps=eps),
        )

    def construct(self, inputs):
        if self.rbr_identity is None:
            id_out = 0.0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def fuse(self):
        # TODO: The reparameterization function will be developed in subsequent versions
        pass


class DownC(nn.Cell):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, n=1, k=2, momentum=0.97, eps=1e-3, sync_bn=False):
        super(DownC, self).__init__()
        c_ = c1  # hidden channels
        self.cv1 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(c_, c2 // 2, 3, k, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.cv3 = ConvNormAct(c1, c2 // 2, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.mp = nn.MaxPool2d(kernel_size=k, stride=k)

    def construct(self, x):
        return ops.concat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), axis=1)


class Focus(nn.Cell):
    # Focus wh information into c-space
    def __init__(
        self, c1, c2, k=1, s=1, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = ConvNormAct(c1 * 4, c2, k, s, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)

    def construct(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(ops.concat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class DWConvNormAct(nn.Cell):
    """Conv2d + BN + Act, depthwise ConvNormAct used in yolox nano scale, an approach to reduce parameter number"""

    def __init__(
        self, c1, c2, k=1, s=1, p=None, d=1, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(DWConvNormAct, self).__init__()
        self.dconv = ConvNormAct(c1, c1, k, s, p, g=c1, d=d, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.pconv = ConvNormAct(c1, c2, k=1, s=1, p=p, g=1, d=d, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)

    def construct(self, x):
        return self.pconv(self.dconv(x))


class AConv(nn.Cell):
    def __init__(self, c1, c2, sync_bn=False):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super(AConv, self).__init__()
        self.avg_pool2d = nn.AvgPool2d(2)
        self.cv1 = ConvNormAct(c1, c2, 3, 2, 1, sync_bn=sync_bn)

    def construct(self, x):
        x = self.avg_pool2d(x)
        return self.cv1(x)


class ELAN1(nn.Cell):

    def __init__(self, c1, c2, c3, c4, sync_bn=False):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(ELAN1, self).__init__()
        self.c = c3//2
        self.cv1 = ConvNormAct(c1, c3, 1, 1, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(c3//2, c4, 3, 1, sync_bn=sync_bn)
        self.cv3 = ConvNormAct(c4, c4, 3, 1, sync_bn=sync_bn)
        self.cv4 = ConvNormAct(c3+(2*c4), c2, 1, 1, sync_bn=sync_bn)

    def construct(self, x):
        y = ()
        x = self.cv1(x)
        _c = x.shape[1] // 2
        x_tuple = ops.split(x, axis=1, split_size_or_sections=_c)
        y += x_tuple
        for m in [self.cv2, self.cv3]:
            out = m(y[-1])
            y += (out,)
        return self.cv4(ops.cat(y, 1))


class SPPELAN(nn.Cell):
    # spp-elan
    def __init__(self, c1, c2, c3, sync_bn=False):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(SPPELAN, self).__init__()
        self.c = c3
        self.cv1 = ConvNormAct(c1, c3, 1, 1, sync_bn=sync_bn)
        self.cv2 = SP(5)
        self.cv3 = SP(5)
        self.cv4 = SP(5)
        self.cv5 = ConvNormAct(4*c3, c2, 1, 1, sync_bn=sync_bn)

    def construct(self, x):
        y = (self.cv1(x),)
        for m in [self.cv2, self.cv3, self.cv4]:
            out = m(y[-1])
            y += (out,)
        return self.cv5(ops.cat(y, 1))


class CBLinear(nn.Cell):
    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):  # ch_in, ch_outs, kernel, stride, padding, groups
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, pad_mode="pad", padding=autopad(k, p), group=g, has_bias=True)

    def construct(self, x):
        outs = self.conv(x).split(self.c2s, axis=1)
        return outs


class CBFuse(nn.Cell):
    def __init__(self, idx):
        super(CBFuse, self).__init__()
        self.idx = idx

    def construct(self, xs):
        target_size = xs[-1].shape[2:]
        res = ()
        for i, x in enumerate(xs[:-1]):
            res += (ops.interpolate(x[self.idx[i]], size=target_size, mode='nearest'), )
        out = ops.sum(ops.stack(res + xs[-1:]), dim=0)
        return out


class ADown(nn.Cell):
    def __init__(self, c1, c2, sync_bn=False):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super(ADown, self).__init__()
        self.c = c2 // 2
        self.avg_pool2d = nn.AvgPool2d(2)
        self.max_pool2d = MaxPool2d(3, 2, padding=1)
        self.cv1 = ConvNormAct(c1 // 2, self.c, 3, 2, 1, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(c1 // 2, self.c, 1, 1, 0, sync_bn=sync_bn)

    def construct(self, x):
        x = self.avg_pool2d(x)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = self.max_pool2d(x2)
        x2 = self.cv2(x2)
        return ops.cat((x1, x2), 1)


class DWConv(ConvNormAct):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True, sync_bn=False):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act, sync_bn=sync_bn)
