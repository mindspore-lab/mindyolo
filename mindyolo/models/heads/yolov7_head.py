import math
import numpy as np

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Parameter, Tensor, nn, ops

from mindyolo.utils import logger
from ..layers.implicit import ImplicitA, ImplicitM
from ..layers.utils import meshgrid


class YOLOv7Head(nn.Cell):
    """
    YOLOv7 Detect Head, convert the output result to a prediction box based on the anchor point.
    """

    def __init__(self, nc=80, anchors=(), stride=(), ch=()):  # detection layer
        super(YOLOv7Head, self).__init__()

        assert isinstance(anchors, (tuple, list)) and len(anchors) > 0
        assert isinstance(stride, (tuple, list)) and len(stride) > 0
        assert isinstance(ch, (tuple, list)) and len(ch) > 0

        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors

        # anchor preprocess
        anchors = np.array(anchors)
        stride = np.array(stride)
        anchors, anchor_grid = self._check_anchor_order(
            anchors=anchors.reshape((self.nl, -1, 2)),
            anchor_grid=anchors.reshape((self.nl, 1, -1, 1, 1, 2)),
            stride=stride,
        )
        anchors = anchors / stride.reshape((-1, 1, 1))

        self.stride = Parameter(Tensor(stride, ms.int32), requires_grad=False)
        self.anchors = Parameter(Tensor(anchors, ms.float32), requires_grad=False)  # shape(nl,na,2)
        self.anchor_grid = Parameter(Tensor(anchor_grid, ms.float32), requires_grad=False)  # shape(nl,1,na,1,1,2)
        self.convert_matrix = Parameter(
            Tensor(np.array([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]]), dtype=ms.float32),
            requires_grad=False,
        )

        self.m = nn.CellList(
            [nn.Conv2d(x, self.no * self.na, 1, pad_mode="valid", has_bias=True) for x in ch]
        )  # output conv

        self.ia = nn.CellList([ImplicitA(x) for x in ch])
        self.im = nn.CellList([ImplicitM(self.no * self.na) for _ in ch])

    def construct(self, x):
        z = ()  # inference output
        outs = ()
        for i in range(self.nl):
            out = self.m[i](self.ia[i](x[i]))  # conv
            out = self.im[i](out)
            bs, _, ny, nx = out.shape  # (bs,255,20,20)
            out = out.view(bs, self.na, self.no, ny, nx).transpose((0, 1, 3, 4, 2))  # (bs,3,20,20,85)
            outs += (out,)

            if not self.training:  # inference
                grid_tensor = self._make_grid(nx, ny, out.dtype)

                # y = ops.sigmoid(out)
                y = ops.Sigmoid()(out)
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + grid_tensor) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z += (y.view(bs, -1, self.no),)

        return outs if self.training else (ops.concat(z, 1), outs)

    def initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self
        for mi, s in zip(m.m, m.stride):  # from
            s = s.asnumpy()
            b = mi.bias.view(m.na, -1).asnumpy()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else np.log(cf / cf.sum())  # cls
            mi.bias = ops.assign(mi.bias, Tensor(b, ms.float32).view(-1))

    @staticmethod
    def _make_grid(nx=20, ny=20, dtype=ms.float32):
        # FIXME: Not supported on a specific model of machine
        xv, yv = meshgrid((mnp.arange(nx), mnp.arange(ny)))
        return ops.cast(ops.stack((xv, yv), 2).view((1, 1, ny, nx, 2)), dtype)

    @staticmethod
    def _check_anchor_order(anchors, anchor_grid, stride):
        # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
        a = np.prod(anchor_grid, -1).reshape((-1,))  # anchor area
        da = a[-1] - a[0]  # delta a
        ds = stride[-1] - stride[0]  # delta s
        if np.sign(da) != np.sign(ds):  # same order
            logger.warning("Reversing anchor order")
            anchors = anchors[::-1, ...]
            anchor_grid = anchor_grid[::-1, ...]
        return anchors, anchor_grid


class YOLOv7AuxHead(nn.Cell):
    """
    YOLOv7 Detect Aux Head, convert the output result to a prediction box based on the anchor point.
    """

    def __init__(self, nc=80, anchors=(), stride=(), ch=()):  # detection layer
        super(YOLOv7AuxHead, self).__init__()

        assert isinstance(anchors, (tuple, list)) and len(anchors) > 0
        assert isinstance(stride, (tuple, list)) and len(stride) > 0
        assert isinstance(ch, (tuple, list)) and len(ch) > 0

        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors

        # anchor preprocess
        anchors = np.array(anchors)
        stride = np.array(stride)
        anchors, anchor_grid = self._check_anchor_order(
            anchors=anchors.reshape((self.nl, -1, 2)),
            anchor_grid=anchors.reshape((self.nl, 1, -1, 1, 1, 2)),
            stride=stride,
        )
        anchors /= stride.reshape((-1, 1, 1))

        self.stride = Parameter(Tensor(stride, ms.int32), requires_grad=False)
        self.anchors = Parameter(Tensor(anchors, ms.float32), requires_grad=False)  # shape(nl,na,2)
        self.anchor_grid = Parameter(Tensor(anchor_grid, ms.float32), requires_grad=False)  # shape(nl,1,na,1,1,2)
        self.convert_matrix = Parameter(
            Tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]], dtype=ms.float32),
            requires_grad=False,
        )

        self.m = nn.CellList(
            [nn.Conv2d(x, self.no * self.na, 1, pad_mode="valid", has_bias=True) for x in ch[: self.nl]]
        )  # output conv
        self.m2 = nn.CellList(
            [nn.Conv2d(x, self.no * self.na, 1, pad_mode="valid", has_bias=True) for x in ch[self.nl :]]
        )  # output conv

        self.ia = nn.CellList([ImplicitA(x) for x in ch[: self.nl]])
        self.im = nn.CellList([ImplicitM(self.no * self.na) for _ in ch[: self.nl]])

    def construct(self, x):
        z = ()  # inference output
        outs_1 = ()
        outs_2 = ()
        for i in range(self.nl):
            out1 = self.m[i](self.ia[i](x[i]))  # conv
            out1 = self.im[i](out1)
            bs, _, ny, nx = out1.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            out1 = ops.Transpose()(out1.view(bs, self.na, self.no, ny, nx), (0, 1, 3, 4, 2))
            outs_1 += (out1,)

            out2 = self.m2[i](x[i + self.nl])
            out2 = ops.Transpose()(out2.view(bs, self.na, self.no, ny, nx), (0, 1, 3, 4, 2))
            outs_2 += (out2,)

            if not self.training:  # inference
                grid_tensor = self._make_grid(nx, ny, out1.dtype)

                # y = ops.sigmoid(out1)
                y = ops.Sigmoid()(out1)
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + grid_tensor) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z += (y.view(bs, -1, self.no),)
        outs = outs_1 + outs_2
        return outs if self.training else (ops.concat(z, 1), outs_1)

    def _initialize_aux_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self
        for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
            s = s.asnumpy()

            b = mi.bias.view(m.na, -1).asnumpy()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else np.log(cf / cf.sum())  # cls
            mi.bias = ops.assign(mi.bias, Tensor(b, ms.float32).view(-1))

            b2 = mi2.bias.view(m.na, -1).asnumpy()  # conv.bias(255) to (3,85)
            b2[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b2[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else np.log(cf / cf.sum())  # cls
            mi2.bias = ops.assign(mi2.bias, Tensor(b2, ms.float32).view(-1))

    @staticmethod
    def _make_grid(nx=20, ny=20, dtype=ms.float32):
        xv, yv = meshgrid((mnp.arange(nx), mnp.arange(ny)))
        return ops.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).astype(dtype)

    @staticmethod
    def _check_anchor_order(anchors, anchor_grid, stride):
        # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
        a = np.prod(anchor_grid, -1).reshape((-1,))  # anchor area
        da = a[-1] - a[0]  # delta a
        ds = stride[-1] - stride[0]  # delta s
        if np.sign(da) != np.sign(ds):  # same order
            logger.warning("Reversing anchor order")
            anchors = anchors[::-1, ...]
            anchor_grid = anchor_grid[::-1, ...]
        return anchors, anchor_grid
