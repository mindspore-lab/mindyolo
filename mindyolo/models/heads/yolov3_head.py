import math
import numpy as np

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor, Parameter
from mindyolo.utils import logger


class YOLOv3Head(nn.Cell):
    """
    YOLOv3 Detect Head, convert the output result to a prediction box based on the anchor point.
    """
    def __init__(self, nc=80, anchors=(), stride=(), ch=()):  # detection layer
        super(YOLOv3Head, self).__init__()
        self.is_export = False

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
        anchors, anchor_grid = self._check_anchor_order(anchors=anchors.reshape((self.nl, -1, 2)),
                                                        anchor_grid=anchors.reshape((self.nl, 1, -1, 1, 1, 2)),
                                                        stride=stride)
        anchors = anchors / stride.reshape((-1, 1, 1))

        self.stride = Parameter(Tensor(stride, ms.int32), requires_grad=False)
        self.anchors = Parameter(Tensor(anchors, ms.float32), requires_grad=False)  # shape(nl,na,2)
        self.anchor_grid = Parameter(Tensor(anchor_grid, ms.float32), requires_grad=False)  # shape(nl,1,na,1,1,2)

        self.m = nn.CellList([nn.Conv2d(x, self.no * self.na, 1,
                                        pad_mode="valid",
                                        has_bias=True) for x in ch])  # output conv

    def construct(self, x):
        z = ()  # inference output
        outs = ()
        for i in range(self.nl):
            out = self.m[i](x[i])  # conv
            if self.is_export:
                outs += (out,)
                continue
            bs, _, ny, nx = out.shape  # (bs,255,20,20)
            out = out.view(bs, self.na, self.no, ny, nx).transpose((0, 1, 3, 4, 2))  # (bs,3,20,20,85)
            outs += (out,)

            if not self.training:  # inference
                grid_tensor = self._make_grid(nx, ny, out.dtype)

                y = ops.Sigmoid()(out)
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid_tensor) * self.stride[i]  # xy
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
            b[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else np.log(cf / cf.sum())  # cls
            mi.bias = ops.assign(mi.bias, Tensor(b, ms.float32).view(-1))

    @staticmethod
    def _make_grid(nx=20, ny=20, dtype=ms.float32):
        # meshgrid is not supported on a specific model of machine
        # an alternative solution is adopted, which will be updated later
        x = mnp.arange(nx).astype(ms.int32)
        y = mnp.arange(ny).astype(ms.int32)
        xv = ops.tile(x.reshape(1, nx), (ny, 1))
        yv = ops.tile(y.reshape(ny, 1), (1, nx))
        #xv, yv = ops.meshgrid((mnp.arange(nx), mnp.arange(ny)))
        return ops.cast(ops.stack((xv, yv), 2).view((1, 1, ny, nx, 2)), dtype)

    @staticmethod
    def _check_anchor_order(anchors, anchor_grid, stride):
        # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
        a = np.prod(anchor_grid, -1).reshape((-1,))  # anchor area
        da = a[-1] - a[0]  # delta a
        ds = stride[-1] - stride[0]  # delta s
        if np.sign(da) != np.sign(ds):  # same order
            logger.warning('Reversing anchor order')
            anchors = anchors[::-1, ...]
            anchor_grid = anchor_grid[::-1, ...]
        return anchors, anchor_grid
