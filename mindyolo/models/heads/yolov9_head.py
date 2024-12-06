import math
import numpy as np

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Parameter, Tensor, nn, ops

from ..layers import DFL, ConvNormAct, Identity
from ..layers.utils import meshgrid, make_divisible


class YOLOv9Head(nn.Cell):
    # YOLOv9 Detect head for detection models
    def __init__(self, nc=80, reg_max=16, stride=(), ch=(), sync_bn=False):  # detection layer
        super().__init__()
        # self.dynamic = False # force grid reconstruction

        assert isinstance(stride, (tuple, list)) and len(stride) > 0
        assert isinstance(ch, (tuple, list)) and len(ch) > 0

        self.nc = nc  # number of classes
        self.nl = len(ch) // 2  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = Parameter(Tensor(stride, ms.int32), requires_grad=False)

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max(
            (ch[0], min((self.nc * 2, 128))))  # channels
        c4, c5 = make_divisible(max((ch[self.nl] // 4, self.reg_max * 4, 16)), 4), max(
            (ch[self.nl], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.CellList(
            [
                nn.SequentialCell(
                    [
                        ConvNormAct(x, c2, 3, sync_bn=sync_bn),
                        ConvNormAct(c2, c2, 3, g=4, sync_bn=sync_bn),
                        nn.Conv2d(c2, 4 * self.reg_max, 1, group=4, has_bias=True),
                    ]
                )
                for x in ch[:self.nl]
            ]
        )
        self.cv3 = nn.CellList(
            [
                nn.SequentialCell(
                    [
                        ConvNormAct(x, c3, 3, sync_bn=sync_bn),
                        ConvNormAct(c3, c3, 3, sync_bn=sync_bn),
                        nn.Conv2d(c3, self.nc, 1, has_bias=True),
                    ]
                )
                for x in ch[:self.nl]
            ]
        )
        self.cv4 = nn.CellList(
            [
                nn.SequentialCell(
                    [
                        ConvNormAct(x, c4, 3, sync_bn=sync_bn),
                        ConvNormAct(c4, c4, 3, g=4, sync_bn=sync_bn),
                        nn.Conv2d(c4, 4 * self.reg_max, 1, group=4, has_bias=True),
                    ]
                )
                for x in ch[self.nl:]
            ]
        )
        self.cv5 = nn.CellList(
            [
                nn.SequentialCell(
                    [
                        ConvNormAct(x, c5, 3, sync_bn=sync_bn),
                        ConvNormAct(c5, c5, 3, sync_bn=sync_bn),
                        nn.Conv2d(c5, self.nc, 1, has_bias=True),
                    ]
                )
                for x in ch[self.nl:]
            ]
        )
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)

    def construct(self, x):
        shape = x[0].shape  # BCHW
        d1 = ()
        d2 = ()
        for i in range(self.nl):
            d1 += (ops.concat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1),)
            d2 += (ops.concat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1),)

        p = None
        if not self.training:
            _anchors, _strides = self.make_anchors(d1, self.stride, 0.5)
            _anchors, _strides = _anchors.swapaxes(0, 1), _strides.swapaxes(0, 1)

            _x = ()
            for i in range(len(d1)):
                _x += (d1[i].view(shape[0], self.no, -1),)
            _x = ops.concat(_x, 2)
            box, cls = _x[:, : self.reg_max * 4, :], _x[:, self.reg_max * 4: self.reg_max * 4 + self.nc, :]
            dbox = self.dist2bbox(self.dfl(box), ops.expand_dims(_anchors, 0), xywh=True, axis=1) * _strides

            _x2 = ()
            for i in range(len(d2)):
                _x2 += (d2[i].view(shape[0], self.no, -1),)
            _x2 = ops.concat(_x2, 2)
            box2, cls2 = _x2[:, : self.reg_max * 4, :], _x2[:, self.reg_max * 4: self.reg_max * 4 + self.nc, :]
            dbox2 = self.dist2bbox(self.dfl2(box2), ops.expand_dims(_anchors, 0), xywh=True, axis=1) * _strides

            p = (ops.concat((dbox, ops.Sigmoid()(cls)), 1), ops.concat((dbox2, ops.Sigmoid()(cls2)), 1))
            p = (ops.transpose(p[0], (0, 2, 1)), ops.transpose(p[1], (0, 2, 1)))  # (bs, no-84, nbox) -> (bs, nbox, no-84)

        return (d1, d2) if self.training else (p, (d1, d2))

    @staticmethod
    def make_anchors(feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = (), ()
        dtype = feats[0].dtype
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = mnp.arange(w, dtype=dtype) + grid_cell_offset  # shift x
            sy = mnp.arange(h, dtype=dtype) + grid_cell_offset  # shift y
            # FIXME: Not supported on a specific model of machine
            sy, sx = meshgrid((sy, sx), indexing="ij")
            anchor_points += (ops.stack((sx, sy), -1).view(-1, 2),)
            stride_tensor += (ops.ones((h * w, 1), dtype) * stride,)
        return ops.concat(anchor_points), ops.concat(stride_tensor)

    @staticmethod
    def dist2bbox(distance, anchor_points, xywh=True, axis=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        lt, rb = ops.split(distance, split_size_or_sections=2, axis=axis)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return ops.concat((c_xy, wh), axis)  # xywh bbox
        return ops.concat((x1y1, x2y2), axis)  # xyxy bbox

    def initialize_biases(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            s = s.asnumpy()
            a[-1].bias = ops.assign(a[-1].bias, Tensor(np.ones(a[-1].bias.shape), ms.float32))
            b_np = b[-1].bias.data.asnumpy()
            b_np[: m.nc] = math.log(5 / m.nc / (640 / int(s)) ** 2)
            b[-1].bias = ops.assign(b[-1].bias, Tensor(b_np, ms.float32))
        for a2, b2, s2 in zip(m.cv4, m.cv5, m.stride):  # from
            s2 = s2.asnumpy()
            a2[-1].bias = ops.assign(a2[-1].bias, Tensor(np.ones(a2[-1].bias.shape), ms.float32))
            b_np2 = b2[-1].bias.data.asnumpy()
            b_np2[: m.nc] = math.log(5 / m.nc / (640 / int(s2)) ** 2)
            b2[-1].bias = ops.assign(b2[-1].bias, Tensor(b_np2, ms.float32))
