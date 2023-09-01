import math
import numpy as np

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Parameter, Tensor, nn, ops

from ..layers import DFL, ConvNormAct, Identity
from ..layers.utils import meshgrid


class YOLOv8Head(nn.Cell):
    # YOLOv8 Detect head for detection models
    def __init__(self, nc=80, reg_max=16, stride=(), ch=(), sync_bn=False):  # detection layer
        super().__init__()
        # self.dynamic = False # force grid reconstruction

        assert isinstance(stride, (tuple, list)) and len(stride) > 0
        assert isinstance(ch, (tuple, list)) and len(ch) > 0

        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = Parameter(Tensor(stride, ms.int32), requires_grad=False)

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.CellList(
            [
                nn.SequentialCell(
                    [
                        ConvNormAct(x, c2, 3, sync_bn=sync_bn),
                        ConvNormAct(c2, c2, 3, sync_bn=sync_bn),
                        nn.Conv2d(c2, 4 * self.reg_max, 1, has_bias=True),
                    ]
                )
                for x in ch
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
                for x in ch
            ]
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else Identity()

    def construct(self, x):
        shape = x[0].shape  # BCHW
        out = ()
        for i in range(self.nl):
            out += (ops.concat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1),)

        p = None
        if not self.training:
            _anchors, _strides = self.make_anchors(out, self.stride, 0.5)
            _anchors, _strides = _anchors.swapaxes(0, 1), _strides.swapaxes(0, 1)
            _x = ()
            for i in range(len(out)):
                _x += (out[i].view(shape[0], self.no, -1),)
            _x = ops.concat(_x, 2)
            box, cls = _x[:, : self.reg_max * 4, :], _x[:, self.reg_max * 4 : self.reg_max * 4 + self.nc, :]
            # box, cls = ops.concat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
            dbox = self.dist2bbox(self.dfl(box), ops.expand_dims(_anchors, 0), xywh=True, axis=1) * _strides
            p = ops.concat((dbox, ops.Sigmoid()(cls)), 1)
            p = ops.transpose(p, (0, 2, 1))  # (bs, no-84, nbox) -> (bs, nbox, no-84)

        return out if self.training else (p, out)

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


class YOLOv8SegHead(YOLOv8Head):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, reg_max=16, nm=32, npr=256, stride=(), ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, reg_max, stride, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = YOLOv8Head.construct

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.CellList([nn.SequentialCell(ConvNormAct(x, c4, 3), ConvNormAct(c4, c4, 3), nn.Conv2d(c4, self.nm, 1, has_bias=True)) for x in ch])

    def construct(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = ops.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)  # x: out if self.training else (p, out)
        if self.training:
            return x, mc, p

        mc = ops.transpose(mc, (0, 2, 1))  # (bs, 32, nbox) -> (bs, nbox, 32)
        # cat: (bs, nbox, no-84), (bs, nbox, 32) -> (bs, nbox, 84+32)
        return ops.cat([x[0], mc], 2), (x[1], mc, p)


class Proto(nn.Cell):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = ConvNormAct(c1, c_, k=3)
        self.upsample = nn.Conv2dTranspose(c_, c_, 2, 2, padding=0, has_bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = ConvNormAct(c_, c_, k=3)
        self.cv3 = ConvNormAct(c_, c2)

    def construct(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))
