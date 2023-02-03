import math
import numpy as np
from copy import deepcopy

import mindspore as ms
from mindspore import nn, ops, Tensor

from .heads.yolov7_head import YOLOv7Head, YOLOv7AuxHead
from .model_factory import build_model_from_cfg


class YOLOv7(nn.Cell):
    def __init__(self, cfg, in_channels=3, num_classes=None,
                 sync_bn=False, recompute=False, recompute_layers=0):
        super(YOLOv7, self).__init__()
        self.cfg = cfg
        ch, nc = in_channels, num_classes

        self.nc = nc  # override yaml value
        self.model = build_model_from_cfg(model_cfg=cfg, in_channels=ch, num_classes=nc, sync_bn=sync_bn)
        self.names = [str(i) for i in range(nc)]  # default names

        # Recompute
        if recompute and recompute_layers > 0:
            for i in range(recompute_layers):
                self.model.model[i].recompute()
            print(f"Turn on recompute, and the results of the first {recompute_layers} layers "
                  f"will be recomputed.")

        # TODO: move to module init (YOLOv7Head/YOLOv7AuxHead)
        # Build strides, anchors
        m = self.model.model[-1]  # Detect Head
        if isinstance(m, YOLOv7Head):
            m.stride = Tensor(np.array(self.cfg.stride), ms.int32)
            self.check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self.stride_max = int(max(self.cfg.stride))
            self._initialize_biases()
        if isinstance(m, YOLOv7AuxHead):
            m.stride = Tensor(np.array(self.cfg.stride), ms.int32)
            self.check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self.stride_max = int(max(self.cfg.stride))
            self._initialize_aux_biases()

    def construct(self, x):
        return self.model(x)

    # TODO: move to module init (YOLOv7Head/YOLOv7AuxHead)
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.model.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            s = s.asnumpy()
            b = mi.bias.view(m.na, -1).asnumpy()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else np.log(cf / cf.sum())  # cls
            mi.bias = ops.assign(mi.bias, Tensor(b, ms.float32).view(-1))

    # TODO: move to module init (YOLOv7Head/YOLOv7AuxHead)
    def _initialize_aux_biases(self, cf=None): # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.model[-1]  # Detect() module
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

    # TODO: move to module init (YOLOv7Head/YOLOv7AuxHead)
    @staticmethod
    def check_anchor_order(m):
        # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
        a = ops.ReduceProd()(m.anchor_grid, -1).view(-1)  # anchor area
        da = a[-1] - a[0]  # delta a
        ds = m.stride[-1] - m.stride[0]  # delta s
        if ops.Sign()(da) != ops.Sign()(ds):  # same order
            print('Reversing anchor order')
            m.anchors[:] = ops.ReverseV2(axis=0)(m.anchors)
            m.anchor_grid[:] = ops.ReverseV2(axis=0)(m.anchor_grid)


# TODO: Add Register
def yolov7(cfg, in_channels=3, num_classes=None, **kwargs) -> YOLOv7:
    """Get GoogLeNet model.
     Refer to the base class `models.GoogLeNet` for more details."""

    model = YOLOv7(cfg=cfg, in_channels=in_channels, num_classes=num_classes, **kwargs)

    return model

if __name__ == '__main__':
    from mindyolo.utils.config import parse_config
    from mindyolo.models.model_factory import create_model
    opt = parse_config()
    model = create_model(model_name="yolov7",
                         model_cfg=opt.YOLOv7,
                         num_classes=opt.Data.nc,
                         sync_bn=opt.sync_bn,
                         recompute=opt.recompute,
                         recompute_layers=opt.recompute_layers)
    x = Tensor(np.random.randn(1, 3, 640, 640), ms.float32)
    out = model(x)
    out = out[0] if isinstance(out, (list, tuple)) else out
    print(f"Output shape is {[o.shape for o in out]}")
