import numpy as np

import mindspore as ms
from mindspore import Tensor, nn

from mindyolo.models.layers.bottleneck import A2C2f, ABlock, C3k
from mindyolo.models.heads.yolov12_head import YOLOv12Head
from mindyolo.models.model_factory import build_model_from_cfg
from mindyolo.models.registry import register_model

__all__ = ["YOLOv12", "yolov12"]


def _cfg(url="", **kwargs):
    return {"url": url, **kwargs}


default_cfgs = {"yolov12": _cfg(url="")}


class YOLOv12(nn.Cell):
    def __init__(self, cfg, in_channels=3, num_classes=None, sync_bn=False):
        super(YOLOv12, self).__init__()
        self.cfg = cfg
        self.stride = Tensor(np.array(cfg.stride), ms.int32)
        self.stride_max = int(max(self.cfg.stride))
        ch, nc = in_channels, num_classes

        self.nc = nc  # override yaml value
        self.model = build_model_from_cfg(model_cfg=cfg, in_channels=ch, num_classes=nc, sync_bn=sync_bn)
        self.names = [str(i) for i in range(nc)]  # default names

        self.initialize_weights()

    def construct(self, x):
        return self.model(x)

    def initialize_weights(self):
        # reset parameter for Detect Head
        for i in range(len(self.model.model)-1):
            m = self.model.model[i]
            if isinstance(m, A2C2f) and isinstance(m.m[0], C3k):
                pass
            elif isinstance(m, A2C2f) and isinstance(m.m[0][0], ABlock):
                for i in range(len(m.m)):
                    for j in range(len(m.m[0])):
                        m.m[i][j].apply(m.m[i][j]._init_weights)

        m = self.model.model[-1]
        if isinstance(m, YOLOv12Head):
            m.initialize_biases()
            m.dfl.initialize_conv_weight()


@register_model
def yolov12(cfg, in_channels=3, num_classes=None, **kwargs) -> YOLOv12:
    """Get yolov12 model."""
    model = YOLOv12(cfg=cfg, in_channels=in_channels, num_classes=num_classes, **kwargs)
    return model
