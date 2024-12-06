import numpy as np

import mindspore as ms
from mindspore import Tensor, nn

from mindyolo.models.heads.yolov9_head import YOLOv9Head
from mindyolo.models.model_factory import build_model_from_cfg
from mindyolo.models.registry import register_model

__all__ = ["YOLOv9", "yolov9"]


def _cfg(url="", **kwargs):
    return {"url": url, **kwargs}


default_cfgs = {"yolov9": _cfg(url="")}


class YOLOv9(nn.Cell):
    def __init__(self, cfg, in_channels=3, num_classes=None, sync_bn=False):
        super(YOLOv9, self).__init__()
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
        m = self.model.model[-1]
        if isinstance(m, YOLOv9Head):
            m.initialize_biases()
            m.dfl.initialize_conv_weight()
            m.dfl2.initialize_conv_weight()


@register_model
def yolov9(cfg, in_channels=3, num_classes=None, **kwargs) -> YOLOv9:
    """Get yolov9 model."""
    model = YOLOv9(cfg=cfg, in_channels=in_channels, num_classes=num_classes, **kwargs)
    return model
