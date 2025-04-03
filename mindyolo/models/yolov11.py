import numpy as np

import mindspore as ms
from mindspore import Tensor, nn

from mindyolo.models.heads.yolov11_head import YOLOv11Head
from mindyolo.models.model_factory import build_model_from_cfg
from mindyolo.models.registry import register_model

__all__ = ["YOLOv11", "yolov11"]


def _cfg(url="", **kwargs):
    return {"url": url, **kwargs}


default_cfgs = {"yolov11": _cfg(url="")}


class YOLOv11(nn.Cell):
    def __init__(self, cfg, in_channels=3, num_classes=None, sync_bn=False):
        super(YOLOv11, self).__init__()
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
        if isinstance(m, YOLOv11Head):
            m.initialize_biases()
            m.dfl.initialize_conv_weight()


@register_model
def yolov11(cfg, in_channels=3, num_classes=None, **kwargs) -> YOLOv11:
    """Get yolov11 model."""
    model = YOLOv11(cfg=cfg, in_channels=in_channels, num_classes=num_classes, **kwargs)
    return model
