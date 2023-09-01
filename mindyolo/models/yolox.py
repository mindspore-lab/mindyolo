import numpy as np

import mindspore as ms
from mindspore import Tensor, nn

from mindyolo.models.registry import register_model
from .heads import YOLOXHead
from .model_factory import build_model_from_cfg

__all__ = ["YOLOX", "yolox"]


def _cfg(url="", **kwargs):
    return {"url": url, **kwargs}

default_cfgs = {"yolox": _cfg(url="")}


class YOLOX(nn.Cell):
    """connect yolox backbone and head"""

    def __init__(self, cfg, in_channels=3, num_classes=80, sync_bn=False):
        super(YOLOX, self).__init__()
        self.cfg = cfg
        self.stride = Tensor(np.array(cfg.stride), ms.int32)
        ch, nc = in_channels, num_classes
        self.nc = nc
        self.model = build_model_from_cfg(model_cfg=cfg, in_channels=ch, num_classes=nc, sync_bn=sync_bn)
        self.names = [str(i) for i in range(nc)]

        self.initialize_weights()

    def construct(self, x):
        return self.model(x)

    def initialize_weights(self):
        # reset parameter for Detect Head
        m = self.model.model[-1]
        assert isinstance(m, YOLOXHead)
        m.initialize_biases()


@register_model
def yolox(cfg, in_channels=3, num_classes=None, **kwargs) -> YOLOX:
    """Get yolox model."""
    model = YOLOX(cfg, in_channels=in_channels, num_classes=num_classes, **kwargs)
    return model
