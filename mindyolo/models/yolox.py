import math

import mindspore as ms
import numpy as np
from mindspore import nn, Tensor

from .initializer import initialize_defult

__all__ = [
    'YOLOX',
    'yolox'
]

from mindyolo.models.registry import register_model
from .model_factory import build_model_from_cfg
from .heads import YOLOXHead

def _cfg(url='', **kwargs):
    return {
        'url': url,
        **kwargs
    }


default_cfgs = {
    'yolox': _cfg(url='')
}


class YOLOX(nn.Cell):
    """ connect yolox backbone and head """

    def __init__(self, cfg, in_channels=3, num_classes=80, sync_bn=False):
        super(YOLOX, self).__init__()
        self.cfg = cfg
        self.stride = Tensor(np.array(cfg.stride), ms.int32)
        ch, nc = in_channels, num_classes
        self.nc = nc
        self.model = build_model_from_cfg(model_cfg=cfg, in_channels=ch, num_classes=nc, sync_bn=sync_bn)
        self.names = [str(i) for i in range(nc)]

        self.reset_parameter()

    def construct(self, x):
        return self.model(x)

    def reset_parameter(self):
        # init default
        initialize_defult(self)

        # reset parameter for Detect Head
        m = self.model.model[-1]
        assert isinstance(m, YOLOXHead)
        m.initialize_biases()


@register_model
def yolox(cfg, in_channels=3, num_classes=None, **kwargs) -> YOLOX:
    """Get GoogLeNet model.
     Refer to the base class `models.GoogLeNet` for more details."""
    model = YOLOX(cfg, in_channels=in_channels, num_classes=num_classes, **kwargs)
    return model