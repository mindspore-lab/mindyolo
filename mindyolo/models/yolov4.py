import mindspore.nn as nn
from .initializer import initialize_defult
from .registry import register_model
from .model_factory import build_model_from_cfg

__all__ = [
    'YOLOv4',
    'yolov4'
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        **kwargs
    }


default_cfgs = {
    'yolov4': _cfg(url='')
}


class YOLOv4(nn.Cell):
    def __init__(self, cfg, in_channels=3, num_classes=None, sync_bn=False):
        super(YOLOv4, self).__init__()
        self.cfg = cfg
        ch, nc = in_channels, num_classes

        self.nc = nc  # override yaml value
        self.model = build_model_from_cfg(model_cfg=cfg, in_channels=ch, num_classes=nc, sync_bn=sync_bn)
        self.names = [str(i) for i in range(nc)]  # default names

        self.reset_parameter()

    def construct(self, x):
        return self.model(x)

    def reset_parameter(self):
        # init default
        initialize_defult(self)


@register_model
def yolov4(cfg, in_channels=3, num_classes=None, **kwargs) -> YOLOv4:
    """Get GoogLeNet model.
     Refer to the base class `models.GoogLeNet` for more details."""
    model = YOLOv4(cfg=cfg, in_channels=in_channels, num_classes=num_classes, **kwargs)
    return model
