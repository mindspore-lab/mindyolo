import numpy as np

import mindspore as ms
from mindspore import Tensor, nn

from mindyolo.models.model_factory import build_model_from_cfg
from mindyolo.models.registry import register_model

__all__ = ["YOLOv4", "yolov4"]


def _cfg(url="", **kwargs):
    return {"url": url, **kwargs}


default_cfgs = {"yolov4": _cfg(url="")}


class YOLOv4(nn.Cell):
    def __init__(self, cfg, in_channels=3, num_classes=None, sync_bn=False):
        super(YOLOv4, self).__init__()
        self.cfg = cfg
        ch, nc = in_channels, num_classes

        self.nc = nc  # override yaml value
        self.model = build_model_from_cfg(model_cfg=cfg, in_channels=ch, num_classes=nc, sync_bn=sync_bn)
        self.names = [str(i) for i in range(nc)]  # default names

    def construct(self, x):
        return self.model(x)


@register_model
def yolov4(cfg, in_channels=3, num_classes=None, **kwargs) -> YOLOv4:
    """Get yolov4 model."""
    model = YOLOv4(cfg=cfg, in_channels=in_channels, num_classes=num_classes, **kwargs)
    return model


if __name__ == "__main__":
    from mindyolo.models.model_factory import create_model
    from mindyolo.utils.config import load_config, Config

    cfg, _, _ = load_config('../../configs/yolov4/yolov4.yaml')
    cfg = Config(cfg)
    network = create_model(
        model_name=cfg.network.model_name,
        model_cfg=cfg.network,
        num_classes=cfg.data.nc,
        sync_bn=cfg.sync_bn if hasattr(cfg, "sync_bn") else False,
    )
    x = Tensor(np.random.randn(1, 3, 640, 640), ms.float32)
    out = network(x)
    out = out[0] if isinstance(out, (list, tuple)) else out
    print(f"Output shape is {[o.shape for o in out]}")
