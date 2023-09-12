import numpy as np

import mindspore as ms
from mindspore import Tensor, nn

from mindyolo.models.registry import register_model
from mindyolo.models.heads import YOLOXHead
from mindyolo.models.model_factory import build_model_from_cfg

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


if __name__ == "__main__":
    from mindyolo.models.model_factory import create_model
    from mindyolo.utils.config import load_config, Config

    cfg, _, _ = load_config('../../configs/yolox/yolox-s.yaml')
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
