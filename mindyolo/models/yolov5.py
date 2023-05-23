import numpy as np

import mindspore as ms
from mindspore import Tensor, nn

from .heads.yolov7_head import YOLOv7AuxHead, YOLOv7Head
from .initializer import initialize_defult
from .model_factory import build_model_from_cfg
from .registry import register_model

__all__ = ["YOLOv5", "yolov5"]


def _cfg(url="", **kwargs):
    return {"url": url, **kwargs}


default_cfgs = {"yolov5": _cfg(url="")}


class YOLOv5(nn.Cell):
    def __init__(self, cfg, in_channels=3, num_classes=None, sync_bn=False):
        super(YOLOv5, self).__init__()
        self.cfg = cfg
        self.stride = Tensor(np.array(cfg.stride), ms.int32)
        self.stride_max = int(max(self.cfg.stride))
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

        # reset parameter for Detect Head
        m = self.model.model[-1]
        if isinstance(m, YOLOv7Head):
            m.initialize_biases()
        if isinstance(m, YOLOv7AuxHead):
            m.initialize_aux_biases()


@register_model
def yolov5(cfg, in_channels=3, num_classes=None, **kwargs) -> YOLOv5:
    """Get GoogLeNet model.
    Refer to the base class `models.GoogLeNet` for more details."""
    model = YOLOv5(cfg=cfg, in_channels=in_channels, num_classes=num_classes, **kwargs)
    return model


# TODO: Preset pre-training model for yolov5-n/s/m


if __name__ == "__main__":
    from mindyolo.models.model_factory import create_model
    from mindyolo.utils.config import parse_config

    opt = parse_config()
    model = create_model(
        model_name="yolov5",
        model_cfg=opt.net,
        num_classes=opt.data.nc,
        sync_bn=opt.sync_bn if hasattr(opt, "sync_bn") else False,
    )
    x = Tensor(np.random.randn(1, 3, 640, 640), ms.float32)
    out = model(x)
    out = out[0] if isinstance(out, (list, tuple)) else out
    print(f"Output shape is {[o.shape for o in out]}")
