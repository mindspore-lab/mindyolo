from mindspore import nn, ops

from .yolov8_loss import YOLOv8Loss
from mindyolo.models.registry import register_model

__all__ = ["YOLOv12Loss"]

@register_model
class YOLOv12Loss(nn.Cell):
    def __init__(self, box, cls, dfl, stride, nc, reg_max=16, tal_topk=10, **kwargs):
        super().__init__()
        self.loss = YOLOv8Loss(box, cls, dfl, stride, nc, reg_max=reg_max, tal_topk=tal_topk, **kwargs)
        # branch name returned by lossitem for print
        self.loss_item_name = ["loss", "lbox", "lcls", "dfl"]

    def construct(self, feats, targets, imgs):
        # YOLOV12 Loss
        return self.loss(feats, targets, imgs)