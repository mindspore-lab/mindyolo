from mindspore import nn, ops

from .yolov8_loss import YOLOv8Loss
from mindyolo.models.registry import register_model

__all__ = ["YOLOv10Loss"]

@register_model
class YOLOv10Loss(nn.Cell):
    def __init__(self, box, cls, dfl, stride, nc, reg_max=16, tal_topk=10, **kwargs):
        super().__init__()
        self.one2many = YOLOv8Loss(box, cls, dfl, stride, nc, reg_max, tal_topk=10)
        self.one2one = YOLOv8Loss(box, cls, dfl, stride, nc, reg_max, tal_topk=1)
        # branch name returned by lossitem for print
        self.loss_item_name = ["loss", "lbox", "lcls", "dfl"]

    def construct(self, preds, batch, imgs):
        one2many = preds[0]
        loss_one2many = self.one2many(one2many, batch, imgs)
        one2one = preds[1]
        loss_one2one = self.one2one(one2one, batch, imgs)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]