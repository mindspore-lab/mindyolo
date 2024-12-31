from mindspore import nn

from mindyolo.models.registry import register_model

from .yolov8_loss import YOLOv8Loss

CLIP_VALUE = 1000.0
EPS = 1e-7

__all__ = ["YOLOv11Loss"]


@register_model
class YOLOv11Loss(nn.Cell):
    def __init__(self, box, cls, dfl, stride, nc, reg_max=16, tal_topk=10, **kwargs):
        super(YOLOv11Loss, self).__init__()

        self.loss = YOLOv8Loss(box, cls, dfl, stride, nc, reg_max=reg_max, tal_topk=tal_topk, **kwargs)

        # branch name returned by lossitem for print
        self.loss_item_name = ["loss", "lbox", "lcls", "dfl"]

    def construct(self, feats, targets, imgs):
        """YOLOv11 Loss
        Args:
            feats: list of tensor, feats[i] shape: (bs, nc+reg_max*4, hi, wi)
            targets: [image_idx,cls,x,y,w,h], shape: (bs, gt_max, 6)
        """
        return self.loss(feats, targets, imgs)
