import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor, nn, ops

from mindyolo.models.registry import register_model

from .iou_loss import bbox_iou
from .yolov8_loss import TaskAlignedAssigner, BboxLoss

CLIP_VALUE = 1000.0
EPS = 1e-7

__all__ = ["YOLOv9Loss"]


@register_model
class YOLOv9Loss(nn.Cell):
    def __init__(self, box, cls, dfl, stride, nc, reg_max=16, **kwargs):
        super(YOLOv9Loss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp_box = box
        self.hyp_cls = cls
        self.hyp_dfl = dfl
        self.stride = stride  # model strides
        self.nc = nc  # number of classes
        self.no = nc + reg_max * 4
        self.reg_max = reg_max

        self.use_dfl = reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.assigner2 = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(reg_max, use_dfl=self.use_dfl)
        self.bbox_loss2 = BboxLoss(reg_max, use_dfl=self.use_dfl)
        self.proj = mnp.arange(reg_max)

        # ops
        self.sigmoid = ops.Sigmoid()

        # branch name returned by lossitem for print
        self.loss_item_name = ["loss", "lbox", "lcls", "dfl"]

    def construct(self, p, targets, imgs):
        """YOLOv9 Loss
        Args:
            targets: [image_idx,cls,x,y,w,h], shape: (bs, gt_max, 6)
        """
        loss = ops.zeros(3, ms.float32)  # box, cls, dfl
        feats, feats2 = p[0], p[1]
        batch_size = feats[0].shape[0]

        _x = ()
        for xi in feats:
            _x += (xi.view(batch_size, self.no, -1),)
        _x = ops.concat(_x, 2)
        pred_distri, pred_scores = _x[:, : self.reg_max * 4, :], _x[:, -self.nc:, :]  # (bs, nc, h*w)
        pred_distri, pred_scores = pred_distri.transpose((0, 2, 1)), pred_scores.transpose((0, 2, 1))

        _x2 = ()
        for xi2 in feats2:
            _x2 += (xi2.view(batch_size, self.no, -1),)
        _x2 = ops.concat(_x2, 2)
        pred_distri2, pred_scores2 = _x2[:, : self.reg_max * 4, :], _x2[:, -self.nc:, :]  # (bs, nc, h*w)
        pred_distri2, pred_scores2 = pred_distri2.transpose((0, 2, 1)), pred_scores2.transpose((0, 2, 1))

        dtype = pred_scores.dtype
        imgsz = get_tensor(feats[0].shape[2:], dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = self.make_anchors(feats, self.stride, 0.5)

        # targets
        targets, mask_gt = self.preprocess(targets, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets[:, :, :1], targets[:, :, 1:5]  # cls, xyxy

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, shape: (b, h*w, 4)
        pred_bboxes2 = self.bbox_decode(anchor_points, pred_distri2)  # xyxy, shape: (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            self.sigmoid(pred_scores),
            (pred_bboxes * stride_tensor).astype(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        _, target_bboxes2, target_scores2, fg_mask2, _ = self.assigner2(
            self.sigmoid(pred_scores2),
            (pred_bboxes2 * stride_tensor).astype(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        # stop gradient
        target_bboxes, target_scores, fg_mask = (
            ops.stop_gradient(target_bboxes),
            ops.stop_gradient(target_scores),
            ops.stop_gradient(fg_mask),
        )
        target_bboxes2, target_scores2, fg_mask2 = (
            ops.stop_gradient(target_bboxes2),
            ops.stop_gradient(target_scores2),
            ops.stop_gradient(fg_mask2),
        )

        target_bboxes /= stride_tensor
        target_scores_sum = ops.maximum(target_scores.sum(), 1)
        target_bboxes2 /= stride_tensor
        target_scores_sum2 = ops.maximum(target_scores2.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, ops.cast(target_scores, dtype)).sum() / target_scores_sum  # BCE
        loss[1] *= 0.25
        loss[1] += self.bce(pred_scores2, ops.cast(target_scores2, dtype)).sum() / target_scores_sum2  # BCE

        # bbox loss
        # if fg_mask.sum():
        loss[0], loss[2] = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
        )
        loss[0] *= 0.25
        loss[2] *= 0.25
        # if fg_mask2.sum():
        loss0_, loss2_ = self.bbox_loss(
            pred_distri2, pred_bboxes2, anchor_points, target_bboxes2, target_scores2, target_scores_sum2, fg_mask2
        )
        loss[0] += loss0_
        loss[2] += loss2_

        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain

        return loss.sum() * batch_size, ops.stop_gradient(
            ops.concat((loss.sum(keepdims=True), loss))
        )  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4)
            # pred_dist = ops.softmax(pred_dist, axis=3) # ms version >= 1.9.0
            pred_dist = ops.Softmax(axis=3)(pred_dist)  # ms version <= 1.8.1
            # (batch, anchors, 4, reg_max) @ (reg_max,) -> (batch, anchors, 4)
            _dtype = pred_dist.dtype
            pred_dist = ops.matmul(pred_dist.astype(ms.float16), self.proj.astype(ms.float16)).astype(_dtype)
        return self.dist2bbox(pred_dist, anchor_points, xywh=False)

    def preprocess(self, targets, scale_tensor):
        """preprocess gt boxes

        Args:
            targets: [image_idx,cls,x,y,w,h], shape: (bs, gt_max, 6)
            scale_tensor: (4,)
        Return:
            out: [cls,x,y,x,y], shape: (bs, gt_max, 5)
            mask_gt: (bs, gt_max)
        """
        mask_gt = targets[:, :, 1] >= 0  # (bs, gt_max)
        out = targets[:, :, 1:] * mask_gt[:, :, None]  # [cls,x,y,w,h], shape: (bs, gt_max, 5)
        out[..., 1:5] = xywh2xyxy(out[..., 1:5] * scale_tensor)
        return out, mask_gt

    @staticmethod
    def dist2bbox(distance, anchor_points, xywh=True, axis=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        lt, rb = ops.split(distance, split_size_or_sections=2, axis=axis)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return ops.concat((c_xy, wh), axis)  # xywh bbox
        return ops.concat((x1y1, x2y2), axis)  # xyxy bbox

    @staticmethod
    def make_anchors(feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = (), ()
        dtype = feats[0].dtype
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = mnp.arange(w, dtype=dtype) + grid_cell_offset  # shift x
            sy = mnp.arange(h, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = ops.meshgrid(sy, sx, indexing="ij")
            anchor_points += (ops.stack((sx, sy), -1).view(-1, 2),)
            stride_tensor += (ops.ones((h * w, 1), dtype) * stride,)
        return ops.concat(anchor_points), ops.concat(stride_tensor)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = ops.Identity()(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format.

    Args:
        x (Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
       y (Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    y = ops.Identity()(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


@ops.constexpr
def get_tensor(x, dtype=ms.float32):
    return Tensor(x, dtype)
