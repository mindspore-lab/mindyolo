import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor, nn, ops

from mindyolo.models.registry import register_model

from .iou_loss import bbox_iou

CLIP_VALUE = 1000.0
EPS = 1e-7

__all__ = ["YOLOv8Loss"]


@register_model
class YOLOv8Loss(nn.Cell):
    def __init__(self, box, cls, dfl, stride, nc, reg_max=16, use_fused_op=False, **kwargs):
        super(YOLOv8Loss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp_box = box
        self.hyp_cls = cls
        self.hyp_dfl = dfl
        self.stride = stride  # model strides
        self.nc = nc  # number of classes
        self.no = nc + reg_max * 4
        self.reg_max = reg_max

        self.use_dfl = reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0, use_fused_op=use_fused_op)
        self.bbox_loss = BboxLoss(reg_max, use_dfl=self.use_dfl, use_fused_op=use_fused_op)
        self.proj = mnp.arange(reg_max)

        # ops
        self.sigmoid = ops.Sigmoid()

        # branch name returned by lossitem for print
        self.loss_item_name = ["loss", "lbox", "lcls", "dfl"]

    def construct(self, feats, targets, imgs):
        """YOLOv8 Loss
        Args:
            feats: list of tensor, feats[i] shape: (bs, nc+reg_max*4, hi, wi)
            targets: [image_idx,cls,x,y,w,h], shape: (bs, gt_max, 6)
        """
        loss = ops.zeros(3, ms.float32)  # box, cls, dfl
        batch_size = feats[0].shape[0]
        _x = ()
        for xi in feats:
            _x += (xi.view(batch_size, self.no, -1),)
        _x = ops.concat(_x, 2)
        pred_distri, pred_scores = _x[:, : self.reg_max * 4, :], _x[:, -self.nc :, :]  # (bs, nc, h*w)
        pred_distri, pred_scores = pred_distri.transpose((0, 2, 1)), pred_scores.transpose((0, 2, 1))

        dtype = pred_scores.dtype
        imgsz = get_tensor(feats[0].shape[2:], dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = self.make_anchors(feats, self.stride, 0.5)

        # targets
        targets, mask_gt = self.preprocess(targets, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets[:, :, :1], targets[:, :, 1:5]  # cls, xyxy

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, shape: (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            self.sigmoid(pred_scores),
            (pred_bboxes * stride_tensor).astype(gt_bboxes.dtype),
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

        target_bboxes /= stride_tensor

        target_scores_sum = ops.maximum(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, ops.cast(target_scores, dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        # if fg_mask.sum():
        loss[0], loss[2] = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
        )

        loss[0] *= self.hyp_box  # box gain
        loss[1] *= self.hyp_cls  # cls gain
        loss[2] *= self.hyp_dfl  # dfl gain

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


class BboxLoss(nn.Cell):
    def __init__(self, reg_max, use_dfl=False, use_fused_op=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.use_fused_op = use_fused_op

    def construct(
        self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
    ):
        """
        Args:
            pred_dist: (bs, N, reg_max * 4)
            pred_bboxes: (bs, N, 4)
            anchor_points: (N, 2)
            target_bboxes: (bs, N, 4)
            target_scores: (bs, N, num_classes)
            target_scores_sum: (1,)
            fg_mask: (bs, N)
        """
        # IoU loss
        weight = target_scores.sum(-1).expand_dims(-1)  # (bs, N, num_classes) -> (bs, N) -> (bs, N, 1)
        iou = bbox_iou(pred_bboxes, target_bboxes, xywh=False, CIoU=True, use_fused_op=self.use_fused_op)
        loss_iou = ((1.0 - iou) * weight * fg_mask.expand_dims(2)).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = self.bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
            loss_dfl = self._df_loss(pred_dist.view(-1, self.reg_max), target_ltrb) * weight * fg_mask[:, :, None]
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = ops.zeros(1, ms.float32)

        return loss_iou, loss_dfl

    @staticmethod
    def bbox2dist(anchor_points, bbox, reg_max):
        """Transform bbox(xyxy) to dist(ltrb)."""
        x1y1, x2y2 = ops.split(bbox, split_size_or_sections=2, axis=-1)
        return ops.concat((anchor_points - x1y1, x2y2 - anchor_points), -1).clip(0, reg_max - 0.01)  # dist (lt, rb)

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        """
        Args:
            pred_dist: (bs*N*4, reg_max)
            target: (bs, N, 4)
            fg_mask: (bs, N)
        Return:
            loss: (bs, N, 1)
        """
        tl = ops.cast(target, ms.int32)  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right

        loss = (
            ops.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + ops.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keep_dims=True)

        return loss


class TaskAlignedAssigner(nn.Cell):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9, use_fused_op=False):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.use_fused_op=use_fused_op

    def construct(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """This code referenced to
               https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores: (bs, N, num_classes)
            pd_bboxes: (bs, N, 4)
            anc_points: (N, 2)
            gt_labels: (bs, n_gt, 1)
            gt_bboxes: (bs, n_gt, 4)
            mask_gt: (bs, n_gt)
        Returns:
            target_labels: (bs, N)
            target_bboxes: (bs, N, 4)
            target_scores: (bs, N, num_classes)
            fg_mask: (bs, N)
            target_gt_idx: (bs, N)
        """
        bs, n_gt, _ = gt_labels.shape
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, n_gt)

        # assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.max(axis=-1, keepdims=True)  # (b, n_gt)
        pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdims=True)  # (b, n_gt)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).max(-2).expand_dims(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, ops.cast(fg_mask, ms.bool_), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)  # (b, n_gt, N)
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)  # (b, n_gt, N)
        mask_topk = self.select_topk_candidates(
            align_metric * mask_in_gts, topk_mask=ops.cast(ops.tile(mask_gt[..., None], (1, 1, self.topk)), ms.bool_)
        )  # (b, n_gt, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt[:, :, None]  # (b, n_gt, N)

        return mask_pos, align_metric, overlaps

    def select_topk_candidates(self, metrics, topk_mask=None):
        """
        Args:
            metrics: (b, n_gt, N).
            topk_mask: (b, n_gt, topk) or None
        Returns:
            mask: (b, n_gt, N)
        """

        num_anchors = metrics.shape[-1]  # N
        topk_metrics, topk_idxs = ops.top_k(metrics, self.topk)  # (b, n_gt, topk)
        if topk_mask is None:
            topk_mask = ops.tile(topk_metrics.max(-1, keepdims=True) > self.eps, (1, 1, self.topk))  # (b, n_gt, topk)
        topk_idxs = mnp.where(topk_mask, topk_idxs, ops.zeros_like(topk_idxs))  # (b, n_gt, topk)
        is_in_topk = ops.one_hot(topk_idxs, num_anchors, ops.ones(1, ms.float32), ops.zeros(1, ms.float32)).sum(
            -2
        )  # (b, n_gt, topk, N) -> (b, n_gt, N)
        # filter invalid bboxes
        is_in_topk = mnp.where(is_in_topk > 1, ops.zeros(1, ms.float32), is_in_topk)
        is_in_topk = ops.cast(is_in_topk, metrics.dtype)

        return is_in_topk

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        bs, n_gt, _ = gt_labels.shape

        ind0 = ops.tile(mnp.arange(bs, dtype=ms.int32).view(-1, 1), (1, n_gt)).view(-1, 1)  # (b*n_gt, 1)
        ind1 = ops.cast(gt_labels, ms.int32).squeeze(-1).view(-1, 1)  # (b*n_gt, 1)
        bbox_scores = ops.gather_nd(
            pd_scores.transpose((0, 2, 1)), ops.concat((ind0, ind1), axis=1)
        )  # (b, N, 80)->(b, 80, N)->(b*n_gt, N)
        bbox_scores = bbox_scores.view(bs, n_gt, -1)

        # (b, n_gt, 1, 4), (b, 1, N, 4) -> (b, n_gt, N)
        overlaps = (
            bbox_iou(gt_bboxes.expand_dims(2), pd_bboxes.expand_dims(1), xywh=False, CIoU=True, use_fused_op=self.use_fused_op).squeeze(3).clip(0, None)
        )
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Args:
            gt_labels: (b, n_gt, 1)
            gt_bboxes: (b, n_gt, 4)
            target_gt_idx: (b, N)
            fg_mask: (b, N)
        """

        # assigned target labels
        bs, n_gt, _ = gt_labels.shape
        batch_ind = mnp.arange(bs)[:, None]  # (b, 1)
        target_gt_idx = target_gt_idx + batch_ind * n_gt  # (b, N)
        target_labels = ops.cast(gt_labels, ms.int32).flatten()[target_gt_idx]  # (b, N)

        # assigned target boxes
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]  # (b, n_gt, 4) -> (b * n_gt, 4) -> (b, N)

        # assigned target scores
        target_labels.clip(0, None)
        target_scores = ops.one_hot(
            target_labels, self.num_classes, on_value=ops.ones(1, ms.int32), off_value=ops.zeros(1, ms.int32)
        )  # (b, N, 80)
        fg_scores_mask = ops.tile(fg_mask[:, :, None], (1, 1, self.num_classes))  # (b, N) -> (b, N, 80)
        target_scores = mnp.where(fg_scores_mask > 0, target_scores, ops.zeros(1, ms.int32))

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, mask_gt=None, eps=1e-9):
        """select the positive anchor center in gt

        Args:
            xy_centers: (N, 2)
            gt_bboxes: (bs, n_gt, 4)
            mask_gt: (bs, n_gt) or None
        Return:
            select: shape(bs, n_gt, N)
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        x, y = ops.split(xy_centers.view(1, -1, 2), split_size_or_sections=1, axis=-1)  # (1, N, 2) -> (1, N, 1)
        left, top, right, bottom = ops.split(
            gt_bboxes.view(-1, 1, 4), split_size_or_sections=1, axis=-1
        )  # (bs, n_gt, 4)->(bs*n_gt, 1, 4)->(bs*n_gt, 1, 1)
        select = ops.logical_and(
            ops.logical_and((x - left) > eps, (y - top) > eps), ops.logical_and((right - x) > eps, (bottom - y) > eps)
        ).view(
            bs, n_boxes, n_anchors
        )  # (bs, n_gt, N)

        if mask_gt is not None:
            select = ops.cast(select, ms.float32) * ops.cast(mask_gt[..., None], ms.float32)

        return select

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_gt):
        """if an anchor box is assigned to multiple gts,
            the one with the highest iou will be selected.

        Args:
            mask_pos: (b, n_gt, N)
            overlaps: (b, n_gt, N)
        Return:
            target_gt_idx: (b, N)
            fg_mask: (b, N)
            mask_pos: (b, n_gt, N)
        """

        fg_mask = mask_pos.sum(-2)  # (b, n_gt, N) -> (b, N)

        # if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        mask_multi_gts = ops.tile(ops.expand_dims(fg_mask, 1), (1, n_gt, 1))  # (b, n_gt, N)
        max_overlaps_idx = overlaps.argmax(1)  # (b, n_gt, N) -> (b, N)
        is_max_overlaps = ops.one_hot(
            max_overlaps_idx, n_gt, on_value=ops.ones(1, ms.int32), off_value=ops.zeros(1, ms.int32)
        )  # (b, N, n_gt)
        is_max_overlaps = ops.cast(
            ops.transpose(is_max_overlaps, (0, 2, 1)), overlaps.dtype
        )  # (b, N, n_gt) -> (b, n_gt, N)
        mask_pos = mnp.where(mask_multi_gts > 0, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(-2)

        # find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = ops.Identity()(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


@ops.constexpr
def get_tensor(x, dtype=ms.float32):
    return Tensor(x, dtype)
