import numpy as np

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor, nn, ops

from mindyolo.models.registry import register_model
from .focal_loss import BCEWithLogitsLoss, smooth_BCE
from .iou_loss import bbox_iou

CLIP_VALUE = 1000.0
EPS = 1e-7

__all__ = ["YOLOv4Loss"]


class ConfidenceLoss(nn.Cell):
    """Loss for confidence."""

    def __init__(self):
        super(ConfidenceLoss, self).__init__()
        self.cross_entropy = ops.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, object_mask, predict_confidence, ignore_mask):
        confidence_loss = self.cross_entropy(predict_confidence, object_mask)
        confidence_loss = object_mask * confidence_loss + (1 - object_mask) * confidence_loss * ignore_mask
        confidence_loss = self.reduce_sum(confidence_loss, ())
        return confidence_loss


@register_model
class YOLOv4Loss(nn.Cell):
    def __init__(self, box, obj, cls, label_smoothing, ignore_threshold, iou_threshold, anchors, nc, **kwargs):
        super(YOLOv4Loss, self).__init__()
        self.ignore_threshold = ignore_threshold
        self.iou = Iou()
        self.iou_threshold = iou_threshold
        self.hyp_box = box
        self.hyp_obj = obj
        self.hyp_cls = cls
        self.nc = nc  # number of classes

        anchors = np.array(anchors)
        self.na = anchors.shape[0]  # number of anchors
        self.nl = 3  # number of layers

        self.anchors = Tensor(anchors, ms.float32)  # shape(na,2)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=label_smoothing)  # positive, negative BCE targets

        self.BCEobj = ConfidenceLoss()
        self.BCEcls = BCEWithLogitsLoss(reduction="sum")

        self.loss_item_name = ["loss", "lbox", "lobj", "lcls"]  # branch name returned by lossitem for print

        self.concat = ops.Concat(axis=-1)
        self.reduce_max = ops.ReduceMax(keep_dims=False)

    def construct(self, p, targets, imgs):
        image_shape = imgs.shape
        gain = get_tensor(image_shape, targets.dtype)[[3, 2]]
        ori_targets = targets.copy()
        lcls, lbox, lobj = 0.0, 0.0, 0.0
        tcls, tbox, indices, anchors, tmasks = self.build_targets(
            p, targets, imgs
        )  # class, box, (image, anchor, gridj, gridi), anchors, mask
        tcls, tbox, indices, anchors, tmasks = (
            ops.stop_gradient(tcls),
            ops.stop_gradient(tbox),
            ops.stop_gradient(indices),
            ops.stop_gradient(anchors),
            ops.stop_gradient(tmasks),
        )

        # Losses
        for layer_index, yolo_out in enumerate(p):  # layer index, layer predictions
            pi = yolo_out[0]
            tmask = tmasks[layer_index]
            b, a, gj, gi = ops.split(indices[layer_index] * tmask[None, :], split_size_or_sections=1, axis=0)  # image, anchor, gridy, gridx
            b, a, gj, gi = b.view(-1), a.view(-1), gj.view(-1), gi.view(-1)

            pi_shape = pi.shape
            y_true = ops.zeros((pi_shape[0], pi_shape[1], pi_shape[2], pi_shape[3], 1), pi.dtype)
            y_true[b, gj, gi, a][:, 0] = 1.0

            n = b.shape[0]  # number of targets
            if n:
                pxy = yolo_out[1][b, gj, gi, a]
                pwh = yolo_out[2][b, gj, gi, a]
                _meta_pred = pi[b, gj, gi, a]  # gather from (bs,na,h,w,nc)
                pcls = _meta_pred[:, 5:]

                # Regression
                pbox = ops.concat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox, GIoU=True).squeeze()  # iou(prediction, target)
                # iou = iou * tmask
                # lbox += ((1.0 - iou) * tmask).mean()  # iou loss
                box_loss_scale = 2 - tbox[:, 2] * tbox[:, 3] / gain[0] / gain[1]
                lbox += (((1.0 - iou) * tmask * box_loss_scale).sum()).astype(iou.dtype)

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = ops.fill(pcls.dtype, pcls.shape, self.cn)  # targets

                    t[mnp.arange(n), tcls] = self.cp
                    lcls += self.BCEcls(pcls, t, ops.tile(tmask[:, None], (1, t.shape[-1])))  # BCE

            gt_box = ori_targets[:, :, 2:]
            pred_boxes = self.concat((yolo_out[1], yolo_out[2]))
            gt_shape = ops.Shape()(gt_box)
            gt_box = ops.Reshape()(gt_box, (gt_shape[0], 1, 1, 1, gt_shape[1], gt_shape[2]))
            iou = self.iou(ops.ExpandDims()(pred_boxes, -2), gt_box)
            best_iou = self.reduce_max(iou, -1)
            ignore_mask = best_iou < self.ignore_threshold
            ignore_mask = ops.Cast()(ignore_mask, ms.float32)
            ignore_mask = ops.ExpandDims()(ignore_mask, -1)
            ignore_mask = ops.stop_gradient(ignore_mask)
            object_mask = y_true[:, :, :, :, 0:1]
            lobj += self.BCEobj(object_mask, pi[:, :, :, :, 4:5], ignore_mask)  # obj loss

        lbox *= self.hyp_box
        lobj *= self.hyp_obj
        lcls *= self.hyp_cls
        bs = p[0][0].shape[0]  # batch size

        loss = lbox + lobj + lcls

        # ops.stack doesn't support type ms.float16 under ascend ms2.0,
        # refer to issue #154 (https://github.com/mindspore-lab/mindyolo/issues/154)
        return loss / bs / 8, ops.stop_gradient(ops.stack(
            (loss.astype(ms.float32) / bs,
             lbox.astype(ms.float32) / bs,
             lobj.astype(ms.float32) / bs,
             lcls.astype(ms.float32) / bs)
        ))

    def build_targets(self, p, targets, imgs):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        image_shape = imgs.shape
        targets = targets.view(-1, 6)
        mask_t = targets[:, 1] >= 0
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch, tmasks = (), (), ()
        gain_wh = ops.ones(7, ms.int32)  # normalized to gridspace gain
        ai = ops.tile(mnp.arange(na).view(-1, 1), (1, nt))  # shape: (na, nt)
        ai = ops.cast(ai, targets.dtype)
        targets_9_anchors = ops.concat(
            (ops.tile(targets, (na, 1, 1)), ai[:, :, None]), 2
        )  # append anchor indices # shape: (na, nt, 7)

        gain_wh[4:6] = get_tensor(image_shape, targets_9_anchors.dtype)[[3, 2]]  # xyxy gain

        # Match targets to anchors
        t_wh = targets_9_anchors * gain_wh
        # Matches
        gt_box = ops.zeros((na, nt, 4), ms.float32)
        gt_box[..., 2:] = t_wh[..., 4:6]

        anchor_shapes = ops.zeros((na, 1, 4), ms.float32)
        anchor_shapes[..., 2:] = ops.ExpandDims()(self.anchors, 1)
        anch_ious = bbox_iou(gt_box, anchor_shapes).squeeze()

        j = anch_ious == anch_ious.max(axis=0)
        l = anch_ious > self.iou_threshold

        j_l = ops.logical_or(j, l).astype(ms.int32).reshape((self.nl, -1, nt))

        anchor_scales = self.anchors.reshape((self.nl, -1, 2))
        ai = ops.tile(mnp.arange(na // self.nl).view(-1, 1), (1, nt))  # shape: (na, nt)
        ai = ops.cast(ai, targets.dtype)
        targets_3_anchors = ops.concat((ops.tile(targets, (na // self.nl, 1, 1)), ai[:, :, None]), 2)
        for i in range(self.nl):
            anchors, shape = anchor_scales[i], p[i][0].shape
            gain_xy = ops.ones(7, ms.int32)  # normalized to gridspace gain
            gain_xy[2:4] = get_tensor(shape, targets_3_anchors.dtype)[[2, 1]]  # xyxy gain

            t = targets_3_anchors * gain_xy
            mask_m_t = (j_l[i] * ops.cast(mask_t[None, :], ms.int32)).view(-1)
            t = t.view(-1, 7)

            # Define
            b, gxy, a = (
                ops.cast(t[:, 0], ms.int32),
                t[:, 2:4],
                ops.cast(t[:, 6], ms.int32),
            )  # (image, class), grid xy, grid wh, anchors
            gij = ops.cast(gxy, ms.int32)
            gij = gij[:]
            gi, gj = gij[:, 0], gij[:, 1]  # grid indices
            gi = gi.clip(0, shape[2] - 1)
            gj = gj.clip(0, shape[1] - 1)

            # Append
            indices += (ops.stack((b, a, gj, gi), 0),)  # image, anchor, grid
            anch += (anchors[a],)  # anchors
            tmasks += (mask_m_t,)

        targets_3_anchors = targets_3_anchors.view(-1, 7)
        tcls = ops.cast(targets_3_anchors[:, 1], ms.int32)  # class
        tbox = targets_3_anchors[:, 2:6]  # box

        return (
            tcls,
            tbox,
            ops.stack(indices),
            ops.stack(anch),
            ops.stack(tmasks),
        )  # class, box, (image, anchor, gridj, gridi), anchors, mask


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = ops.Identity()(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


@ops.constexpr
def get_tensor(x, dtype=ms.float32):
    return Tensor(x, dtype)


class Iou(nn.Cell):
    """Calculate the iou of boxes"""

    def __init__(self):
        super(Iou, self).__init__()
        self.min = ops.Minimum()
        self.max = ops.Maximum()

    def construct(self, box1, box2):
        """
        box1: pred_box [batch, gx, gy, anchors, 1,      4] ->4: [x_center, y_center, w, h]
        box2: gt_box   [batch, 1,  1,  1,       maxbox, 4]
        convert to topLeft and rightDown
        """
        box1_xy = box1[:, :, :, :, :, :2]
        box1_wh = box1[:, :, :, :, :, 2:4]
        box1_mins = box1_xy - box1_wh / ops.scalar_to_tensor(2.0)  # topLeft
        box1_maxs = box1_xy + box1_wh / ops.scalar_to_tensor(2.0)  # rightDown

        box2_xy = box2[:, :, :, :, :, :2]
        box2_wh = box2[:, :, :, :, :, 2:4]
        box2_mins = box2_xy - box2_wh / ops.scalar_to_tensor(2.0)
        box2_maxs = box2_xy + box2_wh / ops.scalar_to_tensor(2.0)

        intersect_mins = self.max(box1_mins, box2_mins)
        intersect_maxs = self.min(box1_maxs, box2_maxs)
        intersect_wh = self.max(intersect_maxs - intersect_mins, ops.scalar_to_tensor(0.0))
        # P.squeeze: for effiecient slice
        intersect_area = ops.Squeeze(-1)(intersect_wh[:, :, :, :, :, 0:1]) * ops.Squeeze(-1)(
            intersect_wh[:, :, :, :, :, 1:2]
        )
        box1_area = ops.Squeeze(-1)(box1_wh[:, :, :, :, :, 0:1]) * ops.Squeeze(-1)(box1_wh[:, :, :, :, :, 1:2])
        box2_area = ops.Squeeze(-1)(box2_wh[:, :, :, :, :, 0:1]) * ops.Squeeze(-1)(box2_wh[:, :, :, :, :, 1:2])
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        # iou : [batch, gx, gy, anchors, maxboxes]
        return iou


if __name__ == "__main__":
    from mindyolo.models.losses.loss_factory import create_loss
    from mindyolo.utils.config import parse_config

    cfg = parse_config()
    loss_fn = create_loss(
        name="YOLOv7Loss",
        **cfg.loss,
        anchors=cfg.network.get("anchors", None),
        stride=cfg.network.get("stride", None),
        nc=cfg.data.get("nc", None),
    )
    print(f"loss_fn is {loss_fn}")
