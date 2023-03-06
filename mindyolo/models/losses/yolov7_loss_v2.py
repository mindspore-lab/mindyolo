import numpy as np

import mindspore as ms
from mindspore import ops, nn, Tensor

from mindyolo.models.registry import register_model
from .focal_loss import FocalLoss
from .bce_loss import BCEWithLogitsLoss
from .iou_loss import bbox_iou
from .label_assignment import YOLOv7LabelAssignment


__all__ = [
    'YOLOv7LossV2',
    'YOLOv7AuxLossV2'
]


def smooth_BCE(eps=0.1):
    # return positive, negative label smoothing BCE targets,
    # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    return 1.0 - 0.5 * eps, 0.5 * eps


@register_model
class YOLOv7LossV2(nn.Cell):
    def __init__(
            self,
            box, obj, cls, anchor_t, label_smoothing, fl_gamma, cls_pw, obj_pw,
            anchors, stride, nc, **kwargs
    ):
        super(YOLOv7LossV2, self).__init__()
        self.hyp_box = box
        self.hyp_obj = obj
        self.hyp_cls = cls
        self.hyp_anchor_t = anchor_t
        self.nc = nc  # number of classes
        self.na = len(anchors[0]) // 2  # number of anchors
        self.nl = len(anchors)  # number of layers

        stride = np.array(stride)
        anchors = np.array(anchors).reshape((self.nl, -1, 2))
        anchors = anchors / stride.reshape((-1, 1, 1))
        self.stride = Tensor(stride, ms.int32)
        self.anchors = Tensor(anchors, ms.float32)  # shape(nl,na,2)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=label_smoothing)  # positive, negative BCE targets
        # Focal loss
        g = fl_gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(bce_pos_weight=Tensor([cls_pw], ms.float32), gamma=g), \
                             FocalLoss(bce_pos_weight=Tensor([obj_pw], ms.float32), gamma=g)
        else:
            # Define criteria
            BCEcls = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([cls_pw]), ms.float32))
            BCEobj = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([obj_pw]), ms.float32))

        _balance = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.balance = ms.Parameter(Tensor(_balance, ms.float32), requires_grad=False)
        self.BCEcls, self.BCEobj, self.gr = BCEcls, BCEobj, 1.0
        self.build_targets = YOLOv7LabelAssignment(self.anchors, na=self.na, bias=0.5, stride=self.stride)
        self.one_hot = nn.OneHot(depth=self.nc, on_value=self.cp, off_value=self.cn)
        self.autobalance = False

    def construct(self, p, targets, imgs):
        lcls, lbox, lobj = 0., 0., 0.
        bs, as_, gjs, gis, targets, anchors, masks = self.build_targets(p, targets, imgs)  # bs: (nl, bs*5*na*gt_max)

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi, target, mask = bs[i], as_[i], gjs[i], gis[i], targets[i], masks[i]
            tobj = ops.zeros_like(pi[..., 0])  # target obj
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            # Regression
            pxy = ops.Sigmoid()(ps[:, :2]) * 2. - 0.5
            pwh = (ops.Sigmoid()(ps[:, 2:4]) * 2) ** 2 * anchors[i]
            pbox = ops.concat((pxy, pwh), 1) * mask  # predicted box
            iou = bbox_iou(pbox, target[:, 2:6], xywh=True, CIoU=True)  # iou(prediction, target)
            lbox += ((1.0 - iou) * mask).sum() / mask.sum()  # iou loss

            # Objectness
            tobj[b, a, gj, gi] = (1.0 - self.gr) + \
                                 (self.gr * ops.stop_gradient(iou).clip(0, None) * mask).reshape((-1,))

            # Classification
            if self.nc > 1:  # cls loss (only if multiple classes)
                t = self.one_hot(ops.cast(target[:, 1], ms.int32))
                lcls += self.BCEcls(ps[:, 5:], t, mask)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / ops.stop_gradient(obji)

        if self.autobalance:
            _balance_ssi = self.balance[self.ssi]
            self.balance /= _balance_ssi
        lbox *= self.hyp_box
        lobj *= self.hyp_obj
        lcls *= self.hyp_cls

        loss = lbox + lobj + lcls
        return loss * p[0].shape[0], ops.stop_gradient(ops.stack((loss, lbox, lobj, lcls)))


@register_model
class YOLOv7AuxLossV2(nn.Cell):
    def __init__(
            self,
            box, obj, cls, anchor_t, label_smoothing, fl_gamma, cls_pw, obj_pw,
            anchors, stride, nc, **kwargs
    ):
        super(YOLOv7AuxLossV2, self).__init__()
        self.hyp_box = box
        self.hyp_obj = obj
        self.hyp_cls = cls
        self.hyp_anchor_t = anchor_t
        self.nc = nc  # number of classes
        self.na = len(anchors[0]) // 2  # number of anchors
        self.nl = len(anchors)  # number of layers

        stride = np.array(stride)
        anchors = np.array(anchors).reshape((self.nl, -1, 2))
        anchors = anchors / stride.reshape((-1, 1, 1))
        self.stride = Tensor(stride, ms.int32)
        self.anchors = Tensor(anchors, ms.float32)  # shape(nl,na,2)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=label_smoothing)  # positive, negative BCE targets
        # Focal loss
        g = fl_gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(bce_pos_weight=Tensor([cls_pw], ms.float32), gamma=g), \
                             FocalLoss(bce_pos_weight=Tensor([obj_pw], ms.float32), gamma=g)
        else:
            # Define criteria
            BCEcls = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([cls_pw]), ms.float32))
            BCEobj = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([obj_pw]), ms.float32))

        _balance = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.balance = ms.Parameter(Tensor(_balance, ms.float32), requires_grad=False)
        self.BCEcls, self.BCEobj, self.gr = BCEcls, BCEobj, 1.0
        self.build_targets = YOLOv7LabelAssignment(self.anchors, na=self.na, bias=0.5, stride=self.stride, use_aux=True)
        self.one_hot = nn.OneHot(depth=self.nc, on_value=self.cp, off_value=self.cn)
        self.autobalance = False

    def construct(self, p, targets, imgs):
        lcls, lbox, lobj = 0., 0., 0.
        # Losses
        bs, as_, gjs, gis, targets, anchors, masks, \
        bs_aux, as_aux, gjs_aux, gis_aux, targets_aux, anchors_aux, masks_aux = \
            self.build_targets(p[:self.nl], targets, imgs)
        for i in range(self.nl):  # layer index, layer predictions
            b, a, gj, gi, target, anchor, mask = bs[i], as_[i], gjs[i], gis[i], targets[i], anchors[i], masks[i]
            b_aux, a_aux, gj_aux, gi_aux, target_aux, anchor_aux, mask_aux = \
                bs_aux[i], as_aux[i], gjs_aux[i], gis_aux[i], targets_aux[i], anchors_aux[i], masks_aux[i]
            pi = p[i]
            pi_aux = p[i + self.nl]
            tobj = ops.zeros_like(pi[..., 0])  # target obj
            tobj_aux = ops.zeros_like(pi_aux[..., 0])
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ops.Sigmoid()(ps[:, :2]) * 2. - 0.5
            pwh = (ops.Sigmoid()(ps[:, 2:4]) * 2) ** 2 * anchor
            pbox = ops.concat((pxy, pwh), 1) * mask  # predicted box
            iou = bbox_iou(pbox, target[:, 2:6], xywh=True, CIoU=True)  # iou(prediction, target)
            lbox += ((1.0 - iou) * mask).sum() / mask.sum()  # iou loss

            # Objectness
            tobj[b, a, gj, gi] = (1.0 - self.gr) + \
                                 (self.gr * ops.stop_gradient(iou).clip(0, None) * mask).reshape((-1,))

            # Classification
            if self.nc > 1:  # cls loss (only if multiple classes)
                t = self.one_hot(ops.cast(target[:, 1], ms.int32))
                lcls += self.BCEcls(ps[:, 5:], t, mask)  # BCE
            n_aux = b_aux.shape[0]  # number of targets
            if n_aux:
                ps_aux = pi_aux[b_aux, a_aux, gj_aux, gi_aux]  # prediction subset corresponding to targets
                pxy_aux = ops.Sigmoid()(ps_aux[:, :2]) * 2. - 0.5
                pwh_aux = (ops.Sigmoid()(ps_aux[:, 2:4]) * 2) ** 2 * anchor_aux
                pbox_aux = ops.concat((pxy_aux, pwh_aux), 1) * mask_aux  # predicted box
                iou_aux = bbox_iou(pbox_aux, target_aux[:, 2:6], xywh=True, CIoU=True)  # iou(prediction, target)
                lbox += 0.25 * (1.0 - iou_aux).mean()  # iou loss

                # Objectness
                tobj_aux[b_aux, a_aux, gj_aux, gi_aux] = \
                    (1.0 - self.gr) + (self.gr * ops.stop_gradient(iou_aux).clip(0, None) * mask_aux).reshape((-1,))

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = self.one_hot(ops.cast(target_aux[:, 1], ms.int32))
                    lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t, mask_aux)  # BCE
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            obji_aux = self.BCEobj(pi_aux[..., 4], tobj_aux)
            lobj += obji * self.balance[i] + 0.25 * obji_aux * self.balance[i]  # obj loss

            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / ops.stop_gradient(obji)

        if self.autobalance:
            _balance_ssi = self.balance[self.ssi]
            self.balance /= _balance_ssi
        lbox *= self.hyp_box
        lobj *= self.hyp_obj
        lcls *= self.hyp_cls

        loss = lbox + lobj + lcls
        return loss * p[0].shape[0], ops.stop_gradient(ops.stack((loss, lbox, lobj, lcls)))
