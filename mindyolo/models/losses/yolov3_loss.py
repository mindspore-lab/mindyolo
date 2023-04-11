import numpy as np

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops, nn, Tensor

from mindyolo.models.registry import register_model

from .focal_loss import FocalLoss, BCEWithLogitsLoss, smooth_BCE
from .iou_loss import bbox_iou, batch_box_iou

CLIP_VALUE = 1000.
EPS = 1e-7

__all__ = [
    'YOLOv3Loss',
]


@register_model
class YOLOv3Loss(nn.Cell):
    def __init__(
            self,
            box, obj, cls, anchor_t, label_smoothing, fl_gamma, cls_pw, obj_pw,
            anchors, stride, nc, **kwargs
    ):
        super(YOLOv3Loss, self).__init__()
        self.hyp_box = box
        self.hyp_obj = obj
        self.hyp_cls = cls
        self.hyp_anchor_t = anchor_t
        self.nc = nc                    # number of classes
        self.na = len(anchors[0]) // 2  # number of anchors
        self.nl = len(anchors)          # number of layers

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

        self._off = Tensor([
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],  # j,k,l,m
        ], dtype=ms.float32)

        self.loss_item_name = ['loss', 'lbox', 'lobj', 'lcls']  # branch name returned by lossitem for print

    def construct(self, p, targets, imgs):
        lcls, lbox, lobj = 0., 0., 0.
        tcls, tbox, indices, anchors, tmasks = self.build_targets(p, targets)  # class, box, (image, anchor, gridj, gridi), anchors, mask
        tcls, tbox, indices, anchors, tmasks = ops.stop_gradient(tcls), ops.stop_gradient(tbox), \
                                               ops.stop_gradient(indices), ops.stop_gradient(anchors), \
                                               ops.stop_gradient(tmasks)

        # Losses
        for layer_index, pi in enumerate(p):  # layer index, layer predictions
            tmask = tmasks[layer_index]
            b, a, gj, gi = ops.split(indices[layer_index] * tmask[None, :], 0, 4)  # image, anchor, gridy, gridx
            b, a, gj, gi = b.view(-1), a.view(-1), gj.view(-1), gi.view(-1)
            tobj = ops.zeros(pi.shape[:4], pi.dtype)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                _meta_pred = pi[b, a, gj, gi]  # gather from (bs,na,h,w,nc)
                pxy, pwh, _, pcls = _meta_pred[:, :2], _meta_pred[:, 2:4], _meta_pred[:, 4:5], _meta_pred[:, 5:]

                # Regression
                pxy = ops.Sigmoid()(pxy) * 2 - 0.5
                pwh = (ops.Sigmoid()(pwh) * 2) ** 2 * anchors[layer_index]
                pbox = ops.concat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[layer_index], CIoU=True).squeeze()  # iou(prediction, target)
                # iou = iou * tmask
                # lbox += ((1.0 - iou) * tmask).mean()  # iou loss
                lbox += (((1.0 - iou) * tmask).sum() / tmask.astype(iou.dtype).sum().clip(1, None)).astype(iou.dtype)

                # Objectness
                iou = ops.stop_gradient(iou).clip(0, None).astype(pi.dtype)
                # tobj[b, a, gj, gi] = iou * tmask  # iou ratio
                tobj[b, a, gj, gi] = ((1.0 - self.gr) + self.gr * ops.stop_gradient(iou).clip(0, None)) * tmask  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = ops.fill(pcls.dtype, pcls.shape, self.cn)  # targets

                    t[mnp.arange(n), tcls[layer_index]] = self.cp
                    lcls += self.BCEcls(pcls, t, ops.tile(tmask[:, None], (1, t.shape[-1])))  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[layer_index]  # obj loss

        lbox *= self.hyp_box
        lobj *= self.hyp_obj
        lcls *= self.hyp_cls
        bs = p[0].shape[0]  # batch size

        loss = lbox + lobj + lcls

        return loss * bs, ops.stop_gradient(ops.stack((loss * bs * 8, lbox, lobj, lcls)))

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets.view(-1, 6)
        mask_t = targets[:, 1] >= 0
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, tmasks = (), (), (), (), ()
        gain = ops.ones(7, ms.int32) # normalized to gridspace gain
        ai = ops.tile(mnp.arange(na).view(-1, 1), (1, nt)) # shape: (na, nt)
        ai = ops.cast(ai, targets.dtype)
        targets = ops.concat((ops.tile(targets, (na, 1, 1)), ai[:, :, None]), 2) # append anchor indices # shape: (na, nt, 7)

        g = 0.5  # bias
        off = ops.cast(self._off, targets.dtype) * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = get_tensor(shape, targets.dtype)[[3, 2, 3, 2]] # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(na,nt,7) # xywhn -> xywh
            # Matches
            # if nt:
            r = t[..., 4:6] / anchors[:, None]  # wh ratio
            j = ops.maximum(r, 1 / r).max(2) < self.hyp_anchor_t # compare

            # t = t[j]  # filter
            mask_m_t = ops.logical_and(j, mask_t[None, :]).view(-1)
            t = t.view(-1, 7)

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            jk = ops.logical_and((gxy % 1 < g), (gxy > 1))
            lm = ops.logical_and((gxi % 1 < g), (gxi > 1))
            j, k = jk[:, 0], jk[:, 1]
            l, m = lm[:, 0], lm[:, 1]

            # # original
            # j = ops.stack((ops.ones_like(j), j, k, l, m)) # shape: (5, *)
            # t = ops.tile(t, (5, 1, 1)) # shape(5, *, 7)
            # mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            # t = t.view(-1, 7)
            # offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :]) #(1,*,2) + (5,1,2) -> (5,*,2)
            # offsets = offsets.view(-1, 2)

            # faster,
            tag1, tag2 = ops.identity(j), ops.identity(k)
            tag1, tag2 = ops.tile(tag1[:, None], (1, 2)), ops.tile(tag2[:, None], (1, 2))
            j_l = ops.logical_or(j, l).astype(ms.int32)
            k_m = ops.logical_or(k, m).astype(ms.int32)
            center = ops.ones_like(j_l)
            j = ops.stack((center, j_l, k_m))
            t = ops.tile(t, (3, 1, 1))  # shape(5, *, 7)
            t = t.view(-1, 7)
            mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
            offsets_new = ops.zeros((3,) + offsets.shape[1:], offsets.dtype)
            # offsets_new[0, :, :] = offsets[0, :, :]
            offsets_new[1:2, :, :] = ops.select(tag1.astype(ms.bool_), offsets[1, :, :], offsets[3, :, :])
            offsets_new[2:3, :, :] = ops.select(tag2.astype(ms.bool_), offsets[2, :, :], offsets[4, :, :])
            offsets = offsets_new
            offsets = offsets.view(-1, 2)

            # Define
            b, c, gxy, gwh, a = ops.cast(t[:, 0], ms.int32), \
                                ops.cast(t[:, 1], ms.int32), \
                                t[:, 2:4], \
                                t[:, 4:6], \
                                ops.cast(t[:, 6], ms.int32) # (image, class), grid xy, grid wh, anchors
            gij = ops.cast(gxy - offsets, ms.int32)
            gij = gij[:]
            gi, gj = gij[:, 0], gij[:, 1]  # grid indices
            gi = gi.clip(0, shape[3] - 1)
            gj = gj.clip(0, shape[2] - 1)


            # Append
            indices += (ops.stack((b, a, gj, gi), 0),)  # image, anchor, grid
            tbox += (ops.concat((gxy - gij, gwh), 1),)  # box
            anch += (anchors[a],)  # anchors
            tcls += (c,)  # class
            tmasks += (mask_m_t,)

        return ops.stack(tcls), \
               ops.stack(tbox), \
               ops.stack(indices), \
               ops.stack(anch), \
               ops.stack(tmasks) # class, box, (image, anchor, gridj, gridi), anchors, mask


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


if __name__ == '__main__':
    from mindyolo.utils.config import parse_config
    from mindyolo.models.losses.loss_factory import create_loss
    cfg = parse_config()
    loss_fn = create_loss(name='YOLOv7Loss',
                          **cfg.loss,
                          anchors=cfg.network.get('anchors', None),
                          stride=cfg.network.get('stride', None),
                          nc=cfg.data.get('nc', None))
    print(f"loss_fn is {loss_fn}")
