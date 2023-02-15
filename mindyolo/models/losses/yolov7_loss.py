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
    'YOLOv7Loss',
    'YOLOv7AuxLoss'
]


@register_model
class YOLOv7Loss(nn.Cell):
    def __init__(
            self,
            box, obj, cls, anchor_t, label_smoothing, fl_gamma, cls_pw, obj_pw,
            anchors, stride, nc, **kwargs
    ):
        super(YOLOv7Loss, self).__init__()
        self.hyp_box = box
        self.hyp_obj = obj
        self.hyp_cls = cls
        self.hyp_anchor_t = anchor_t
        self.anchors = anchors
        self.stride = stride
        self.nc = nc                    # number of classes
        self.na = len(anchors[0]) // 2  # number of anchors
        self.nl = len(anchors)          # number of layers

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

    def construct(self, p, targets, imgs):
        lcls, lbox, lobj = 0., 0., 0.
        bs, as_, gjs, gis, targets, anchors, tmasks = self.build_targets(p, targets, imgs) # bs: (nl, bs*5*na*gt_max)
        bs, as_, gjs, gis, targets, anchors, tmasks = ops.stop_gradient(bs), ops.stop_gradient(as_), \
                                                      ops.stop_gradient(gjs), ops.stop_gradient(gis), \
                                                      ops.stop_gradient(targets), ops.stop_gradient(anchors), \
                                                      ops.stop_gradient(tmasks)

        pre_gen_gains = ()
        for pp in p:
            pre_gen_gains += (get_tensor(pp.shape, targets.dtype)[[3, 2, 3, 2]],)

        # Losses
        # for i, pi in enumerate(p):  # layer index, layer predictions
        for i in range(self.nl): # layer index
            pi = p[i] # layer predictions
            b, a, gj, gi, tmask = bs[i], as_[i], gjs[i], gis[i], tmasks[i]  # image, anchor, gridy, gridx, tmask
            tobj = ops.zeros_like(pi[..., 0])  # target obj

            n = b.shape[0]  # number of targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            grid = ops.stack([gi, gj], axis=1)
            pxy = ops.Sigmoid()(ps[:, :2]) * 2. - 0.5
            pwh = (ops.Sigmoid()(ps[:, 2:4]) * 2) ** 2 * anchors[i]
            pbox = ops.concat((pxy, pwh), 1)  # predicted box
            selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
            selected_tbox[:, :2] -= grid
            iou = bbox_iou(pbox, selected_tbox, xywh=True, CIoU=True).view(-1)
            lbox += ((1.0 - iou) * tmask).sum() / tmask.astype(iou.dtype).sum().clip(1, None) # iou loss

            # Objectness
            tobj[b, a, gj, gi] = ((1.0 - self.gr) + self.gr * ops.stop_gradient(iou).clip(0, None)) * tmask  # iou ratio

            # Classification
            selected_tcls = ops.cast(targets[i][:, 1], ms.int32)
            if self.nc > 1:  # cls loss (only if multiple classes)
                t = ops.ones_like(ps[:, 5:]) * self.cn # targets
                t[mnp.arange(n), selected_tcls] = self.cp
                lcls += self.BCEcls(ps[:, 5:], t, ops.tile(tmask[:, None], (1, t.shape[1])))  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss

        lbox *= self.hyp_box
        lobj *= self.hyp_obj
        lcls *= self.hyp_cls
        bs = p[0].shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, ops.stop_gradient(ops.stack((loss, lbox, lobj, lcls)))

    def build_targets(self, p, targets, imgs):
        indices, anch, tmasks = self.find_3_positive(p, targets)

        na, n_gt_max = self.na, targets.shape[1]
        nl, batch_size, img_size = len(p), p[0].shape[0], imgs[0].shape[1]

        this_target = targets.view(-1, 6)

        txywh = this_target[:, 2:6] * img_size
        txyxy = xywh2xyxy(txywh)
        txyxy = txyxy.view(batch_size, n_gt_max, 4)
        this_target = this_target.view(batch_size, n_gt_max, 6)
        this_mask = this_target[:, :, 1] >= 0  # (bs, gt_max)

        pxyxys = ()
        p_cls = ()
        p_obj = ()
        all_b = ()
        all_a = ()
        all_gj = ()
        all_gi = ()
        all_anch = ()
        all_tmasks = ()

        # for i, pi in enumerate(p):
        for i in range(self.nl):
            pi = p[i]
            _this_indices = indices[i].view(4, 3 * na, batch_size, n_gt_max).transpose(0, 2, 1, 3).view(4, -1)
            _this_anch = anch[i].view(3 * na, batch_size, n_gt_max * 2).transpose(1, 0, 2).view(-1, 2)
            _this_mask = tmasks[i].view(3 * na, batch_size, n_gt_max).transpose(1, 0, 2).view(-1)

            # zhy_test
            _this_indices *= _this_mask[None, :]
            _this_anch *= _this_mask[:, None]

            b, a, gj, gi = ops.split(_this_indices, 0, 4)
            b, a, gj, gi = b.view(-1), a.view(-1), \
                           gj.view(-1), gi.view(-1)

            fg_pred = pi[b, a, gj, gi]
            p_obj += (fg_pred[:, 4:5].view(batch_size, 3 * na * n_gt_max, 1),)
            p_cls += (fg_pred[:, 5:].view(batch_size, 3 * na * n_gt_max, -1),)

            grid = ops.stack((gi, gj), axis=1)
            pxy = (ops.Sigmoid()(fg_pred[:, :2]) * 2. - 0.5 + grid) * self.stride[i]  # / 8.
            pwh = (ops.Sigmoid()(fg_pred[:, 2:4]) * 2) ** 2 * _this_anch * self.stride[i]  # / 8.
            pxywh = ops.concat((pxy, pwh), axis=-1)
            pxyxy = xywh2xyxy(pxywh)

            b, a, gj, gi, pxyxy, _this_anch, _this_mask = b.view(batch_size, -1), a.view(batch_size, -1), \
                                                          gj.view(batch_size, -1), gi.view(batch_size, -1), \
                                                          pxyxy.view(batch_size, -1, 4), \
                                                          _this_anch.view(batch_size, -1, 2), \
                                                          _this_mask.view(batch_size, -1)
            all_b += (b,)
            all_a += (a,)
            all_gj += (gj,)
            all_gi += (gi,)
            pxyxys += (pxyxy,)
            all_anch += (_this_anch,)
            all_tmasks += (_this_mask,)

        pxyxys = ops.concat(pxyxys, axis=1)  # nl * (bs, 5*na*gt_max, 4) -> cat -> (bs, c, 4) # nt = bs * gt_max
        p_obj = ops.concat(p_obj, axis=1)
        p_cls = ops.concat(p_cls, axis=1) # nl * (bs, 5*na*gt_max, 80) -> (bs, nl*5*na*gt_max, 80)
        all_b = ops.concat(all_b, axis=1) # nl * (bs, 5*na*gt_max) -> (bs, nl*5*na*gt_max)
        all_a = ops.concat(all_a, axis=1)
        all_gj = ops.concat(all_gj, axis=1)
        all_gi = ops.concat(all_gi, axis=1)
        all_anch = ops.concat(all_anch, axis=1)
        all_tmasks = ops.concat(all_tmasks, axis=1) # (bs, nl*5*na*gt_max)

        this_mask = all_tmasks[:, None, :] * this_mask[:, :, None] # (bs, gt_max, nl*5*na*gt_max,)

        # (bs, gt_max, 4), (bs, nl*5*na*gt_max, 4) -> (bs, gt_max, nl*5*na*gt_max)
        pair_wise_iou = batch_box_iou(txyxy, pxyxys) * this_mask  # (bs, gt_max, nl*5*na*gt_max,)
        pair_wise_iou_loss = -ops.log(pair_wise_iou + EPS)

        v, _ = ops.top_k(pair_wise_iou, 10) # (bs, gt_max, 10)
        dynamic_ks = ops.cast(v.sum(-1).clip(1, 10), ms.int32) # (bs, gt_max)

        # (bs, gt_max, 80)
        gt_cls_per_image = ops.one_hot(indices=ops.cast(this_target[:, :, 1], ms.int32),
                                       depth=self.nc,
                                       on_value=ops.ones(1, p_cls.dtype),
                                       off_value=ops.zeros(1, p_cls.dtype))
        # (bs, gt_max, nl*5*na*gt_max, 80)
        gt_cls_per_image = ops.tile(ops.expand_dims(ops.cast(gt_cls_per_image, p_cls.dtype), 2),
                                    (1, 1, pxyxys.shape[1], 1))

        cls_preds_ = ops.sqrt(ops.Sigmoid()(p_cls) * ops.Sigmoid()(p_obj))
        cls_preds_ = ops.tile(ops.expand_dims(cls_preds_, 1), (1, n_gt_max, 1, 1)) # (bs, nl*5*na*gt_max, 80) -> (bs, gt_max, nl*5*na*gt_max, 80)
        y = cls_preds_

        pair_wise_cls_loss = ops.binary_cross_entropy_with_logits(
            ops.log(y / (1 - y) + EPS),
            gt_cls_per_image,
            ops.ones(1, cls_preds_.dtype),
            ops.ones(1, cls_preds_.dtype),
            reduction="none",
        ).sum(-1) # (bs, gt_max, nl*5*na*gt_max)

        cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss
        cost = cost * this_mask
        cost += CLIP_VALUE * (1.0 - ops.cast(this_mask, cost.dtype))

        sort_cost, sort_idx = ops.top_k(-cost, 10, sorted=True) # (bs, gt_max, 10)
        sort_cost = -sort_cost
        pos_idx = ops.stack((mnp.arange(batch_size * n_gt_max), dynamic_ks.view(-1) - 1), -1)
        pos_v = ops.gather_nd(sort_cost.view(batch_size * n_gt_max, 10), pos_idx).view(batch_size, n_gt_max)
        matching_matrix = ops.cast(cost <= pos_v[:, :, None], ms.int32) * this_mask

        ## delete reduplicate match label, one anchor only match one gt
        cost_argmin = mnp.argmin(cost, axis=1)  # (bs, nl*5*na*gt_max)
        anchor_matching_gt_mask = ops.one_hot(cost_argmin,
                                              n_gt_max,
                                              ops.ones(1, ms.float16),
                                              ops.zeros(1, ms.float16), axis=-1).transpose(0, 2, 1)  # (bs, gt_max, nl*5*na*gt_max)
        matching_matrix = matching_matrix * ops.cast(anchor_matching_gt_mask, matching_matrix.dtype)

        fg_mask_inboxes = matching_matrix.astype(ms.float16).sum(1) > 0.0  # (bs, gt_max, nl*5*na*gt_max) -> (bs, nl*5*na*gt_max)
        all_tmasks = all_tmasks * ops.cast(fg_mask_inboxes, ms.int32) # (bs, nl*5*na*gt_max)
        matched_gt_inds = matching_matrix.argmax(1) # (bs, gt_max, nl*5*na*gt_max) -> (bs, nl*5*na*gt_max)
        matched_bs_inds = ops.tile(mnp.arange(batch_size)[:, None], (1, matching_matrix.shape[2]))  # (bs, nl*5*na*gt_max)
        matched_inds = ops.stack((matched_bs_inds.view(-1), matched_gt_inds.view(-1)), 1)  # (bs*nl*5*na*gt_max, 2)
        # zhy_test
        matched_inds *= all_tmasks.view(-1)[:, None]
        this_target = ops.gather_nd(this_target, matched_inds)  # (bs*nl*5*na*gt_max, 6)
        # this_target = this_target.view(-1, 6)[matched_gt_inds.view(-1,)] # (bs*nl*5*na*gt_max, 6)

        # (bs, nl*5*na*gt_max,) -> (bs, nl, 5*na*gt_max) -> (nl, bs*5*na*gt_max)
        # zhy_test
        matching_tmasks = all_tmasks.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1)
        matching_bs = all_b.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_as = all_a.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_gjs = all_gj.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_gis = all_gi.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_targets = this_target.view(batch_size, nl, -1, 6).transpose(1, 0, 2, 3).view(nl, -1, 6) * matching_tmasks[..., None]
        matching_anchs = all_anch.view(batch_size, nl, -1, 2).transpose(1, 0, 2, 3).view(nl, -1, 2) * matching_tmasks[..., None]

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs, matching_tmasks

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets.view(-1, 6) # (bs, gt_max, 6) -> (bs*gt_max, 6)
        mask_t = targets[:, 1] >= 0 # (bs*gt_max,)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch, tmasks = (), (), ()
        gain = ops.ones(7, ms.int32)  # normalized to gridspace gain
        ai = ops.tile(mnp.arange(na, dtype=targets.dtype).view(na, 1), (1, nt)) # shape: (na, nt)
        targets = ops.concat((ops.tile(targets, (na, 1, 1)), ai[:, :, None]), 2)  # append anchor indices # (na, nt, 7)

        g = 0.5  # bias
        off = ops.cast(self._off, targets.dtype) * g # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = get_tensor(shape, targets.dtype)[[3, 2, 3, 2]]  # xyxy gain # [W, H, W, H]

            # Match targets to anchors
            t = targets * gain # (na, nt, 7)
            # Matches
            # if nt:
            r = t[:, :, 4:6] / anchors[:, None, :]  # wh ratio
            j = ops.maximum(r, 1. / r).max(2) < self.hyp_anchor_t  # compare # (na, nt)

            # t = t[j]  # filter
            mask_m_t = ops.logical_and(j, mask_t[None, :]).view(-1)
            t = t.view(-1, 7) # (na*nt, 7)

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            jk = ops.logical_and((gxy % 1. < g), (gxy > 1.))
            lm = ops.logical_and((gxi % 1. < g), (gxi > 1.))
            j, k = jk[:, 0], jk[:, 1]
            l, m = lm[:, 0], lm[:, 1]

            # original
            # j = ops.stack((ops.ones_like(j), j, k, l, m))  # shape: (5, *)
            # t = ops.tile(t, (5, 1, 1))  # shape(5, *, 7)
            # t = t.view(-1, 7)
            # mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            # # t = t.repeat((5, 1, 1))[j]
            # offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
            # offsets = offsets.view(-1, 2) # (5*na*nt, 2)
            # # offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

            # faster,
            tag1, tag2 = ops.tile(j[:, None], (1, 2)), ops.tile(k[:, None], (1, 2))
            j_l = ops.logical_or(j, l).astype(ms.int32)
            k_m = ops.logical_or(k, m).astype(ms.int32)
            center = ops.ones_like(j_l)
            j = ops.stack((center, j_l, k_m))
            mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            t = ops.tile(t, (3, 1, 1))  # shape(5, *, 7)
            t = t.view(-1, 7)
            offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
            offsets_new = ops.zeros((3,) + offsets.shape[1:], offsets.dtype)
            # offsets_new[0, :, :] = offsets[0, :, :]
            offsets_new[1, :, :] = ops.select(tag1.astype(ms.bool_), offsets[1, ...], offsets[3, ...])
            offsets_new[2, :, :] = ops.select(tag2.astype(ms.bool_), offsets[2, ...], offsets[4, ...])
            offsets = offsets_new
            offsets = offsets.view(-1, 2)

            # Define
            b, c, gxy, gwh, a = ops.cast(t[:, 0], ms.int32), \
                                ops.cast(t[:, 1], ms.int32), \
                                t[:, 2:4], \
                                t[:, 4:6], \
                                ops.cast(t[:, 6], ms.int32)  # (image, class), grid xy, grid wh, anchors # b: (5*na*nt,), gxy: (5*na*nt, 2)
            # gij = gxy - offsets
            gij = ops.cast(gxy - offsets, ms.int32)
            gi, gj = gij[:, 0], gij[:, 1]  # grid indices
            gi = gi.clip(0, shape[3] - 1)
            gj = gj.clip(0, shape[2] - 1)


            # Append
            indices += (ops.stack((b, a, gj, gi), 0),)  # image, anchor, grid
            anch += (anchors[a],)  # anchors
            tmasks += (mask_m_t,)

        return indices, anch, tmasks


@register_model
class YOLOv7AuxLoss(nn.Cell):
    def __init__(self, model, autobalance=False):
        super(YOLOv7AuxLoss, self).__init__()
        h = model.opt
        self.hyp_box = h.box
        self.hyp_obj = h.obj
        self.hyp_cls = h.cls
        self.hyp_anchor_t = h.anchor_t

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.label_smoothing)  # positive, negative BCE targets
        # Focal loss
        g = h.fl_gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(bce_pos_weight=Tensor([h.cls_pw], ms.float32), gamma=g), \
                             FocalLoss(bce_pos_weight=Tensor([h.obj_pw], ms.float32), gamma=g)
        else:
            # Define criteria
            BCEcls = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h.cls_pw]), ms.float32))
            BCEobj = BCEWithLogitsLoss(bce_pos_weight=Tensor(np.array([h.obj_pw]), ms.float32))

        m = model.model[-1]  # Detect() module
        _balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.balance = ms.Parameter(Tensor(_balance, ms.float32), requires_grad=False)
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0, autobalance

        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.stride = m.stride

        self._off = Tensor([
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],  # j,k,l,m
            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
        ], dtype=ms.float32)

    def construct(self, p, targets, imgs):
        lcls, lbox, lobj = 0., 0., 0.
        targets_ori = targets
        bs, as_, gjs, gis, targets, anchors, tmasks = self.build_targets(p[:self.nl], targets_ori, imgs) # bs: (nl, bs*3*na*gt_max)
        bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux, tmasks_aux = self.build_targets_2(p[:self.nl], targets_ori, imgs) # bs: (nl, bs*5*na*gt_max)

        bs, as_, gjs, gis, targets, anchors, tmasks = ops.stop_gradient(bs), ops.stop_gradient(as_), \
                                                      ops.stop_gradient(gjs), ops.stop_gradient(gis), \
                                                      ops.stop_gradient(targets), ops.stop_gradient(anchors), \
                                                      ops.stop_gradient(tmasks)
        bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux, tmasks_aux = ops.stop_gradient(bs_aux),\
                                                                                  ops.stop_gradient(as_aux_), \
                                                                                  ops.stop_gradient(gjs_aux),\
                                                                                  ops.stop_gradient(gis_aux), \
                                                                                  ops.stop_gradient(targets_aux), \
                                                                                  ops.stop_gradient(anchors_aux), \
                                                                                  ops.stop_gradient(tmasks_aux)

        pre_gen_gains = ()
        # pre_gen_gains_aux = ()
        for pp in p[:self.nl]:
            pre_gen_gains += (get_tensor(pp.shape, targets.dtype)[[3, 2, 3, 2]],)
            # pre_gen_gains_aux += (get_tensor(pp.shape, targets.dtype)[[3, 2, 3, 2]],)

        # Losses
        for i in range(self.nl): # layer index
            pi = p[i] # layer predictions
            pi_aux = p[i + self.nl]
            b, a, gj, gi, tmask = bs[i], as_[i], gjs[i], gis[i], tmasks[i]  # image, anchor, gridy, gridx, tmask
            b_aux, a_aux, gj_aux, gi_aux, tmask_aux = bs_aux[i], as_aux_[i], gjs_aux[i], gis_aux[i], tmasks_aux[i]
            tobj = ops.zeros_like(pi[..., 0])  # target obj
            tobj_aux = ops.zeros_like(pi_aux[..., 0])  # target obj


            # 1. Branch1, Compute main branch loss
            n = b.shape[0]  # number of targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            # 1.1. Regression
            grid = ops.stack([gi, gj], axis=1)
            pxy = ops.Sigmoid()(ps[:, :2]) * 2. - 0.5
            pwh = (ops.Sigmoid()(ps[:, 2:4]) * 2) ** 2 * anchors[i]
            pbox = ops.concat((pxy, pwh), 1)  # predicted box
            selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
            selected_tbox[:, :2] -= grid
            iou = bbox_iou(pbox, selected_tbox, xywh=True, CIoU=True).view(-1)
            lbox += ((1.0 - iou) * tmask).sum() / tmask.astype(iou.dtype).sum().clip(1, None) # iou loss
            # 1.2. Objectness
            tobj[b, a, gj, gi] = ((1.0 - self.gr) + self.gr * ops.stop_gradient(iou).clip(0, None)) * tmask  # iou ratio
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / ops.stop_gradient(obji)
            # 1.3. Classification
            selected_tcls = ops.cast(targets[i][:, 1], ms.int32)
            if self.nc > 1:  # cls loss (only if multiple classes)
                t = ops.ones_like(ps[:, 5:]) * self.cn # targets
                t[mnp.arange(n), selected_tcls] = self.cp
                lcls += self.BCEcls(ps[:, 5:], t, ops.tile(tmask[:, None], (1, t.shape[1])))  # BCE

            # 2. Branch2, Compute Aux branch loss
            n_aux = b_aux.shape[0]  # number of targets
            ps_aux = pi[b_aux, a_aux, gj_aux, gi_aux]  # prediction subset corresponding to targets
            # 2.1. Regression
            grid_aux = ops.stack([gi_aux, gj_aux], axis=1)
            pxy_aux = ops.Sigmoid()(ps_aux[:, :2]) * 2. - 0.5
            pwh_aux = (ops.Sigmoid()(ps_aux[:, 2:4]) * 2) ** 2 * anchors_aux[i]
            pbox_aux = ops.concat((pxy_aux, pwh_aux), 1)  # predicted box
            selected_tbox_aux = targets_aux[i][:, 2:6] * pre_gen_gains[i]
            selected_tbox_aux[:, :2] -= grid_aux
            iou_aux = bbox_iou(pbox_aux, selected_tbox_aux, xywh=True, CIoU=True).view(-1)
            lbox += 0.25 * ((1.0 - iou_aux) * tmask_aux).sum() / tmask_aux.astype(iou_aux.dtype).sum().clip(1, None)  # iou loss
            # 1.2. Objectness
            tobj_aux[b_aux, a_aux, gj_aux, gi_aux] = ((1.0 - self.gr) + self.gr * ops.stop_gradient(iou_aux).clip(0, None)) * tmask_aux  # iou ratio
            obji_aux = self.BCEobj(pi_aux[..., 4], tobj_aux)
            lobj += 0.25 * obji_aux * self.balance[i]  # obj loss
            # 1.3. Classification
            selected_tcls_aux = ops.cast(targets_aux[i][:, 1], ms.int32)
            if self.nc > 1:  # cls loss (only if multiple classes)
                t_aux = ops.ones_like(ps_aux[:, 5:]) * self.cn  # targets
                t_aux[mnp.arange(n_aux), selected_tcls_aux] = self.cp
                lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux, ops.tile(tmask_aux[:, None], (1, t_aux.shape[1])))  # BCE

            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / ops.stop_gradient(obji)

        if self.autobalance:
            _balance_ssi = self.balance[self.ssi]
            self.balance /= _balance_ssi
        lbox *= self.hyp_box
        lobj *= self.hyp_obj
        lcls *= self.hyp_cls
        bs = p[0].shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, ops.stop_gradient(ops.stack((loss, lbox, lobj, lcls)))

    def build_targets(self, p, targets, imgs):
        indices, anch, tmasks = self.find_3_positive(p, targets)

        na, n_gt_max = self.na, targets.shape[1]
        nl, batch_size, img_size = len(p), p[0].shape[0], imgs[0].shape[1]

        this_target = targets.view(-1, 6)

        txywh = this_target[:, 2:6] * img_size
        txyxy = xywh2xyxy(txywh)
        txyxy = txyxy.view(batch_size, n_gt_max, 4)
        this_target = this_target.view(batch_size, n_gt_max, 6)
        this_mask = this_target[:, :, 1] >= 0  # (bs, gt_max)

        pxyxys = ()
        p_cls = ()
        p_obj = ()
        all_b = ()
        all_a = ()
        all_gj = ()
        all_gi = ()
        all_anch = ()
        all_tmasks = ()

        # for i, pi in enumerate(p):
        for i in range(self.nl):
            pi = p[i]
            _this_indices = indices[i].view(4, 3 * na, batch_size, n_gt_max).transpose(0, 2, 1, 3).view(4, -1)
            _this_anch = anch[i].view(3 * na, batch_size, n_gt_max * 2).transpose(1, 0, 2).view(-1, 2)
            _this_mask = tmasks[i].view(3 * na, batch_size, n_gt_max).transpose(1, 0, 2).view(-1)

            # zhy_test
            _this_indices *= _this_mask[None, :]
            _this_anch *= _this_mask[:, None]

            b, a, gj, gi = ops.split(_this_indices, 0, 4)
            b, a, gj, gi = b.view(-1), a.view(-1), \
                           gj.view(-1), gi.view(-1)

            fg_pred = pi[b, a, gj, gi]
            p_obj += (fg_pred[:, 4:5].view(batch_size, 3 * na * n_gt_max, 1),)
            p_cls += (fg_pred[:, 5:].view(batch_size, 3 * na * n_gt_max, -1),)

            grid = ops.stack((gi, gj), axis=1)
            pxy = (ops.Sigmoid()(fg_pred[:, :2]) * 2. - 0.5 + grid) * self.stride[i]  # / 8.
            pwh = (ops.Sigmoid()(fg_pred[:, 2:4]) * 2) ** 2 * _this_anch * self.stride[i]  # / 8.
            pxywh = ops.concat((pxy, pwh), axis=-1)
            pxyxy = xywh2xyxy(pxywh)

            b, a, gj, gi, pxyxy, _this_anch, _this_mask = b.view(batch_size, -1), a.view(batch_size, -1), \
                                                          gj.view(batch_size, -1), gi.view(batch_size, -1), \
                                                          pxyxy.view(batch_size, -1, 4), \
                                                          _this_anch.view(batch_size, -1, 2), \
                                                          _this_mask.view(batch_size, -1)
            all_b += (b,)
            all_a += (a,)
            all_gj += (gj,)
            all_gi += (gi,)
            pxyxys += (pxyxy,)
            all_anch += (_this_anch,)
            all_tmasks += (_this_mask,)

        pxyxys = ops.concat(pxyxys, axis=1)  # nl * (bs, 5*na*gt_max, 4) -> cat -> (bs, c, 4) # nt = bs * gt_max
        p_obj = ops.concat(p_obj, axis=1)
        p_cls = ops.concat(p_cls, axis=1) # nl * (bs, 5*na*gt_max, 80) -> (bs, nl*5*na*gt_max, 80)
        all_b = ops.concat(all_b, axis=1) # nl * (bs, 5*na*gt_max) -> (bs, nl*5*na*gt_max)
        all_a = ops.concat(all_a, axis=1)
        all_gj = ops.concat(all_gj, axis=1)
        all_gi = ops.concat(all_gi, axis=1)
        all_anch = ops.concat(all_anch, axis=1)
        all_tmasks = ops.concat(all_tmasks, axis=1) # (bs, nl*5*na*gt_max)

        this_mask = all_tmasks[:, None, :] * this_mask[:, :, None] # (bs, gt_max, nl*5*na*gt_max,)

        # (bs, gt_max, 4), (bs, nl*5*na*gt_max, 4) -> (bs, gt_max, nl*5*na*gt_max)
        pair_wise_iou = batch_box_iou(txyxy, pxyxys) * this_mask  # (bs, gt_max, nl*5*na*gt_max,)
        pair_wise_iou_loss = -ops.log(pair_wise_iou + EPS)

        # Top 20 iou sum for aux, default 10
        v, _ = ops.top_k(pair_wise_iou, 20) # (bs, gt_max, 20)
        dynamic_ks = ops.cast(v.sum(-1).clip(1, 20), ms.int32) # (bs, gt_max)

        # (bs, gt_max, 80)
        gt_cls_per_image = ops.one_hot(indices=ops.cast(this_target[:, :, 1], ms.int32),
                                       depth=self.nc,
                                       on_value=ops.ones(1, p_cls.dtype),
                                       off_value=ops.zeros(1, p_cls.dtype))
        # (bs, gt_max, nl*5*na*gt_max, 80)
        gt_cls_per_image = ops.tile(ops.expand_dims(ops.cast(gt_cls_per_image, p_cls.dtype), 2),
                                    (1, 1, pxyxys.shape[1], 1))

        cls_preds_ = ops.sqrt(ops.Sigmoid()(p_cls) * ops.Sigmoid()(p_obj))
        cls_preds_ = ops.tile(ops.expand_dims(cls_preds_, 1), (1, n_gt_max, 1, 1)) # (bs, nl*5*na*gt_max, 80) -> (bs, gt_max, nl*5*na*gt_max, 80)
        y = cls_preds_

        pair_wise_cls_loss = ops.binary_cross_entropy_with_logits(
            ops.log(y / (1 - y) + EPS),
            gt_cls_per_image,
            ops.ones(1, cls_preds_.dtype),
            ops.ones(1, cls_preds_.dtype),
            reduction="none",
        ).sum(-1) # (bs, gt_max, nl*5*na*gt_max)

        cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss
        cost = cost * this_mask
        cost += CLIP_VALUE * (1.0 - ops.cast(this_mask, cost.dtype))

        sort_cost, sort_idx = ops.top_k(-cost, 20, sorted=True) # (bs, gt_max, 20)
        sort_cost = -sort_cost
        pos_idx = ops.stack((mnp.arange(batch_size * n_gt_max), dynamic_ks.view(-1) - 1), -1)
        pos_v = ops.gather_nd(sort_cost.view(batch_size * n_gt_max, 20), pos_idx).view(batch_size, n_gt_max)
        matching_matrix = ops.cast(cost <= pos_v[:, :, None], ms.int32) * this_mask

        ## delete reduplicate match label, one anchor only match one gt
        cost_argmin = mnp.argmin(cost, axis=1)  # (bs, nl*5*na*gt_max)
        anchor_matching_gt_mask = ops.one_hot(cost_argmin,
                                              n_gt_max,
                                              ops.ones(1, ms.float16),
                                              ops.zeros(1, ms.float16), axis=-1).transpose(0, 2, 1)  # (bs, gt_max, nl*5*na*gt_max)
        matching_matrix = matching_matrix * ops.cast(anchor_matching_gt_mask, matching_matrix.dtype)

        fg_mask_inboxes = matching_matrix.astype(ms.float16).sum(1) > 0.0  # (bs, gt_max, nl*5*na*gt_max) -> (bs, nl*5*na*gt_max)
        all_tmasks = all_tmasks * ops.cast(fg_mask_inboxes, ms.int32) # (bs, nl*5*na*gt_max)
        matched_gt_inds = matching_matrix.argmax(1) # (bs, gt_max, nl*5*na*gt_max) -> (bs, nl*5*na*gt_max)
        matched_bs_inds = ops.tile(mnp.arange(batch_size)[:, None], (1, matching_matrix.shape[2]))  # (bs, nl*5*na*gt_max)
        matched_inds = ops.stack((matched_bs_inds.view(-1), matched_gt_inds.view(-1)), 1)  # (bs*nl*5*na*gt_max, 2)
        matched_inds *= all_tmasks.view(-1)[:, None]
        this_target = ops.gather_nd(this_target, matched_inds)  # (bs*nl*5*na*gt_max, 6)
        # this_target = this_target.view(-1, 6)[matched_gt_inds.view(-1,)] # (bs*nl*5*na*gt_max, 6)

        # (bs, nl*5*na*gt_max,) -> (bs, nl, 5*na*gt_max) -> (nl, bs*5*na*gt_max)
        matching_tmasks = all_tmasks.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1)
        matching_bs = all_b.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_as = all_a.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_gjs = all_gj.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_gis = all_gi.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_targets = this_target.view(batch_size, nl, -1, 6).transpose(1, 0, 2, 3).view(nl, -1, 6) * matching_tmasks[..., None]
        matching_anchs = all_anch.view(batch_size, nl, -1, 2).transpose(1, 0, 2, 3).view(nl, -1, 2) * matching_tmasks[..., None]

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs, matching_tmasks

    def build_targets_2(self, p, targets, imgs):
        indices, anch, tmasks = self.find_5_positive(p, targets)

        na, n_gt_max = self.na, targets.shape[1]
        nl, batch_size, img_size = len(p), p[0].shape[0], imgs[0].shape[1]

        this_target = targets.view(-1, 6)

        txywh = this_target[:, 2:6] * img_size
        txyxy = xywh2xyxy(txywh)
        txyxy = txyxy.view(batch_size, n_gt_max, 4)
        this_target = this_target.view(batch_size, n_gt_max, 6)
        this_mask = this_target[:, :, 1] >= 0  # (bs, gt_max)

        pxyxys = ()
        p_cls = ()
        p_obj = ()
        all_b = ()
        all_a = ()
        all_gj = ()
        all_gi = ()
        all_anch = ()
        all_tmasks = ()

        # for i, pi in enumerate(p):
        for i in range(self.nl):
            pi = p[i]
            _this_indices = indices[i].view(4, 5 * na, batch_size, n_gt_max).transpose(0, 2, 1, 3).view(4, -1)
            _this_anch = anch[i].view(5 * na, batch_size, n_gt_max * 2).transpose(1, 0, 2).view(-1, 2)
            _this_mask = tmasks[i].view(5 * na, batch_size, n_gt_max).transpose(1, 0, 2).view(-1)

            # zhy_test
            _this_indices *= _this_mask[None, :]
            _this_anch *= _this_mask[:, None]

            b, a, gj, gi = ops.split(_this_indices, 0, 4)
            b, a, gj, gi = b.view(-1), a.view(-1), \
                           gj.view(-1), gi.view(-1)

            fg_pred = pi[b, a, gj, gi]
            p_obj += (fg_pred[:, 4:5].view(batch_size, 5 * na * n_gt_max, 1),)
            p_cls += (fg_pred[:, 5:].view(batch_size, 5 * na * n_gt_max, -1),)

            grid = ops.stack((gi, gj), axis=1)
            pxy = (ops.Sigmoid()(fg_pred[:, :2]) * 2. - 0.5 + grid) * self.stride[i]  # / 8.
            pwh = (ops.Sigmoid()(fg_pred[:, 2:4]) * 2) ** 2 * _this_anch * self.stride[i]  # / 8.
            pxywh = ops.concat((pxy, pwh), axis=-1)
            pxyxy = xywh2xyxy(pxywh)

            b, a, gj, gi, pxyxy, _this_anch, _this_mask = b.view(batch_size, -1), a.view(batch_size, -1), \
                                                          gj.view(batch_size, -1), gi.view(batch_size, -1), \
                                                          pxyxy.view(batch_size, -1, 4), \
                                                          _this_anch.view(batch_size, -1, 2), \
                                                          _this_mask.view(batch_size, -1)
            all_b += (b,)
            all_a += (a,)
            all_gj += (gj,)
            all_gi += (gi,)
            pxyxys += (pxyxy,)
            all_anch += (_this_anch,)
            all_tmasks += (_this_mask,)

        pxyxys = ops.concat(pxyxys, axis=1)  # nl * (bs, 5*na*gt_max, 4) -> cat -> (bs, c, 4) # nt = bs * gt_max
        p_obj = ops.concat(p_obj, axis=1)
        p_cls = ops.concat(p_cls, axis=1)  # nl * (bs, 5*na*gt_max, 80) -> (bs, nl*5*na*gt_max, 80)
        all_b = ops.concat(all_b, axis=1)  # nl * (bs, 5*na*gt_max) -> (bs, nl*5*na*gt_max)
        all_a = ops.concat(all_a, axis=1)
        all_gj = ops.concat(all_gj, axis=1)
        all_gi = ops.concat(all_gi, axis=1)
        all_anch = ops.concat(all_anch, axis=1)
        all_tmasks = ops.concat(all_tmasks, axis=1)  # (bs, nl*5*na*gt_max)

        this_mask = all_tmasks[:, None, :] * this_mask[:, :, None]  # (bs, gt_max, nl*5*na*gt_max,)

        # (bs, gt_max, 4), (bs, nl*5*na*gt_max, 4) -> (bs, gt_max, nl*5*na*gt_max)
        pair_wise_iou = batch_box_iou(txyxy, pxyxys) * this_mask  # (bs, gt_max, nl*5*na*gt_max,)
        pair_wise_iou_loss = -ops.log(pair_wise_iou + EPS)

        # Top 20 iou sum for aux, default 10
        v, _ = ops.top_k(pair_wise_iou, 20)  # (bs, gt_max, 20)
        dynamic_ks = ops.cast(v.sum(-1).clip(1, 20), ms.int32)  # (bs, gt_max)

        # (bs, gt_max, 80)
        gt_cls_per_image = ops.one_hot(indices=ops.cast(this_target[:, :, 1], ms.int32),
                                       depth=self.nc,
                                       on_value=ops.ones(1, p_cls.dtype),
                                       off_value=ops.zeros(1, p_cls.dtype))
        # (bs, gt_max, nl*5*na*gt_max, 80)
        gt_cls_per_image = ops.tile(ops.expand_dims(ops.cast(gt_cls_per_image, p_cls.dtype), 2),
                                    (1, 1, pxyxys.shape[1], 1))

        cls_preds_ = ops.sqrt(ops.Sigmoid()(p_cls) * ops.Sigmoid()(p_obj))
        cls_preds_ = ops.tile(ops.expand_dims(cls_preds_, 1),
                              (1, n_gt_max, 1, 1))  # (bs, nl*5*na*gt_max, 80) -> (bs, gt_max, nl*5*na*gt_max, 80)
        y = cls_preds_

        pair_wise_cls_loss = ops.binary_cross_entropy_with_logits(
            ops.log(y / (1 - y) + EPS),
            gt_cls_per_image,
            ops.ones(1, cls_preds_.dtype),
            ops.ones(1, cls_preds_.dtype),
            reduction="none",
        ).sum(-1)  # (bs, gt_max, nl*5*na*gt_max)

        cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss
        cost = cost * this_mask
        cost += CLIP_VALUE * (1.0 - ops.cast(this_mask, cost.dtype))

        sort_cost, sort_idx = ops.top_k(-cost, 20, sorted=True)  # (bs, gt_max, 20)
        sort_cost = -sort_cost
        pos_idx = ops.stack((mnp.arange(batch_size * n_gt_max), dynamic_ks.view(-1) - 1), -1)
        pos_v = ops.gather_nd(sort_cost.view(batch_size * n_gt_max, 20), pos_idx).view(batch_size, n_gt_max)
        matching_matrix = ops.cast(cost <= pos_v[:, :, None], ms.int32) * this_mask

        ## delete reduplicate match label, one anchor only match one gt
        cost_argmin = mnp.argmin(cost, axis=1)  # (bs, nl*5*na*gt_max)
        anchor_matching_gt_mask = ops.one_hot(cost_argmin,
                                              n_gt_max,
                                              ops.ones(1, ms.float16),
                                              ops.zeros(1, ms.float16), axis=-1).transpose(0, 2,
                                                                                           1)  # (bs, gt_max, nl*5*na*gt_max)
        matching_matrix = matching_matrix * ops.cast(anchor_matching_gt_mask, matching_matrix.dtype)

        fg_mask_inboxes = matching_matrix.astype(ms.float16).sum(
            1) > 0.0  # (bs, gt_max, nl*5*na*gt_max) -> (bs, nl*5*na*gt_max)
        all_tmasks = all_tmasks * ops.cast(fg_mask_inboxes, ms.int32)  # (bs, nl*5*na*gt_max)
        matched_gt_inds = matching_matrix.argmax(1)  # (bs, gt_max, nl*5*na*gt_max) -> (bs, nl*5*na*gt_max)
        matched_bs_inds = ops.tile(mnp.arange(batch_size)[:, None],
                                   (1, matching_matrix.shape[2]))  # (bs, nl*5*na*gt_max)
        matched_inds = ops.stack((matched_bs_inds.view(-1), matched_gt_inds.view(-1)), 1)  # (bs*nl*5*na*gt_max, 2)
        matched_inds *= all_tmasks.view(-1)[:, None]
        this_target = ops.gather_nd(this_target, matched_inds)  # (bs*nl*5*na*gt_max, 6)
        # this_target = this_target.view(-1, 6)[matched_gt_inds.view(-1,)] # (bs*nl*5*na*gt_max, 6)

        # (bs, nl*5*na*gt_max,) -> (bs, nl, 5*na*gt_max) -> (nl, bs*5*na*gt_max)
        matching_tmasks = all_tmasks.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1)
        matching_bs = all_b.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_as = all_a.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_gjs = all_gj.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_gis = all_gi.view(batch_size, nl, -1).transpose(1, 0, 2).view(nl, -1) * matching_tmasks
        matching_targets = this_target.view(batch_size, nl, -1, 6).transpose(1, 0, 2, 3).view(nl, -1, 6) * \
                           matching_tmasks[..., None]
        matching_anchs = all_anch.view(batch_size, nl, -1, 2).transpose(1, 0, 2, 3).view(nl, -1, 2) * matching_tmasks[..., None]

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs, matching_tmasks

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets.view(-1, 6) # (bs, gt_max, 6) -> (bs*gt_max, 6)
        mask_t = targets[:, 1] >= 0 # (bs*gt_max,)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch, tmasks = (), (), ()
        gain = ops.ones(7, ms.int32)  # normalized to gridspace gain
        ai = ops.tile(mnp.arange(na, dtype=targets.dtype).view(na, 1), (1, nt)) # shape: (na, nt)
        targets = ops.concat((ops.tile(targets, (na, 1, 1)), ai[:, :, None]), 2)  # append anchor indices # (na, nt, 7)

        g = 0.5  # bias
        off = ops.cast(self._off, targets.dtype) * g # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = get_tensor(shape, targets.dtype)[[3, 2, 3, 2]]  # xyxy gain # [W, H, W, H]

            # Match targets to anchors
            t = targets * gain # (na, nt, 7)
            # Matches
            # if nt:
            r = t[:, :, 4:6] / anchors[:, None, :]  # wh ratio
            j = ops.maximum(r, 1. / r).max(2) < self.hyp_anchor_t  # compare # (na, nt)

            # t = t[j]  # filter
            mask_m_t = ops.logical_and(j, mask_t[None, :]).view(-1)
            t = t.view(-1, 7) # (na*nt, 7)

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            jk = ops.logical_and((gxy % 1. < g), (gxy > 1.))
            lm = ops.logical_and((gxi % 1. < g), (gxi > 1.))
            j, k = jk[:, 0], jk[:, 1]
            l, m = lm[:, 0], lm[:, 1]

            # original
            # j = ops.stack((ops.ones_like(j), j, k, l, m))  # shape: (5, *)
            # t = ops.tile(t, (5, 1, 1))  # shape(5, *, 7)
            # t = t.view(-1, 7)
            # mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            # # t = t.repeat((5, 1, 1))[j]
            # offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
            # offsets = offsets.view(-1, 2) # (5*na*nt, 2)
            # # offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

            # faster,
            tag1, tag2 = ops.tile(j[:, None], (1, 2)), ops.tile(k[:, None], (1, 2))
            j_l = ops.logical_or(j, l).astype(ms.int32)
            k_m = ops.logical_or(k, m).astype(ms.int32)
            center = ops.ones_like(j_l)
            j = ops.stack((center, j_l, k_m))
            mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            t = ops.tile(t, (3, 1, 1))  # shape(5, *, 7)
            t = t.view(-1, 7)
            offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
            offsets_new = ops.zeros((3,) + offsets.shape[1:], offsets.dtype)
            # offsets_new[0, :, :] = offsets[0, :, :]
            offsets_new[1, :, :] = ops.select(tag1.astype(ms.bool_), offsets[1, ...], offsets[3, ...])
            offsets_new[2, :, :] = ops.select(tag2.astype(ms.bool_), offsets[2, ...], offsets[4, ...])
            offsets = offsets_new
            offsets = offsets.view(-1, 2)

            # Define
            b, c, gxy, gwh, a = ops.cast(t[:, 0], ms.int32), \
                                ops.cast(t[:, 1], ms.int32), \
                                t[:, 2:4], \
                                t[:, 4:6], \
                                ops.cast(t[:, 6], ms.int32)  # (image, class), grid xy, grid wh, anchors # b: (5*na*nt,), gxy: (5*na*nt, 2)
            # gij = gxy - offsets
            gij = ops.cast(gxy - offsets, ms.int32)
            gi, gj = gij[:, 0], gij[:, 1]  # grid indices
            gi = gi.clip(0, shape[3] - 1)
            gj = gj.clip(0, shape[2] - 1)


            # Append
            indices += (ops.stack((b, a, gj, gi), 0),)  # image, anchor, grid
            anch += (anchors[a],)  # anchors
            tmasks += (mask_m_t,)

        return indices, anch, tmasks

    def find_5_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = targets.view(-1, 6)  # (bs, gt_max, 6) -> (bs*gt_max, 6)
        mask_t = targets[:, 1] >= 0  # (bs*gt_max,)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch, tmasks = (), (), ()
        gain = ops.ones(7, ms.int32)  # normalized to gridspace gain
        ai = ops.tile(mnp.arange(na, dtype=targets.dtype).view(na, 1), (1, nt))  # shape: (na, nt)
        targets = ops.concat((ops.tile(targets, (na, 1, 1)), ai[:, :, None]), 2)  # append anchor indices # (na, nt, 7)

        g = 1.0  # bias
        off = ops.cast(self._off, targets.dtype) * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = get_tensor(shape, targets.dtype)[[3, 2, 3, 2]]  # xyxy gain # [W, H, W, H]

            # Match targets to anchors
            t = targets * gain  # (na, nt, 7)
            # Matches
            r = t[:, :, 4:6] / anchors[:, None, :]  # wh ratio
            j = ops.maximum(r, 1. / r).max(2) < self.hyp_anchor_t  # compare # (na, nt)

            # t = t[j]  # filter
            mask_m_t = ops.logical_and(j, mask_t[None, :]).view(-1) # filter
            t = t.view(-1, 7)  # (na*nt, 7)

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            jk = ops.logical_and((gxy % 1. < g), (gxy > 1.)).astype(ms.int32)
            lm = ops.logical_and((gxi % 1. < g), (gxi > 1.)).astype(ms.int32)
            j, k = jk[:, 0], jk[:, 1]
            l, m = lm[:, 0], lm[:, 1]

            # original
            j = ops.stack((ops.ones_like(j), j, k, l, m))  # shape: (5, *)
            t = ops.tile(t, (5, 1, 1))  # shape(5, *, 7)
            t = t.view(-1, 7)
            mask_m_t = (ops.cast(j, ms.int32) * ops.cast(mask_m_t[None, :], ms.int32)).view(-1)
            offsets = (ops.zeros_like(gxy)[None, :, :] + off[:, None, :])  # (1,*,2) + (5,1,2) -> (5,na*nt,2)
            offsets = offsets.view(-1, 2) # (5*na*nt, 2)

            # Define
            b, c, gxy, gwh, a = ops.cast(t[:, 0], ms.int32), \
                                ops.cast(t[:, 1], ms.int32), \
                                t[:, 2:4], \
                                t[:, 4:6], \
                                ops.cast(t[:, 6], ms.int32)  # (image, class), grid xy, grid wh, anchors # b: (5*na*nt,), gxy: (5*na*nt, 2)
            # gij = gxy - offsets
            gij = ops.cast(gxy - offsets, ms.int32)
            gi, gj = gij[:, 0], gij[:, 1]  # grid indices
            gi = gi.clip(0, shape[3] - 1)
            gj = gj.clip(0, shape[2] - 1)

            # Append
            indices += (ops.stack((b, a, gj, gi), 0),)  # image, anchor, grid
            anch += (anchors[a],)  # anchors
            tmasks += (mask_m_t,)

        return indices, anch, tmasks


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
