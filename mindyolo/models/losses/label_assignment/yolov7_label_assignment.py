from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mindspore as ms
from mindspore import nn, ops

from mindyolo.utils import logger

__all__ = ['YOLOv7LabelAssignment']


@ops.constexpr
def list2tensor(list_x, dtype=ms.int32):
    return ms.Tensor(np.array(list_x), dtype)


def box_iou_np(box1, box2):
    """
    Calculate the iou of box1 and box2 with numpy.
    Args:
        box1 (ndarray): [N, 4]
        box2 (ndarray): [M, 4], usually N != M
    Return:
        iou (ndarray): iou of box1 and box2, [N, M]
    """
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(
        box1[:, None, :2], box2[:, :2])).clip(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def topk(x, k):
    x_shape = x.shape
    out_shape = x_shape[:-1] + (k,)
    matrix = x.reshape((-1, x_shape[-1]))
    c, n = matrix.shape
    index_part = np.argpartition(matrix, -k)[:, -k:]
    index_channel = np.arange(c)[:, None]
    part_sort_k = np.argsort(matrix[index_channel, index_part], axis=-1)
    top_k_index = np.flip(index_part[index_channel, part_sort_k], axis=-1)
    top_k_scores = matrix[index_channel, top_k_index].reshape(out_shape)
    return top_k_scores, top_k_index.reshape(out_shape)


def sigmoid(x):
    y = 1 / (1 + (np.exp((-x))))
    return y


def one_hot(x, num_clases):
    return np.eye(num_clases)[x.astype(np.int32)]


def binary_cross_entropy_with_logits(x, y):
    x = sigmoid(x)
    return -(y * np.log(x) + (1 - y) * np.log(1 - x))


class YOLOv7LabelAssignmentNp:
    """
    Label Assignment of YOLOv7
    Args:
        anchors (Tensor): anchors in config
        na (int): channel numbers
        bias (float): bias in find positive
        stride (list): stride list of YOLO out's feature
        anchor_t(float): filter threshold in find_positive
        use_aux (bool): whether use aux loss
        thread_num(int): number of multi-threaded parallels
    Inputs:
        p (list(Tensor)): predicts(layer_num, batch_size, anchors_num, feature_size_h, feature_size_w, class_num+1+4).
                    1 is positive object predict, 4 is x, y, w, h
        targets (Tensor): targets(batch_size, pre_batch_target_num, 6). 6 is image_index, cls_id, x, y, w, h
        img (Tensor): input image
    """

    def __init__(self, anchors, na=3, bias=0.5, stride=[8, 16, 32], anchor_t=4, use_aux=False, thread_num=4):
        super(YOLOv7LabelAssignmentNp, self).__init__()
        if isinstance(anchors, ms.Tensor):
            anchors = anchors.asnumpy()
        if isinstance(stride, ms.Tensor):
            stride = stride.asnumpy()
        self.anchors = anchors
        self.na = na
        self.bias = bias
        self.stride = stride
        self.off = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
            ],
            dtype=np.float32) * bias  # offsets
        self.use_aux = use_aux
        self.anchor_t = anchor_t
        self.pool = ThreadPoolExecutor(max_workers=thread_num)
        logger.info(f"start ThreadPoolExecutor, max_workers is {thread_num}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.shutdown()
        logger.info("ThreadPoolExecutor shutdown")

    def find_positive(self, outputs, targets, all_anchors, g=0.5):
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = np.ones(7, dtype=np.float32)  # normalized to gridspace gain
        ai = np.tile(np.arange(na, dtype=np.float32).reshape(na, 1), [1, nt])
        targets = np.concatenate((np.tile(
            np.expand_dims(targets, 0), [na, 1, 1]), ai[:, :, None]), 2)

        for i in range(len(all_anchors)):
            anchors = all_anchors[i]
            gain[2:6] = np.array(outputs[i].shape, dtype=np.float32)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets_labels to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = np.maximum(r, 1. / r).max(2) < self.anchor_t
                if not np.any(j):
                    t = targets[0]
                    offsets = 0
                    continue
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = np.stack([np.ones_like(j), j, k, l, m])
                t = np.tile(t, [5, 1, 1])[j]
                offsets = (np.zeros_like(gxy)[None] + self.off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].astype(np.int64).T
            gxy = t[:, 2:4]  # grid xy
            gij = (gxy - offsets).astype(np.int64)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].astype(np.int64)  # anchor indices
            gj, gi = gj.clip(0, gain[3] - 1).astype(np.int64), gi.clip(
                0, gain[2] - 1).astype(np.int64)
            indices.append((b, a, gj, gi))
            anch.append(anchors[a])  # anchors
        # return numpy rather than tensor
        return indices, anch

    def build_target_batch(self, inp):
        batch_idx, p, targets, indices, anch, imgs_h, min_topk = inp
        b_idx = targets[:, 0] == batch_idx
        if b_idx.sum() == 0:
            return None
        this_target = targets[b_idx]
        txywh = this_target[:, 2:6] * imgs_h
        # this_target[:, 2:6] * 640
        txyxy = xywh2xyxy(txywh)  # tensor op

        pxyxys, p_cls, p_obj = [], [], []
        from_which_layer = []
        all_b, all_a, all_gj, all_gi = [], [], [], []
        all_anch = []

        empty_feats_num = 0

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            idx = (b == batch_idx)
            if idx.sum() == 0:
                empty_feats_num += 1
                continue
            b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
            all_b.append(b)
            all_a.append(a)
            all_gj.append(gj)
            all_gi.append(gi)
            all_anch.append(anch[i][idx])
            from_which_layer.append(np.ones([len(b)]) * i)

            fg_pred = pi[b, a, gj, gi]  # numpy index
            if len(fg_pred.shape) == 1:  # Note: when only one sample
                fg_pred = fg_pred[None, :]
            p_obj.append(fg_pred[:, 4:5])
            p_cls.append(fg_pred[:, 5:])

            grid = np.stack([gi, gj], 1)
            pxy = (sigmoid(fg_pred[:, :2]) * 2. - 0.5 + grid) * self.stride[i]
            anch_inx = anch[i][idx]
            pwh = (sigmoid(fg_pred[:, 2:4]) * 2) ** 2 * anch_inx * self.stride[i]
            pxywh = np.concatenate([pxy, pwh], -1)
            pxyxy = xywh2xyxy(pxywh)
            pxyxys.append(pxyxy)

        if empty_feats_num == len(p) or len(pxyxys) == 0:  # Note: empty
            return None
        pxyxys = np.concatenate(pxyxys, 0)

        p_obj = np.concatenate(p_obj, 0)
        p_cls = np.concatenate(p_cls, 0)

        from_which_layer = np.concatenate(from_which_layer, 0)
        all_b = np.concatenate(all_b, 0)
        all_a = np.concatenate(all_a, 0)
        all_gj = np.concatenate(all_gj, 0)
        all_gi = np.concatenate(all_gi, 0)
        all_anch = np.concatenate(all_anch, 0)

        pairwise_ious = box_iou_np(txyxy, pxyxys)
        # [N, 4] [M, 4] to get [N, M] ious

        pairwise_iou_loss = -np.log(pairwise_ious + 1e-8)

        topk_ious, _ = topk(pairwise_ious, min(min_topk, pairwise_ious.shape[1]))
        dynamic_ks = np.maximum(topk_ious.sum(1).astype(np.int32), 1)
        # this_target: (6,) image_index, cls_id, x, y, w, h
        # gt_cls_per_image: (target_num, M, class_num)
        gt_cls_per_image = np.tile(one_hot(this_target[:, 1], p_cls.shape[-1])[:, None, :],
                                   [1, pxyxys.shape[0], 1])

        num_gt = this_target.shape[0]
        cls_preds = (
                sigmoid(np.tile(p_cls[None, :, :], [num_gt, 1, 1])) *
                sigmoid(np.tile(p_obj[None, :, :], [num_gt, 1, 1])))

        y = np.sqrt(cls_preds + 1e-8)
        pairwise_cls_loss = binary_cross_entropy_with_logits(
            np.log(y / (1 - y)), gt_cls_per_image).sum(-1)

        cost = (pairwise_cls_loss + 3.0 * pairwise_iou_loss)

        matching_matrix = np.zeros(cost.shape)
        for gt_idx in range(num_gt):
            _, pos_idx = topk(-cost[gt_idx], k=dynamic_ks[gt_idx])
            matching_matrix[gt_idx, pos_idx] = 1.0

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            cost_argmin = np.argmin(cost[:, anchor_matching_gt > 1], 0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

        from_which_layer = from_which_layer[fg_mask_inboxes]
        all_b = all_b[fg_mask_inboxes]
        all_a = all_a[fg_mask_inboxes]
        all_gj = all_gj[fg_mask_inboxes]
        all_gi = all_gi[fg_mask_inboxes]
        all_anch = all_anch[fg_mask_inboxes]

        this_target = this_target[matched_gt_inds]
        return batch_idx, all_b, all_a, all_gj, all_gi, this_target, all_anch, from_which_layer

    def build_targets(self, p, targets, batch_size, img_h, min_topk=10, g=0.5, max_gt=None):
        targets = targets.reshape((-1, 6))
        targets = targets[targets[:, 1] >= 0]
        indices, anch = self.find_positive(p, targets, self.anchors, g)
        # numpy indices, anch for fast assign
        matching_bs = [[] for _ in p]
        matching_as = [[] for _ in p]
        matching_gjs = [[] for _ in p]
        matching_gis = [[] for _ in p]
        matching_targets = [[] for _ in p]
        matching_anchs = [[] for _ in p]

        nl = len(p)
        data = [(batch_idx, p, targets, indices, anch, img_h, min_topk)
                for batch_idx in range(p[0].shape[0])]
        for res in self.pool.map(self.build_target_batch, data):
            if res is not None:
                batch_idx, all_b, all_a, all_gj, all_gi, this_target, all_anch, from_which_layer = res
                for i in range(nl):
                    layer_idx = from_which_layer == i
                    matching_bs[i].append(all_b[layer_idx])
                    matching_as[i].append(all_a[layer_idx])
                    matching_gjs[i].append(all_gj[layer_idx])
                    matching_gis[i].append(all_gi[layer_idx])
                    matching_targets[i].append(this_target[layer_idx])
                    matching_anchs[i].append(all_anch[layer_idx])
        max_gt = batch_size * 10 * min_topk
        masks = [[] for _ in p]
        for i in range(nl):
            h, w = p[i].shape[2], p[i].shape[3]
            gains = [w, h, w, h]
            bs, as_, gjs, gis, tgts, anchs, mask = np.zeros(max_gt, np.int32), np.zeros(max_gt, np.int32), \
                                                   np.zeros(max_gt, np.int32), np.zeros(max_gt, np.int32), \
                                                   np.zeros((max_gt, 6), np.float32), \
                                                   np.zeros((max_gt, 2), np.float32), \
                                                   np.zeros((max_gt, 1), np.float32)

            if matching_targets[i] != []:
                m_b = np.concatenate(matching_bs[i], 0)
                pos_size = m_b.shape[0]
                if pos_size > max_gt:
                    print(f"WARNING posistive box {pos_size} more than max_gt {max_gt}")
                    pos_size = max_gt
                bs[:pos_size] = m_b[:pos_size]
                as_[:pos_size] = np.concatenate(matching_as[i], 0)[:pos_size]
                gjs[:pos_size] = np.concatenate(matching_gjs[i], 0)[:pos_size]
                gis[:pos_size] = np.concatenate(matching_gis[i], 0)[:pos_size]
                tgts[:pos_size] = np.concatenate(matching_targets[i], 0)[:pos_size]
                grid = np.stack([gis, gjs], axis=1)
                tgts[:, 2:6] = tgts[:, 2:6] * gains
                tgts[:, 2:4] -= grid
                anchs[:pos_size] = np.concatenate(matching_anchs[i], 0)[:pos_size]
                mask[:pos_size] = np.ones((pos_size, 1), np.float32)[:pos_size]
            matching_bs[i] = bs
            matching_as[i] = as_
            matching_gjs[i] = gjs
            matching_gis[i] = gis
            matching_targets[i] = tgts
            matching_anchs[i] = anchs
            masks[i] = mask

        matching_bs = np.stack(matching_bs, 0)
        matching_as = np.stack(matching_as, 0)
        matching_gjs = np.stack(matching_gjs, 0)
        matching_gis = np.stack(matching_gis, 0)
        matching_targets = np.stack(matching_targets, 0)
        matching_anchs = np.stack(matching_anchs, 0)
        masks = np.stack(masks, 0)
        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs, masks

    def call(self, fuse_p, targets, imgs, w_list, h_list):
        p = []
        last_idx = 0
        B, A, _, C = fuse_p.shape
        for w, h in zip(w_list, h_list):
            p.append(fuse_p[:, :, last_idx:w*h+last_idx, :].reshape((B, A, h, w, C)))
            last_idx += w*h
        batch_size, img_h = imgs.shape[0], imgs.shape[2]
        if not self.use_aux:
            return self.build_targets(p, targets, batch_size, img_h, min_topk=10, g=self.bias)
        bs_aux, as_aux, gjs_aux, gis_aux, targets_aux, anchors_aux, mask_aux = self.build_targets(
            p, targets, batch_size, img_h, min_topk=20, g=self.bias * 2)
        bs, as_, gjs, gis, targets, anchors, mask = self.build_targets(p, targets, batch_size, img_h,
                                                                       min_topk=10, g=self.bias)
        return bs, as_, gjs, gis, targets, anchors, mask, \
               bs_aux, as_aux, gjs_aux, gis_aux, targets_aux, anchors_aux, mask_aux


class YOLOv7LabelAssignment(nn.Cell):
    def __init__(self, anchors, na=3, bias=0.5, stride=[8, 16, 32], anchor_t=4, use_aux=False, thread_num=4):
        super(YOLOv7LabelAssignment, self).__init__()
        label_assignment = YOLOv7LabelAssignmentNp(anchors, na, bias, stride, anchor_t, use_aux, thread_num)
        self.nl = len(anchors)
        self.anchor_t = anchor_t

        def infer_shape(p, targets, imgs, w_list, h_list):
            batch_size = p[0]
            max_gt = batch_size * 100
            out_shapes = ((self.nl, max_gt,), (self.nl, max_gt,), (self.nl, max_gt,), (self.nl, max_gt,),
                          (self.nl, max_gt, 6), (self.nl, max_gt, 2), (self.nl, max_gt, 1))
            if use_aux:
                max_gt = batch_size * 200
                out_shapes += ((self.nl, max_gt,), (self.nl, max_gt,), (self.nl, max_gt,), (self.nl, max_gt,),
                               (self.nl, max_gt, 6), (self.nl, max_gt, 2), (self.nl, max_gt, 1))
            return out_shapes

        def infer_type(p, targets, imgs, w_list, h_list):
            out_types = (ms.int32, ms.int32, ms.int32, ms.int32, ms.float32, ms.float32, ms.float32)
            if use_aux:
                out_types += (ms.int32, ms.int32, ms.int32, ms.int32, ms.float32, ms.float32, ms.float32)
            return out_types

        def bprob(p, targets, imgs, w_list, h_list, out, dout):
            return ops.zeros_like(p), ops.zeros_like(targets), ops.zeros_like(imgs),\
                   ops.zeros_like(w_list), ops.zeros_like(h_list)

        self.run_op = ops.Custom(label_assignment.call, out_shape=infer_shape, out_dtype=infer_type,
                                 func_type="pyfunc", bprop=bprob).add_prim_attr('primitive_target', 'CPU')

    def stop_gradient(self, inputs):
        res = ()
        for inp in inputs:
            res += (ops.stop_gradient(inp),)
        return res

    def construct(self, p, targets, imgs):
        w_list = []
        h_list = []
        fuse_p = []
        for pp in p:
            B, A, H, W, C = pp.shape
            fuse_p.append(pp.reshape((B, A, -1, C)))
            w_list.append(W)
            h_list.append(H)
        fuse_p = ops.concat(fuse_p, 2)
        return self.stop_gradient(self.run_op(fuse_p, targets, imgs, list2tensor(w_list), list2tensor(h_list)))
