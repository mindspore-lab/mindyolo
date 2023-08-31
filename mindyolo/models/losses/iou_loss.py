import math

import mindspore as ms
from mindspore import Tensor, ops

from mindyolo.models.layers.utils import box_cxcywh_to_xyxy

PI = Tensor(math.pi, ms.float32)
EPS = 1e-7


def box_area(box):
    """
    Return area of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box (Tensor[N, 4])
    Returns:
        area (Tensor[N,])
    """
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])


def batch_box_area(box):
    """
    Return area of batch boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box (Tensor[B, N, 4])
    Returns:
        area (Tensor[B, N])
    """
    return (box[:, :, 2] - box[:, :, 0]) * (box[:, :, 3] - box[:, :, 1])


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    area1 = box_area(box1)
    area2 = box_area(box2)

    expand_size_1 = box2.shape[0]
    expand_size_2 = box1.shape[0]

    box1 = ops.tile(ops.expand_dims(box1, 1), (1, expand_size_1, 1))
    box2 = ops.tile(ops.expand_dims(box2, 0), (expand_size_2, 1, 1))

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    # inter = ops.minimum(box1[:, None, 2:], box2[None, :, 2:]) - ops.maximum(box1[:, None, :2], box2[None, :, :2])
    inter = ops.minimum(box1[..., 2:], box2[..., 2:]) - ops.maximum(box1[..., :2], box2[..., :2])
    inter = inter.clip(0.0, None)
    inter = inter[:, :, 0] * inter[:, :, 1]
    return inter / (area1[:, None] + area2[None, :] - inter).clip(EPS, None)  # iou = inter / (area1 + area2 - inter)


def batch_box_iou(batch_box1, batch_box2, xywh=False):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[B, N, 4])
        box2 (Tensor[B, M, 4])
    Returns:
        iou (Tensor[B, N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    if xywh:
        batch_box1 = box_cxcywh_to_xyxy(batch_box1)
        batch_box2 = box_cxcywh_to_xyxy(batch_box2)

    area1 = batch_box_area(batch_box1)
    area2 = batch_box_area(batch_box2)

    expand_size_1 = batch_box2.shape[1]
    expand_size_2 = batch_box1.shape[1]
    batch_box1 = ops.tile(ops.expand_dims(batch_box1, 2), (1, 1, expand_size_1, 1))
    batch_box2 = ops.tile(ops.expand_dims(batch_box2, 1), (1, expand_size_2, 1, 1))

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = ops.minimum(batch_box1[..., 2:], batch_box2[..., 2:]) - ops.maximum(
        batch_box1[..., :2], batch_box2[..., :2]
    )
    inter = inter.clip(0.0, None)
    inter = inter[:, :, :, 0] * inter[:, :, :, 1]
    return inter / (area1[:, :, None] + area2[:, None, :] - inter).clip(
        EPS, None
    )  # iou = inter / (area1 + area2 - inter)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Return intersection-over-union (IoU) of boxes.
    Arguments:
        box1 (Tensor[N, 4]) or (Tensor[bs, N, 4])
        box2 (Tensor[N, 4]) or (Tensor[bs, N, 4])
        xywh (bool): Whether the box format is (x_center, y_center, w, h) or (x1, y1, x2, y2). Default: True.
        GIoU (bool): Whether to use GIoU. Default: False.
        DIoU (bool): Whether to use DIoU. Default: False.
        CIoU (bool): Whether to use CIoU. Default: False.
    Returns:
        iou (Tensor[N,]): the IoU values for every element in boxes1 and boxes2
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        x1, y1, w1, h1 = ops.split(box1, split_size_or_sections=1, axis=-1)
        x2, y2, w2, h2 = ops.split(box2, split_size_or_sections=1, axis=-1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = ops.split(box1, split_size_or_sections=1, axis=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = ops.split(box2, split_size_or_sections=1, axis=-1)

    # Intersection area
    inter = (ops.minimum(b1_x2, b2_x2) - ops.maximum(b1_x1, b2_x1)).clip(0., None) * \
            (ops.minimum(b1_y2, b2_y2) - ops.maximum(b1_y1, b2_y1)).clip(0., None)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = ops.maximum(b1_x2, b2_x2) - ops.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = ops.maximum(b1_y2, b2_y2) - ops.minimum(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                # v = (4 / get_pi(iou.dtype) ** 2) * ops.pow(ops.atan(w2 / (h2 + eps)) - ops.atan(w1 / (h1 + eps)), 2)
                v = (4 / PI.astype(iou.dtype) ** 2) * ops.pow(ops.atan(w2 / (h2 + eps)) - ops.atan(w1 / (h1 + eps)), 2)
                alpha = v / (v - iou + (1 + eps))
                alpha = ops.stop_gradient(alpha)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU
