import numpy as np

import mindspore as ms
from mindspore import ops, Tensor

from fused_op import fused_get_ciou, fused_get_center_dist, fused_get_iou, \
    fused_get_convex_diagonal_squared, fused_get_ciou_diagonal_angle, \
    fused_get_boundding_boxes_coord, fused_get_intersection_area


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
        b1_x1, b1_x2, b1_y1, b1_y2,b2_x1, b2_x2, b2_y1, b2_y2=fused_get_boundding_boxes_coord(x1, y1, w1, h1,x2, y2, w2, h2)
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = ops.split(box1, split_size_or_sections=1, axis=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = ops.split(box2, split_size_or_sections=1, axis=-1)

    # Intersection area
    inter = fused_get_intersection_area(b1_x1, b1_x2, b2_x1, b2_x2, b1_y1, b1_y2, b2_y1, b2_y2)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    iou, union = fused_get_iou(w1, h1, w2, h2, inter)

    if CIoU or DIoU or GIoU:
        cw = ops.maximum(b1_x2, b2_x2) - ops.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = ops.maximum(b1_y2, b2_y2) - ops.minimum(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = fused_get_convex_diagonal_squared(b1_x1, b1_x2, b2_x1, b2_x2, b1_y1, b1_y2, b2_y1, b2_y2)
            rho2 = fused_get_center_dist(b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2)
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = fused_get_ciou_diagonal_angle(w1, h1, w2, h2)
                _, res = fused_get_ciou(v, iou, rho2, c2)
                return res
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

if __name__ =="__main__":
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_device("GPU")
    box1 = Tensor(np.random.rand(32, 4).astype(np.float32))
    box2 = Tensor(np.random.rand(32, 4).astype(np.float32))
    iou = bbox_iou(box1, box2, xywh=True, CIoU=True)
