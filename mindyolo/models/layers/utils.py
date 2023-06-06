import math
from typing import Tuple

from mindspore import Tensor, ops


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    if isinstance(p, list):
        assert len(p) == 2
        p = (p[0], p[0], p[1], p[1])
    return p


# ------------------------box operation starts--------------------------
def meshgrid(inputs, indexing="xy"):
    # An alternative implementation of ops.meshgrid, Only supports inputs with a length of 2.
    # Meshgrid op is not supported on a specific model of machine an alternative
    # solution is adopted, which will be updated later.
    x, y = inputs
    nx, ny = x.shape[0], y.shape[0]
    xv, yv = None, None
    if indexing == "xy":
        xv = ops.tile(x.view(1, -1), (ny, 1))
        yv = ops.tile(y.view(-1, 1), (1, nx))
    elif indexing == "ij":
        xv = ops.tile(x.view(-1, 1), (1, ny))
        yv = ops.tile(y.view(1, -1), (nx, 1))

    return xv, yv


def box_cxcywh_to_xyxy(bbox) -> Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2)

    Args:
        bbox (ms.Tensor): Shape (n, 4) for bboxes.

    Returns:
        torch.Tensor: Converted bboxes.
    """
    cx, cy, w, h = ops.unstack(bbox, axis=-1)
    new_bbox = tuple([(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)])
    aa = ops.stack(new_bbox, axis=-1)
    return aa


def box_xyxy_to_cxcywh(bbox) -> Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h)

    Args:
        bbox (torch.Tensor): Shape (n, 4) for bboxes.

    Returns:
        torch.Tensor: Converted bboxes.
    """
    x0, y0, x1, y1 = ops.unstack(bbox, axis=-1)
    new_bbox = tuple([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)])
    return ops.stack(new_bbox, axis=-1)


def box_scale(boxes, scale, sclale_reciprocal=False) -> Tensor:
    """
    Scale the box with horizontal and vertical scaling factors

    Args:
        boxes (Tensor[N, 4] or [bs, N, 4]): boxes are specified by their (x1, y1, x2, y2) coordinates
        scale (Tuple[2]): scale factors for x and y coordinates
    """
    assert len(boxes.shape) in [2, 3]
    scale_x, scale_y = scale
    if sclale_reciprocal:
        scale_x, scale_y = 1.0 / scale_x, 1.0 / scale_y
    new_scale = Tensor([scale_x, scale_y, scale_x, scale_y])  # (4,) or (bs, 4)
    boxes *= new_scale
    return boxes


def box_clip(boxes, clip_size: Tuple[int, int]) -> Tensor:
    """
    Clip (in place) the boxes by limiting x coordinates to the range [0, width]
    and y coordinates to the range [0, height].

    Args:
        boxes (Tensor[N, 4]): boxes are specified by their (x1, y1, x2, y2) coordinates
        clip_size (height, width): The clipping box's size.
    """
    h, w = clip_size
    x1 = boxes[..., 0].clip(0, w)
    y1 = boxes[..., 1].clip(0, h)
    x2 = boxes[..., 2].clip(0, w)
    y2 = boxes[..., 3].clip(0, h)
    boxes = ops.stack((x1, y1, x2, y2), axis=-1)
    return boxes


# ------------------------box operation ends--------------------------
