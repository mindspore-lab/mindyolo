import math
from mindspore import ops


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


def meshgrid(inputs, indexing='xy'):
    # An alternative implementation of ops.meshgrid, Only supports inputs with a length of 2.
    # Meshgrid op is not supported on a specific model of machine an alternative
    # solution is adopted, which will be updated later.
    x, y = inputs
    nx, ny = x.shape[0], y.shape[0]
    xv, yv = None, None
    if indexing == 'xy':
        xv = ops.tile(x.view(1, -1), (ny, 1))
        yv = ops.tile(y.view(-1, 1), (1, nx))
    elif indexing == 'ij':
        xv = ops.tile(x.view(-1, 1), (1, ny))
        yv = ops.tile(y.view(1, -1), (nx, 1))

    return xv, yv
