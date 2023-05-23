from mindspore import nn, ops


class Upsample(nn.Cell):
    """
    Using the interpolate method specified by `mode` resize the input tensor.

    Args:
        scales (tuple[float], optional): a tuple of float. Describe the scale along each dimension.
            Its length is the same as that of shape of `x`. The numbers in `scales` must all be positive. Only one of
            `scales` and `sizes` can be specified.
        sizes (tuple[int], optional): a tuple of int, describes the shape of the output tensor. The numbers in `sizes`
            must all be positive. Only one of `scales` and `sizes` can be specified.  If `sizes` is specified, then set
            `scales` to 'None' in this operator's input list. It is 1 int elements :math:`(new\_width,)` when `mode`
            is "linear". It is 2 int elements :math:`(new\_height, new\_width)` when `mode` is "bilinear".
        mode (string): The method used to interpolate: 'linear' | 'bilinear'. Default is 'linear'.
    """

    def __init__(self, sizes=None, scales=None, mode="nearest"):
        super(Upsample, self).__init__()
        self.sizes = sizes
        self.scales = scales
        self.mode = mode

    def construct(self, x):
        if self.mode == "nearest" and self.scales:
            return ops.ResizeNearestNeighbor((x.shape[-2] * self.scales, x.shape[-1] * self.scales))(x)
        else:
            return ops.interpolate(x, sizes=self.sizes, scales=self.scales, mode=self.mode)
