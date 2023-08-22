import mindspore as ms
from mindspore import Tensor, nn, ops


class YOLOv4Head(nn.Cell):
    """
    YOLOv4 Detect Head, convert the output result to a prediction box based on the anchor point.
    """

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(YOLOv4Head, self).__init__()

        assert isinstance(anchors, (tuple, list)) and len(anchors) > 0
        assert isinstance(ch, (tuple, list)) and len(ch) > 0

        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = 3  # number of detection layers
        self.na = len(anchors) // 3  # number of anchors

        self.m = nn.CellList(
            [nn.Conv2d(x, self.no * self.na, 1, pad_mode="valid", has_bias=True) for x in ch]
        )  # output conv

        # prediction on the default anchor boxes
        self.detect_1 = DetectionBlock("l", anchors, self.no)
        self.detect_2 = DetectionBlock("m", anchors, self.no)
        self.detect_3 = DetectionBlock("s", anchors, self.no)

    def construct(self, x):
        big_object_output = self.m[0](x[0])
        medium_object_output = self.m[1](x[1])
        small_object_output = self.m[2](x[2])
        bs = small_object_output.shape[0]
        output_big = self.detect_1(big_object_output)
        output_me = self.detect_2(medium_object_output)
        output_small = self.detect_3(small_object_output)
        if not self.training:
            big = output_big.view(bs, -1, self.no)
            me = output_me.view(bs, -1, self.no)
            small = output_small.view(bs, -1, self.no)
            return ops.concat((big, me, small), 1), (output_big, output_me, output_small)

        return output_big, output_me, output_small


class DetectionBlock(nn.Cell):
    """
    YOLOv4 detection Network. It will finally output the detection result.
    """

    def __init__(self, scale, anchor_scales, no):
        super(DetectionBlock, self).__init__()
        if scale == "s":
            idx = (6, 7, 8)
            self.scale_x_y = 1.2
            self.offset_x_y = 0.1
            self.stride = 8
        elif scale == "m":
            idx = (3, 4, 5)
            self.scale_x_y = 1.1
            self.offset_x_y = 0.05
            self.stride = 16
        elif scale == "l":
            idx = (0, 1, 2)
            self.scale_x_y = 1.05
            self.offset_x_y = 0.025
            self.stride = 32
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = Tensor([anchor_scales[i] for i in idx], ms.float32)
        self.num_anchors_per_scale = 3
        self.num_attrib = no

        self.sigmoid = ops.Sigmoid()

    def construct(self, x):
        """construct method"""
        num_batch = x.shape[0]
        grid_size = x.shape[2:4]
        input_shape = [size * self.stride for size in grid_size]
        input_shape = Tensor(tuple(input_shape[::-1]), ms.float32)

        # Reshape and transpose the feature to [n, grid_size[0], grid_size[1], 3, num_attrib]
        prediction = x.view(num_batch, self.num_anchors_per_scale, self.num_attrib, grid_size[0], grid_size[1])
        prediction = prediction.transpose((0, 3, 4, 1, 2))

        range_x = range(grid_size[1])
        range_y = range(grid_size[0])
        grid_x = ops.cast(ops.tuple_to_array(range_x), ms.float32)
        grid_y = ops.cast(ops.tuple_to_array(range_y), ms.float32)
        # Tensor of shape [grid_size[0], grid_size[1], 1, 1] representing the coordinate of x/y axis for each grid
        # [batch, gridx, gridy, 1, 1]
        grid_x = ops.tile(grid_x.view(1, 1, -1, 1, 1), (1, grid_size[0], 1, 1, 1))
        grid_y = ops.tile(grid_y.view(1, -1, 1, 1, 1), (1, 1, grid_size[1], 1, 1))
        # Shape is [grid_size[0], grid_size[1], 1, 2]
        grid = ops.concat((grid_x, grid_y), -1)

        box_xy = prediction[:, :, :, :, :2]
        box_wh = prediction[:, :, :, :, 2:4]
        box_confidence = prediction[:, :, :, :, 4:5]
        box_probs = prediction[:, :, :, :, 5:]

        # gridsize1 is x
        # gridsize0 is y
        box_xy = (self.scale_x_y * self.sigmoid(box_xy) - self.offset_x_y + grid) / ops.cast(
            ops.tuple_to_array((grid_size[1], grid_size[0])), ms.float32
        )
        # box_wh is w->h
        box_wh = ops.exp(box_wh) * self.anchors / input_shape
        box_confidence = self.sigmoid(box_confidence)
        box_probs = self.sigmoid(box_probs)

        if self.training:
            return prediction, box_xy, box_wh
        box_xy *= input_shape
        box_wh *= input_shape
        return ops.concat((box_xy.astype(ms.float32),
                           box_wh.astype(ms.float32),
                           box_confidence.astype(ms.float32),
                           box_probs.astype(ms.float32)), -1)
