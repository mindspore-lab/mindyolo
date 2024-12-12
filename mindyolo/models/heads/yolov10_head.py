import math
import numpy as np
from copy import deepcopy

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Parameter, Tensor, nn, ops

from ..layers import DFL, ConvNormAct, Identity
from ..layers.utils import meshgrid

class YOLOv10Head(nn.Cell):
    # YOLOv10 Detect head for detection models
    def __init__(self, nc=80, reg_max=16, stride=(), ch=(), sync_bn=False):  # detection layer
        super().__init__()
        # self.dynamic = False # force grid reconstruction

        assert isinstance(stride, (tuple, list)) and len(stride) > 0
        assert isinstance(ch, (tuple, list)) and len(ch) > 0

        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = Parameter(Tensor(stride, ms.int32), requires_grad=False)
        self.max_det = 300  # max_det

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.CellList(
            [
                nn.SequentialCell(
                    [
                        ConvNormAct(x, c2, 3, sync_bn=sync_bn),
                        ConvNormAct(c2, c2, 3, sync_bn=sync_bn),
                        nn.Conv2d(c2, 4 * self.reg_max, 1, has_bias=True),
                    ]
                )
                for x in ch
            ]
        )
        self.cv3 = nn.CellList(
            [
                nn.SequentialCell(
                    [
                        nn.SequentialCell(
                            [
                                ConvNormAct(x, x, 3, g=x),
                                ConvNormAct(x, c3, 1)
                            ]
                        ),
                        nn.SequentialCell([
                                ConvNormAct(c3, c3, 3, g=c3),
                                ConvNormAct(c3, c3, 1)
                            ]
                        ),
                        nn.Conv2d(c3, self.nc, 1, has_bias=True)
                    ]
                )
                for i, x in enumerate(ch)
            ]
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else Identity()

        self.one2one_cv2 = deepcopy(self.cv2)
        self.one2one_cv3 = deepcopy(self.cv3)

    def construct(self, x):
        """
        Performs forward pass of the YOLOv10Head module. Returns predicted bounding boxes and class probabilities

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [ops.stop_gradient(xi) for xi in x]
        one2one = ()
        for i in range(self.nl):
            one2one += (ops.concat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1),)
        one2many = ()
        for i in range(self.nl):
            one2many += (ops.concat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1), )
        if self.training:  # Training path
            return (one2many, one2one)
        y = self._inference(one2one)
        y = self.postprocess(ops.transpose(y, (0, 2, 1)), self.max_det, self.nc)
        return (y, (one2many, one2one))
    
    def _inference(self, x):
        # Inference path
        shape = x[0].shape  # BCHW
        _anchors, _strides = self.make_anchors(x, self.stride, 0.5)
        _anchors, _strides = _anchors.swapaxes(0, 1), _strides.swapaxes(0, 1)
        
        _x = ()
        for i in range(len(x)):
            _x += (x[i].view(shape[0], self.no, -1),)
        _x = ops.concat(_x, 2)
        box, cls = _x[:, : self.reg_max * 4, :], _x[:, self.reg_max * 4 : self.reg_max * 4 + self.nc, :]
        # box, cls = ops.concat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = self.dist2bbox(self.dfl(box), ops.expand_dims(_anchors, 0), xywh=False, axis=1) * _strides
        
        return ops.concat((dbox, ops.Sigmoid()(cls)), 1)

    @staticmethod
    def make_anchors(feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = (), ()
        dtype = feats[0].dtype
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = mnp.arange(w, dtype=dtype) + grid_cell_offset  # shift x
            sy = mnp.arange(h, dtype=dtype) + grid_cell_offset  # shift y
            # FIXME: Not supported on a specific model of machine
            sy, sx = meshgrid((sy, sx), indexing="ij")
            anchor_points += (ops.stack((sx, sy), -1).view(-1, 2),)
            stride_tensor += (ops.ones((h * w, 1), dtype) * stride,)
        return ops.concat(anchor_points), ops.concat(stride_tensor)
    
    @staticmethod
    def dist2bbox(distance, anchor_points, xywh=True, axis=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        lt, rb = ops.split(distance, split_size_or_sections=2, axis=axis)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return ops.concat((c_xy, wh), axis)  # xywh bbox
        return ops.concat((x1y1, x2y2), axis)  # xyxy bbox

    @staticmethod
    def postprocess(preds, max_det, nc=80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, _, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], axis=-1)
        max_scores = ops.amax(scores, axis=-1)
        max_scores, index = ops.topk(max_scores, max_det, dim=-1)
        index = ops.expand_dims(index, -1)
        boxes = ops.gather_elements(boxes, dim=1, index=ops.tile(index, (1, 1, boxes.shape[-1])))
        scores = ops.gather_elements(scores, dim=1, index=ops.tile(index,(1, 1, nc)))

        scores, index = ops.topk(ops.flatten(scores, start_dim=1), max_det, dim=-1)
        i = ops.arange(batch_size)[..., None]  # batch indices
        return ops.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], -1)
    
    def initialize_biases(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            s = s.asnumpy()
            a[-1].bias = ops.assign(a[-1].bias, Tensor(np.ones(a[-1].bias.shape), ms.float32))
            b_np = b[-1].bias.data.asnumpy()
            b_np[: m.nc] = math.log(5 / m.nc / (640 / int(s)) ** 2)
            b[-1].bias = ops.assign(b[-1].bias, Tensor(b_np, ms.float32))
        for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
            s = s.asnumpy()
            a[-1].bias = ops.assign(a[-1].bias, Tensor(np.ones(a[-1].bias.shape), ms.float32))
            b_np = b[-1].bias.data.asnumpy()
            b_np[: m.nc] = math.log(5 / m.nc / (640 / int(s)) ** 2)
            b[-1].bias = ops.assign(b[-1].bias, Tensor(b_np, ms.float32))