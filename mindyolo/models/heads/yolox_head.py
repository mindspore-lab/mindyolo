import math

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore import numpy as mnp
from mindspore.common import initializer as init

from mindyolo.models.layers.conv import ConvNormAct, DWConvNormAct
from ..layers.utils import meshgrid


class YOLOXHead(nn.Cell):
    def __init__(
        self,
        nc=80,
        stride=(8, 16, 32),
        ch=(256, 512, 1024),
        is_standard_backbone=True,
        act=True,
        depth_wise=False,
        sync_bn=False,
    ):
        """
        YOlOx head
        Args:
            is_standard_backbone: whether the predecessor backbone is a standard one or darknet53. default, True
        """
        super().__init__()
        assert isinstance(stride, (tuple, list)) and len(stride) > 0
        assert isinstance(ch, (tuple, list)) and len(ch) > 0

        self.nc = nc
        self.nl = len(ch)
        self.no = nc + 4 + 1
        self.stride = Tensor(stride, ms.int32)

        self.stems = nn.CellList()  # len = num_layer
        self.cls_convs = nn.CellList()
        self.reg_convs = nn.CellList()
        self.cls_preds = nn.CellList()
        self.reg_preds = nn.CellList()
        self.obj_preds = nn.CellList()

        hidden_ch = ch[2] // 4 if is_standard_backbone else 256
        HeadCNA = DWConvNormAct if depth_wise else ConvNormAct
        for i in range(self.nl):  # three kind of resolution, 80, 40, 20
            self.stems.append(ConvNormAct(ch[i], hidden_ch, 1, act=act, sync_bn=sync_bn))
            self.cls_convs.append(
                nn.SequentialCell(
                    [
                        HeadCNA(hidden_ch, hidden_ch, 3, act=act, sync_bn=sync_bn),
                        HeadCNA(hidden_ch, hidden_ch, 3, act=act, sync_bn=sync_bn),
                    ]
                )
            )
            self.reg_convs.append(
                nn.SequentialCell(
                    [
                        HeadCNA(hidden_ch, hidden_ch, 3, act=act, sync_bn=sync_bn),
                        HeadCNA(hidden_ch, hidden_ch, 3, act=act, sync_bn=sync_bn),
                    ]
                )
            )
            self.cls_preds.append(nn.Conv2d(hidden_ch, self.nc, 1, pad_mode="pad", has_bias=True))
            self.reg_preds.append(nn.Conv2d(hidden_ch, 4, 1, pad_mode="pad", has_bias=True))
            self.obj_preds.append(nn.Conv2d(hidden_ch, 1, 1, pad_mode="pad", has_bias=True))

    def construct(self, feat_list):
        assert isinstance(feat_list, (tuple, list)) and len(feat_list) == self.nl
        outputs = []
        for i in range(self.nl):  # 80, 40, 20
            # Get head features
            x = self.stems[i](feat_list[i])

            cls_feat = self.cls_convs[i](x)
            cls_output = self.cls_preds[i](cls_feat)

            reg_feat = self.reg_convs[i](x)
            reg_output = self.reg_preds[i](reg_feat)
            obj_output = self.obj_preds[i](reg_feat)

            # Convert to origin image scale (640)
            output = (
                ops.concat([reg_output, obj_output, cls_output], 1)
                if self.training
                else ops.concat([reg_output, ops.sigmoid(obj_output), ops.sigmoid(cls_output)], 1)
            )
            output = self.convert_to_origin_scale(output, stride=self.stride[i])
            outputs.append(output)
        outputs_cat = ops.concat(outputs, 1)
        return outputs_cat if self.training else (outputs_cat, 1)

    def initialize_biases(self, prior_prob=1e-2):
        for i in range(self.nl):  # 80, 40, 20
            for cell in [self.cls_preds[i], self.obj_preds[i]]:
                cell.bias.set_data(
                    init.initializer(-math.log((1 - prior_prob) / prior_prob), cell.bias.shape, cell.bias.dtype)
                )

    def convert_to_origin_scale(self, output, stride):
        """map to origin image scale for each fpn"""
        batch_size = ops.shape(output)[0]
        grid_size = ops.shape(output)[2:4]
        stride = ops.cast(stride, output.dtype)

        # reshape predictions
        output = ops.transpose(output, (0, 2, 3, 1))  # (bs,85,80,80)-->(bs, 80, 80, 85)
        output = ops.reshape(output, (batch_size, 1 * grid_size[0] * grid_size[1], -1))  # bs, 6400, 85

        # make grid
        grid = self._make_grid(nx=grid_size[1], ny=grid_size[0], dtype=output.dtype)  # (1,1,80,80,2)
        grid = ops.reshape(grid, (1, -1, 2))  # grid(1, 6400, 2)

        # feature map scale to origin scale
        output_xy = output[..., :2]
        output_xy = (output_xy + grid) * stride
        output_wh = output[..., 2:4]
        output_wh = ops.exp(output_wh) * stride
        output_other = output[..., 4:]
        output_t = ops.concat([output_xy, output_wh, output_other], -1)
        return output_t  # bs, 6400, 85

    @staticmethod
    def _make_grid(nx=20, ny=20, dtype=ms.float32):
        # FIXME: Not supported on a specific model of machine
        xv, yv = meshgrid((mnp.arange(nx), mnp.arange(ny)))
        return ops.cast(ops.stack((xv, yv), 2).view((1, 1, ny, nx, 2)), dtype)
