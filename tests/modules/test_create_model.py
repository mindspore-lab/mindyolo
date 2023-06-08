import sys

sys.path.append(".")

import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor, nn

from mindyolo.models import create_model
from mindyolo.models.registry import register_model
from mindyolo.models.model_factory import build_model_from_cfg
from mindyolo.utils.config import Config
from mindyolo.models.layers import *


model_dict = {
    'stride': [], 'depth_multiple': 1.0, 'width_multiple': 1.0,
    'backbone':
        [
            [-1, 1, ConvNormAct, [32, 3, 2]],
            [-1, 2, Bottleneck, [32]]
        ],
    'head':
        [
            [-1, 1, ConvNormAct, [64, 1, 1]],
            [-1, 1, Bottleneck, [64, False]],
        ]
}

model_cfg = Config(model_dict)


@register_model
class SimpleCNN(nn.Cell):
    def __init__(self, cfg, in_channels=3, num_classes=80, sync_bn=False):
        super(SimpleCNN, self).__init__()

        self.model = build_model_from_cfg(model_cfg=cfg, in_channels=in_channels, num_classes=num_classes, sync_bn=sync_bn)

    def construct(self, x):
        return self.model(x)

            
@pytest.mark.parametrize("mode", [0, 1])
def test_create_model(mode):
    ms.set_context(mode=mode)

    bs = 2
    model = create_model(
        model_name='SimpleCNN',
        model_cfg=model_cfg,
    )
    model.set_train(True)
    input_size = (bs, 3, 640, 640)
    dummy_input = Tensor(np.random.rand(*input_size), dtype=ms.float32)
    output = model(dummy_input)
    assert output.shape == (bs, 64, 320, 320), "output shape not match"


if __name__ == '__main__':
    test_create_model()
