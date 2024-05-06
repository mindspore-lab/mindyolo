import sys

sys.path.append(".")

import os
import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor, ops

from mindyolo.models import create_loss
from mindyolo.utils.config import load_config, Config

yaml_name_list = ['yolov3.yaml', 'yolov4.yaml', 'yolov5n.yaml', 'yolov7-tiny.yaml', 'yolov8n.yaml', 'yolox-nano.yaml']


@pytest.mark.parametrize("yaml_name", yaml_name_list)
@pytest.mark.parametrize("mode", [0, 1])
def test_create_loss(yaml_name, mode):
    ms.set_context(mode=mode)

    parent_dir = yaml_name[:6] if yaml_name != 'yolox-nano.yaml' else 'yolox'
    yaml_path = os.path.join('./configs', parent_dir, yaml_name)
    cfg, _, _ = load_config(yaml_path)

    cfg = Config(cfg)

    bs = 2
    nc = 16

    loss_fn = create_loss(
        **cfg.loss, anchors=cfg.network.get("anchors", 1), stride=cfg.network.stride, nc=nc
    )

    input_size = cfg.img_size
    stride = cfg.network.stride
    if input_size:
        input_size = (bs, 3) + tuple([input_size] * 2)
    else:
        input_size = (bs, 3, 224, 224)

    imgs = Tensor(np.random.rand(*input_size), dtype=ms.float32)
    if yaml_name == 'yolox-nano.yaml':
        box_num = 0
        for s in stride:
            box_num += (input_size[2] // s) * (input_size[3] // s)
        outputs = Tensor(np.random.rand(bs, box_num, nc + 5), dtype=ms.float32)
    elif yaml_name == 'yolov4.yaml':
        outputs = ()
        for s in stride:
            output = ()
            output += (Tensor(np.random.rand(bs, input_size[2] // s, input_size[3] // s, 3, nc + 5), dtype=ms.float32),)
            output += (Tensor(np.random.rand(bs, input_size[2] // s, input_size[3] // s, 3, 2), dtype=ms.float32),)
            output += (Tensor(np.random.rand(bs, input_size[2] // s, input_size[3] // s, 3, 2), dtype=ms.float32),)
            outputs += (output,)
    elif yaml_name == 'yolov8n.yaml':
        outputs = ()
        for s in stride:
            output_size = (bs, 80, input_size[2] // s, input_size[3] // s)
            output = (Tensor(np.random.rand(*output_size), dtype=ms.float32),)
            outputs += output
    else:
        outputs = ()
        for s in stride:
            output_size = (bs, 3, input_size[2] // s, input_size[3] // s, nc + 5)
            output = (Tensor(np.random.rand(*output_size), dtype=ms.float32),)
            outputs += output
    targets = np.random.rand(bs, 160, 6)
    targets[:, :, 1] *= nc
    for i, l in enumerate(targets):
        l[:, 0] = i  # add target image index for build_targets()
    targets = Tensor(targets, ms.float32)

    loss, loss_items = loss_fn(outputs, targets, imgs)
    assert loss.size == 1, "output shape not match"
    assert not ops.isnan(loss) and not ops.isinf(loss), "invalid loss"


if __name__ == '__main__':
    test_create_loss()
