import sys

sys.path.append(".")

import os
import pytest

from mindyolo.models import create_model
from mindyolo.utils.config import load_config, Config
from mindyolo.optim import create_group_param

yaml_name_list = ['yolov3.yaml', 'yolov4.yaml', 'yolov5n.yaml', 'yolov7-tiny.yaml', 'yolov8n.yaml', 'yolox-nano.yaml']


@pytest.mark.parametrize("yaml_name", yaml_name_list)
def test_create_group_param(yaml_name):
    parent_dir = yaml_name[:6] if yaml_name != 'yolox-nano.yaml' else 'yolox'
    yaml_path = os.path.join('./configs', parent_dir, yaml_name)
    cfg, _, _ = load_config(yaml_path)

    cfg = Config(cfg)

    cfg.optimizer.epochs = 300
    cfg.optimizer.accumulate = 1
    cfg.optimizer.total_batch_size = 128
    cfg.optimizer.steps_per_epoch = 924
    cfg.optimizer.min_warmup_step = 1000
    nc = 10
    model = create_model(
        model_name=cfg.network.model_name,
        model_cfg=cfg.network,
        num_classes=nc
    )
    model.set_train(True)
    params = create_group_param(params=model.trainable_params(), **cfg.optimizer)
    assert params is not None


if __name__ == '__main__':
    test_create_group_param()
