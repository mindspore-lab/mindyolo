import sys

sys.path.append(".")

import os
import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor, ops
import mindspore.dataset as de
from mindspore.amp import StaticLossScaler

from mindyolo.utils.config import load_config, Config
from mindyolo.utils import logger
from mindyolo.models import create_loss, create_model
from mindyolo.optim import create_optimizer
from mindyolo.utils.train_step_factory import create_train_step_fn
from mindyolo.utils.trainer_factory import create_trainer


@pytest.mark.parametrize("yaml_name", ['yolov7-tiny.yaml'])
@pytest.mark.parametrize("mode", [0])
def test_create_trainer(yaml_name, mode):
    ms.set_context(mode=mode)
    parent_dir = yaml_name[:6]
    yaml_path = os.path.join('./configs', parent_dir, yaml_name)
    cfg, _, _ = load_config(yaml_path)

    cfg = Config(cfg)

    logger.info("parse_cfg:")
    logger.info("\n" + str(cfg))
    logger.info("Please check the above information for the configurations")

    # Create Network
    nc = 16
    network = create_model(
        model_name=cfg.network.model_name,
        model_cfg=cfg.network,
        num_classes=nc,
    )
    network.set_train(True)

    # Create Dataloaders
    bs = 6
    # create data
    x = np.random.randn(bs, 3, 32, 32).astype(np.float32)
    y = np.random.rand(bs, 160, 6).astype(np.float32)
    y[:, :, 1] *= nc
    for i, l in enumerate(y):
        l[:, 0] = i  # add target image index for build_targets()

    data = (x, y)
    dataset = de.NumpySlicesDataset(data=data, column_names=["images", "labels"])
    dataset = dataset.batch(batch_size=bs)
    dataloader = dataset.repeat(10)

    # Create Loss
    loss_fn = create_loss(
        **cfg.loss, anchors=cfg.network.get("anchors", 1), stride=cfg.network.stride, nc=nc
    )
    
    x = Tensor(x, ms.float32)
    y = Tensor(y, ms.float32)
    output = network(x)
    begin_loss, _ = loss_fn(output, y, x)
    begin_loss = ops.stop_gradient(begin_loss)

    # Create Optimizer
    optimizer = create_optimizer(params=network.trainable_params(), lr=0.001, optimizer='momentum', weight_decay=1e-7)

    # Create train_step_fn
    scaler = StaticLossScaler(1.0)
    train_step_fn = create_train_step_fn(
        task="detect",
        network=network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        loss_ratio=1,
        scaler=scaler,
        reducer=ops.functional.identity,
        overflow_still_update=True,
    )

    # Create Trainer
    network.set_train(True)
    optimizer.set_train(True)
    model_name = yaml_name[:-5]  # delete ".yaml"
    trainer = create_trainer(
        model_name=model_name,
        train_step_fn=train_step_fn,
        scaler=scaler,
        dataloader=dataloader,
        steps_per_epoch=1,
        network=network,
        ema=None,
        optimizer=optimizer,
        loss_fn=loss_fn,
        callback=[],
        reducer=ops.functional.identity,
        data_sink=False,
        profiler=False
    )

    trainer.train(
        epochs=10,
        main_device=True,
        overflow_still_update=True,
        loss_item_name=[] if not hasattr(loss_fn, "loss_item_name") else loss_fn.loss_item_name,      
    )
    logger.info("Training completed.")

    output = network(x)
    cur_loss, _ = loss_fn(output, y, x)
    cur_loss = ops.stop_gradient(cur_loss)

    assert cur_loss < begin_loss, "Loss does NOT decrease"


if __name__ == '__main__':
    test_create_trainer()
    