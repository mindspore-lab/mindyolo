import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import mindspore as ms

from train import get_parser_train
from mindyolo.data import COCODataset, create_loader
from mindyolo.models import create_loss, create_model
from mindyolo.optim import (EMA, create_group_param, create_lr_scheduler,
                            create_optimizer, create_warmup_momentum_scheduler)
from mindyolo.utils import logger
from mindyolo.utils.config import parse_args
from mindyolo.utils.train_step_factory import (create_train_step_fn,
                                               get_gradreducer,
                                               get_loss_scaler)
from mindyolo.utils.trainer_factory import create_trainer
from mindyolo.utils.utils import (freeze_layers, load_pretrain, set_default,
                                  set_seed)


def train_shwd(args):
    # Set Default
    set_seed(args.seed)
    set_default(args)
    main_device = args.rank % args.rank_size == 0

    logger.info("parse_args:")
    logger.info("\n" + str(args))
    logger.info("Please check the above information for the configurations")

    # Create Network
    args.network.recompute = args.recompute
    args.network.recompute_layers = args.recompute_layers
    network = create_model(
        model_name=args.network.model_name,
        model_cfg=args.network,
        num_classes=args.data.nc,
        sync_bn=args.sync_bn,
    )

    if args.ema and main_device:
        ema_network = create_model(
            model_name=args.network.model_name,
            model_cfg=args.network,
            num_classes=args.data.nc,
        )
        ema = EMA(network, ema_network)
    else:
        ema = None
    load_pretrain(network, args.weight, ema, args.ema_weight)  # load pretrain
    freeze_layers(network, args.freeze)  # freeze Layers
    ms.amp.auto_mixed_precision(network, amp_level=args.ms_amp_level)
    if ema:
        ms.amp.auto_mixed_precision(ema.ema, amp_level=args.ms_amp_level)

    # Create Dataloaders
    transforms = args.data.train_transforms
    dataset = COCODataset(
        dataset_path=args.data.train_set,
        img_size=args.img_size,
        transforms_dict=transforms,
        is_training=True,
        augment=True,
        rect=args.rect,
        single_cls=args.single_cls,
        batch_size=args.total_batch_size,
        stride=max(args.network.stride),
    )
    dataloader = create_loader(
        dataset=dataset,
        batch_collate_fn=dataset.train_collate_fn,
        dataset_column_names=dataset.dataset_column_names,
        batch_size=args.per_batch_size,
        epoch_size=args.epochs,
        rank=args.rank,
        rank_size=args.rank_size,
        shuffle=True,
        drop_remainder=True,
        num_parallel_workers=args.data.num_parallel_workers,
        python_multiprocessing=True,
    )
    steps_per_epoch = dataloader.get_dataset_size() // args.epochs

    # Create Loss
    loss_fn = create_loss(
        **args.loss, anchors=args.network.get("anchors", 1), stride=args.network.stride, nc=args.data.nc
    )
    ms.amp.auto_mixed_precision(loss_fn, amp_level="O0" if args.keep_loss_fp32 else args.ms_amp_level)

    # Create Optimizer
    args.optimizer.steps_per_epoch = steps_per_epoch
    lr = create_lr_scheduler(**args.optimizer)
    params = create_group_param(params=network.trainable_params(), **args.optimizer)
    optimizer = create_optimizer(params=params, lr=lr, **args.optimizer)
    warmup_momentum = create_warmup_momentum_scheduler(**args.optimizer)

    # Create train_step_fn
    reducer = get_gradreducer(args.is_parallel, optimizer.parameters)
    scaler = get_loss_scaler(args.ms_loss_scaler, scale_value=args.ms_loss_scaler_value)
    train_step_fn = create_train_step_fn(
        network=network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        loss_ratio=args.rank_size,
        scaler=scaler,
        reducer=reducer,
        overflow_still_update=args.overflow_still_update,
        ms_jit=args.ms_jit,
    )

    # Create Trainer
    network.set_train(True)
    optimizer.set_train(True)
    model_name = os.path.basename(args.config)[:-5]  # delete ".yaml"
    trainer = create_trainer(
        model_name=model_name,
        train_step_fn=train_step_fn,
        scaler=scaler,
        dataloader=dataloader,
        steps_per_epoch=steps_per_epoch,
        network=network,
        ema=ema,
        optimizer=optimizer,
        summary=args.summary,
        loss_fn=loss_fn,
        callback=[],
        reducer=reducer
    )

    trainer.train(
        epochs=args.epochs,
        main_device=main_device,
        warmup_step=max(round(args.optimizer.warmup_epochs * steps_per_epoch), args.optimizer.min_warmup_step),
        warmup_momentum=warmup_momentum,
        accumulate=args.accumulate,
        overflow_still_update=args.overflow_still_update,
        keep_checkpoint_max=args.keep_checkpoint_max,
        log_interval=args.log_interval,
        loss_item_name=[] if not hasattr(loss_fn, "loss_item_name") else loss_fn.loss_item_name,
        save_dir=args.save_dir,
        enable_modelarts=args.enable_modelarts,
        train_url=args.train_url,
        run_eval=args.run_eval,
    )
    logger.info("Training completed.")


if __name__ == "__main__":
    parser = get_parser_train()
    args = parse_args(parser)
    train_shwd(args)
