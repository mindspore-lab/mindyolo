import os
import ast
import argparse
from functools import partial

import mindspore as ms

from mindyolo.data import COCODataset, create_loader
from mindyolo.models import create_loss, create_model
from mindyolo.optim import create_group_param, create_lr_scheduler, create_warmup_momentum_scheduler, \
    create_optimizer, EMA
from mindyolo.utils import logger
from mindyolo.utils.config import parse_args
from mindyolo.utils.utils import set_seed, set_default, load_pretrain, freeze_layers
from mindyolo.utils.train_step_factory import get_gradreducer, get_loss_scaler, create_train_step_fn
from mindyolo.utils.trainer_factory import create_trainer

from test import test


def get_parser_train(parents=None):
    parser = argparse.ArgumentParser(description='Train', parents=[parents] if parents else [])
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--save_dir', type=str, default='./runs', help='save dir')
    parser.add_argument('--device_per_servers', type=int, default=8, help='device number on a server')
    parser.add_argument('--log_level', type=str, default='INFO', help='save dir')
    parser.add_argument('--is_parallel', type=ast.literal_eval, default=False, help='Distribute train or not')
    parser.add_argument('--ms_mode', type=int, default=0, help='Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)')
    parser.add_argument('--ms_amp_level', type=str, default='O0', help='amp level, O0/O1/O2/O3')
    parser.add_argument('--keep_loss_fp32', type=ast.literal_eval, default=True, help='Whether to maintain loss using fp32/O0-level calculation')
    parser.add_argument('--ms_loss_scaler', type=str, default='static', help='train loss scaler, static/dynamic/none')
    parser.add_argument('--ms_loss_scaler_value', type=float, default=1024.0, help='static loss scale value')
    parser.add_argument('--ms_grad_sens', type=float, default=1024.0, help='gard sens')
    parser.add_argument('--ms_jit', type=ast.literal_eval, default=True, help='use jit or not')
    parser.add_argument('--ms_enable_graph_kernel', type=ast.literal_eval, default=False, help='use enable_graph_kernel or not')
    parser.add_argument('--overflow_still_update', type=ast.literal_eval, default=True, help='overflow still update')
    parser.add_argument('--ema', type=ast.literal_eval, default=True, help='ema')
    parser.add_argument('--weight', type=str, default='', help='initial weight path')
    parser.add_argument('--ema_weight', type=str, default='', help='initial ema weight path')
    parser.add_argument('--freeze', type=list, default=[],
                        help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--epochs', type=int, default=300, help="total train epochs")
    parser.add_argument('--per_batch_size', type=int, default=32, help='per batch size for each device')
    parser.add_argument('--img_size', type=list, default=640, help='train image sizes')
    parser.add_argument('--nbs', type=list, default=64, help='nbs')
    parser.add_argument('--accumulate', type=int, default=1,
                        help='grad accumulate step, recommended when batch-size is less than 64')
    parser.add_argument('--auto_accumulate', type=ast.literal_eval, default=False, help='auto accumulate')
    parser.add_argument('--log_interval', type=int, default=100, help='log interval')
    parser.add_argument('--single_cls', type=ast.literal_eval, default=False,
                        help='train multi-class data as single-class')
    parser.add_argument('--sync_bn', type=ast.literal_eval, default=False,
                        help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--keep_checkpoint_max', type=int, default=100)
    parser.add_argument('--run_eval', type=ast.literal_eval, default=False,
                        help='Whether to run eval during training')
    parser.add_argument('--rect', type=ast.literal_eval, default=False, help='rectangular training')
    parser.add_argument('--nms_time_limit', type=float, default=20.0, help='time limit for NMS')
    parser.add_argument('--recompute', type=ast.literal_eval, default=False, help='Recompute')
    parser.add_argument('--recompute_layers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2, help='set global seed')

    # args for ModelArts
    parser.add_argument('--enable_modelarts', type=ast.literal_eval, default=False, help='enable modelarts')
    parser.add_argument('--data_url', type=str, default='', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--ckpt_url', type=str, default='', help='ModelArts: obs path to checkpoint folder')
    parser.add_argument('--train_url', type=str, default='', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--data_dir', type=str, default='/cache/data/',
                        help='ModelArts: local device path to dataset folder')
    parser.add_argument('--ckpt_dir', type=str, default='/cache/pretrain_ckpt/',
                        help='ModelArts: local device path to checkpoint folder')
    return parser


def train(args):
    # Set Default
    set_seed(args.seed)
    set_default(args)
    main_device = (args.rank % args.rank_size == 0)

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

    # Create Dataloader
    dataset = COCODataset(
        dataset_path=args.data.train_set,
        img_size=args.img_size,
        transforms_dict=args.data.train_transforms,
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
        epoch_size=1,
        rank=args.rank,
        rank_size=args.rank_size,
        shuffle=True,
        drop_remainder=True,
        num_parallel_workers=args.data.num_parallel_workers,
        python_multiprocessing=True
    )
    steps_per_epoch = dataloader.get_dataset_size()

    if args.run_eval:
        eval_dataset = COCODataset(
            dataset_path=args.data.val_set,
            img_size=args.img_size,
            transforms_dict=args.data.test_transforms,
            is_training=False, augment=False, rect=args.rect, single_cls=args.single_cls,
            batch_size=args.per_batch_size * 2, stride=max(args.network.stride),
        )
        eval_dataloader = create_loader(
            dataset=eval_dataset,
            batch_collate_fn=eval_dataset.test_collate_fn,
            dataset_column_names=eval_dataset.dataset_column_names,
            batch_size=args.per_batch_size * 2,
            epoch_size=1, rank=0, rank_size=1, shuffle=False, drop_remainder=False,
            num_parallel_workers=args.data.num_parallel_workers,
            python_multiprocessing=True
        )
    else:
        eval_dataset, eval_dataloader = None, None

    # Create Loss
    loss_fn = create_loss(
        **args.loss,
        anchors=args.network.anchors,
        stride=args.network.stride,
        nc=args.data.nc
    )
    ms.amp.auto_mixed_precision(loss_fn, amp_level='O0' if args.keep_loss_fp32 else args.ms_amp_level)

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
        network=network, loss_fn=loss_fn, optimizer=optimizer,
        loss_ratio=args.rank_size, scaler=scaler, reducer=reducer,
        overflow_still_update=args.overflow_still_update, ms_jit=args.ms_jit
    )

    # Create test function for run eval while train
    if args.run_eval:
        is_coco_dataset = ('coco' in args.data.dataset_name)
        test_fn = partial(
            test,
            dataloader=eval_dataloader,
            anno_json_path=os.path.join(args.data.val_set[:-len(args.data.val_set.split('/')[-1])],
                                        'annotations/instances_val2017.json'),
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            nms_time_limit=args.nms_time_limit,
            is_coco_dataset=is_coco_dataset,
            imgIds=None if not is_coco_dataset else eval_dataset.imgIds,
            per_batch_size=args.per_batch_size
        )
    else:
        test_fn = None

    # Create Trainer
    network.set_train(True)
    optimizer.set_train(True)
    model_name = os.path.basename(args.config)[:-5]  # delete ".yaml"
    trainer = create_trainer(
        model_name=model_name,
        train_step_fn=train_step_fn, scaler=scaler,
        dataloader=dataloader,
        network=network, ema=ema, optimizer=optimizer,
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
        loss_item_name=[] if not hasattr(loss_fn, 'loss_item_name') else loss_fn.loss_item_name,
        save_dir=args.save_dir,
        enable_modelarts=args.enable_modelarts,
        train_url=args.train_url,
        run_eval=args.run_eval,
        test_fn=test_fn,
    )
    logger.info('Training completed.')


if __name__ == '__main__':
    parser = get_parser_train()
    args = parse_args(parser)
    train(args)
