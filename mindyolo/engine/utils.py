import random, os, yaml, glob, re
import numpy as np
from datetime import datetime
from pathlib import Path

import mindspore as ms
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size

from mindyolo.utils import logger

__all__ = ['set_default', 'set_seed']


def set_seed(seed=2):
    np.random.seed(seed)
    random.seed(seed)
    ms.set_seed(seed)


def set_default(cfg):
    # Set Context
    context.set_context(mode=cfg.ms_mode, device_target=cfg.device_target, max_call_depth=2000)
    if cfg.device_target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', 0))
        context.set_context(device_id=device_id)
    elif cfg.device_target == "GPU" and cfg.ms_enable_graph_kernel:
        context.set_context(enable_graph_kernel=True)

    # Set Parallel
    rank, rank_size, parallel_mode = 0, 1, ParallelMode.STAND_ALONE
    if cfg.is_parallel:
        init()
        rank, rank_size, parallel_mode = get_rank(), get_group_size(), ParallelMode.DATA_PARALLEL
    context.set_auto_parallel_context(device_num=rank_size, parallel_mode=parallel_mode, gradients_mean=True)
    cfg.rank, cfg.rank_size = rank, rank_size

    # Set default cfg
    cfg.total_batch_size = cfg.per_batch_size * cfg.rank_size
    cfg.sync_bn = cfg.sync_bn and context.get_context("device_target") == "Ascend" and cfg.rank_size > 1
    cfg.accumulate = max(1, np.round(cfg.nbs / cfg.total_batch_size)) \
        if cfg.auto_accumulate else cfg.accumulate
    # optimizer
    cfg.optimizer.warmup_epochs = cfg.optimizer.get('warmup_epochs', 0)
    cfg.optimizer.min_warmup_step = cfg.optimizer.get('min_warmup_step', 0)
    cfg.optimizer.epochs = cfg.epochs
    cfg.optimizer.nbs = cfg.nbs
    cfg.optimizer.accumulate = cfg.accumulate
    cfg.optimizer.total_batch_size = cfg.total_batch_size
    # data
    cfg.data.nc = 1 if cfg.single_cls else int(cfg.data.nc)  # number of classes
    cfg.data.names = ['item'] if cfg.single_cls and len(cfg.names) != 1 else cfg.data.names  # class names
    # loss
    cfg.loss.loss_item_name = cfg.loss.get('loss_item_name', ['loss', 'lbox', 'lobj', 'lcls'])
    assert len(cfg.data.names) == cfg.data.nc, '%g names found for nc=%g dataset in %s' % \
                                               (len(cfg.data.names), cfg.data.nc, cfg.config)

    # Directories and Save run settings
    cfg.save_dir = os.path.join(cfg.save_dir, datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
    cfg.ckpt_save_dir = os.path.join(cfg.save_dir, 'weights')
    cfg.sync_lock_dir = os.path.join(cfg.save_dir, 'sync_locks') if not cfg.enable_modelarts else '/tmp/sync_locks'
    os.makedirs(cfg.save_dir, exist_ok=True)
    if rank % rank_size == 0:
        os.makedirs(cfg.ckpt_save_dir, exist_ok=True)
        with open(os.path.join(cfg.save_dir, "cfg.yaml"), 'w') as f:
            yaml.dump(vars(cfg), f, sort_keys=False)
        # sync_lock for run_eval
        os.makedirs(cfg.sync_lock_dir, exist_ok=False)

    # Set Logger
    logger.setup_logging(logger_name="MindYOLO", log_level="INFO", rank_id=rank, device_per_servers=rank_size)
    logger.setup_logging_file(log_dir=os.path.join(cfg.save_dir, "logs"))

    # Modelarts: Copy data, from the s3 bucket to the computing node; Reset dataset dir.
    if cfg.enable_modelarts:
        from mindyolo.utils.modelarts import sync_data
        os.makedirs(cfg.data_dir, exist_ok=True)
        sync_data(cfg.data_url, cfg.data_dir)
        sync_data(cfg.save_dir, cfg.train_url)
        if cfg.ckpt_url:
            sync_data(cfg.ckpt_url, cfg.ckpt_dir)  # pretrain ckpt
        cfg.data.dataset_dir = os.path.join(cfg.data_dir, cfg.data.dataset_dir)
        cfg.weight = os.path.join(cfg.ckpt_dir, cfg.weight) if cfg.weight else ''
        cfg.ema_weight = os.path.join(cfg.ckpt_dir, cfg.ema_weight) if cfg.ema_weight else ''
