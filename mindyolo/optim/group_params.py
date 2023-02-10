import numpy as np

from .lr_scheduler import cosine_decay_lr

__all__ = ['create_group_param']


def create_group_param(params, cfg=None):
    if isinstance(cfg, dict) and 'group_param' in cfg:
        gp_strategy = cfg['group_param']
        if gp_strategy == 'filter_bias_and_bn':
            return filter_bias_and_bn(params, cfg['weight_decay'])
        elif gp_strategy == "yolov7":
            return group_param_yolov7(params, cfg)
        else:
            raise NotImplementedError
    else:
        return params


def filter_bias_and_bn(params, weight_decay):
    no_decay_params, decay_params = _group_param_common2(params)

    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]


def group_param_yolov7(params, cfg):
    pg0, pg1, pg2 = _group_param_common3(params) # bias/beta, weight, others
    lrs = cosine_decay_lr(**cfg)

    lr_pg0, lr_pg1, lr_pg2, momentum_pg = [], [], [], []

    init_lr, weight_decay, momentum = \
        cfg.lr_init, cfg.weight_decay, cfg.momentum
    warmup_bias_lr, warmup_momentum, warmup_epochs, min_warmup_step = \
        cfg.warmup_bias_lr, cfg.warmup_momentum, cfg.warmup_epochs, cfg.get('min_warmup_step', 1000)
    total_batch_size, accumulate, epochs, steps_per_epoch = \
        cfg.total_batch_size, cfg.accumulate, cfg.epochs, cfg.steps_per_epoch

    warmup_steps = max(round(warmup_epochs * steps_per_epoch), min_warmup_step)
    warmup_bias_steps_first = min(max(round(3 * steps_per_epoch), min_warmup_step), warmup_steps)
    warmup_bias_lr_first = np.interp(warmup_bias_steps_first, [0, warmup_steps], [0.0, init_lr])
    xi = [0, warmup_steps]
    for i in range(epochs * steps_per_epoch):
        _lr = lrs[i]
        if i < warmup_steps:
            lr_pg0.append(np.interp(i, xi, [0.0, _lr]))
            lr_pg1.append(np.interp(i, xi, [0.0, _lr]))
            lr_pg2.append(np.interp(i,
                                    [0, warmup_bias_steps_first, warmup_steps],
                                    [warmup_bias_lr, warmup_bias_lr_first, _lr]))
            momentum_pg.append(np.interp(i, xi, [warmup_momentum, momentum]))
        else:
            lr_pg0.append(_lr)
            lr_pg1.append(_lr)
            lr_pg2.append(_lr)

        nbs = 64
        weight_decay *= total_batch_size * accumulate / nbs  # scale weight_decay
        group_params = [{'params': pg0, 'lr': lr_pg0},
                        {'params': pg1, 'lr': lr_pg1, 'weight_decay': weight_decay},
                        {'params': pg2, 'lr': lr_pg2}]
        return group_params


def _group_param_common2(params):
    pg0, pg1 = [], [] # optimizer parameter groups
    for p in params:
        if 'bias' in p.name or 'beta' in p.name or 'gamma' in p.name:
            pg0.append(p)
        else:
            pg1.append(p)

    return pg0, pg1 # bias/beta/gamma, others


def _group_param_common3(params):
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for p in params:
        if 'bias' in p.name or 'beta' in p.name:
            pg0.append(p)
        elif 'weight' in p.name:
            pg1.append(p)
        else:
            pg2.append(p)

    return pg0, pg1, pg2 # bias/beta, weight, others
