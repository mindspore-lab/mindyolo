import numpy as np

from .scheduler import cosine_decay_lr, linear_lr

__all__ = ["create_group_param"]


def create_group_param(params, gp_weight_decay=0.0, **kwargs):
    """
    Create group parameters for optimizer.

    Args:
        params: Network parameters
        gp_weight_decay: Weight decay. Default: 0.0
        **kwargs: Others
    """
    if "group_param" in kwargs:
        gp_strategy = kwargs["group_param"]
        if gp_strategy == "filter_bias_and_bn":
            return filter_bias_and_bn(params, gp_weight_decay)
        elif gp_strategy == "yolov8":
            return group_param_yolov8(params, weight_decay=gp_weight_decay, **kwargs)
        elif gp_strategy == "yolov7":
            return group_param_yolov7(params, weight_decay=gp_weight_decay, **kwargs)
        elif gp_strategy == "yolov5":
            return group_param_yolov5(params, weight_decay=gp_weight_decay, **kwargs)
        elif gp_strategy == "yolov4":
            return group_param_yolov4(params, weight_decay=gp_weight_decay, **kwargs)
        elif gp_strategy == "yolov3":
            return group_param_yolov3(params, weight_decay=gp_weight_decay, **kwargs)
        else:
            raise NotImplementedError
    else:
        return params


def filter_bias_and_bn(params, weight_decay):
    no_decay_params, decay_params = _group_param_common2(params)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params},
    ]


def group_param_yolov3(
    params,
    weight_decay,
    start_factor,
    end_factor,
    lr_init,
    warmup_bias_lr,
    warmup_epochs,
    min_warmup_step,
    accumulate,
    epochs,
    steps_per_epoch,
    total_batch_size,
    **kwargs
):
    # old: # weight, gamma, bias/beta
    # new: # bias/beta, weight, others
    pg0, pg1, pg2 = _group_param_common3(params)

    lr_pg0, lr_pg1, lr_pg2 = [], [], []
    lrs = cosine_decay_lr(start_factor, end_factor, lr_init, steps_per_epoch, epochs)

    warmup_steps = max(round(warmup_epochs * steps_per_epoch), min_warmup_step)
    xi = [0, warmup_steps]
    for i in range(epochs * steps_per_epoch):
        _lr = lrs[i]
        if i < warmup_steps:
            lr_pg0.append(np.interp(i, xi, [warmup_bias_lr, _lr]))
            lr_pg1.append(np.interp(i, xi, [0.0, _lr]))
            lr_pg2.append(np.interp(i, xi, [0.0, _lr]))
        else:
            lr_pg0.append(_lr)
            lr_pg1.append(_lr)
            lr_pg2.append(_lr)

    nbs = 64
    weight_decay *= total_batch_size * accumulate / nbs  # scale weight_decay
    group_params = [
        {"params": pg0, "lr": lr_pg0},
        {"params": pg1, "lr": lr_pg1, "weight_decay": weight_decay},
        {"params": pg2, "lr": lr_pg2},
    ]
    return group_params


def group_param_yolov4(
    params,
    weight_decay,
    start_factor,
    end_factor,
    lr_init,
    warmup_epochs,
    min_warmup_step,
    accumulate,
    epochs,
    steps_per_epoch,
    total_batch_size,
    **kwargs
):
    pg0, pg1 = _group_param_common2(params)  # bias/beta/gamma, others

    lr_pg0, lr_pg1 = [], []
    lrs = cosine_decay_lr(start_factor, end_factor, lr_init, steps_per_epoch, epochs)

    warmup_steps = max(round(warmup_epochs * steps_per_epoch), min_warmup_step)

    xi = [0, warmup_steps]
    for i in range(epochs * steps_per_epoch):
        _lr = lrs[i]
        if i < warmup_steps:
            lr_pg0.append(np.interp(i, xi, [0.0, lr_init]))
            lr_pg1.append(np.interp(i, xi, [0.0, lr_init]))

        else:
            lr_pg0.append(_lr)
            lr_pg1.append(_lr)

    group_params = [{"params": pg0, "lr": lr_pg0}, {"params": pg1, "lr": lr_pg1, "weight_decay": weight_decay}]
    return group_params


def group_param_yolov5(
    params,
    weight_decay,
    start_factor,
    end_factor,
    lr_init,
    warmup_bias_lr,
    warmup_epochs,
    min_warmup_step,
    accumulate,
    epochs,
    steps_per_epoch,
    total_batch_size,
    **kwargs
):
    # old: # weight, gamma, bias/beta
    # new: # bias/beta, weight, others
    pg0, pg1, pg2 = _group_param_common3(params)

    lr_pg0, lr_pg1, lr_pg2 = [], [], []
    lrs = linear_lr(start_factor, end_factor, lr_init, steps_per_epoch, epochs)

    warmup_steps = max(round(warmup_epochs * steps_per_epoch), min_warmup_step)
    xi = [0, warmup_steps]
    for i in range(epochs * steps_per_epoch):
        _lr = lrs[i]
        if i < warmup_steps:
            lr_pg0.append(np.interp(i, xi, [warmup_bias_lr, _lr]))
            lr_pg1.append(np.interp(i, xi, [0.0, _lr]))
            lr_pg2.append(np.interp(i, xi, [0.0, _lr]))
        else:
            lr_pg0.append(_lr)
            lr_pg1.append(_lr)
            lr_pg2.append(_lr)

    nbs = 64
    weight_decay *= total_batch_size * accumulate / nbs  # scale weight_decay
    group_params = [
        {"params": pg0, "lr": lr_pg0},
        {"params": pg1, "lr": lr_pg1, "weight_decay": weight_decay},
        {"params": pg2, "lr": lr_pg2},
    ]
    return group_params


def group_param_yolov7(
    params,
    weight_decay,
    start_factor,
    end_factor,
    lr_init,
    warmup_bias_lr,
    warmup_epochs,
    min_warmup_step,
    accumulate,
    epochs,
    steps_per_epoch,
    total_batch_size,
    **kwargs
):
    pg0, pg1, pg2 = _group_param_common3(params)  # bias/beta, weight, others

    lr_pg0, lr_pg1, lr_pg2 = [], [], []
    lrs = cosine_decay_lr(start_factor, end_factor, lr_init, steps_per_epoch, epochs)

    warmup_steps = max(round(warmup_epochs * steps_per_epoch), min_warmup_step)
    warmup_bias_steps_first = min(max(round(3 * steps_per_epoch), min_warmup_step), warmup_steps)
    warmup_bias_lr_first = np.interp(warmup_bias_steps_first, [0, warmup_steps], [0.0, lr_init])
    xi = [0, warmup_steps]
    for i in range(epochs * steps_per_epoch):
        _lr = lrs[i]
        if i < warmup_steps:
            lr_pg0.append(
                np.interp(i, [0, warmup_bias_steps_first, warmup_steps], [warmup_bias_lr, warmup_bias_lr_first, _lr])
            )
            lr_pg1.append(np.interp(i, xi, [0.0, _lr]))
            lr_pg2.append(np.interp(i, xi, [0.0, _lr]))

        else:
            lr_pg0.append(_lr)
            lr_pg1.append(_lr)
            lr_pg2.append(_lr)

    nbs = 64
    weight_decay *= total_batch_size * accumulate / nbs  # scale weight_decay
    group_params = [
        {"params": pg0, "lr": lr_pg0},
        {"params": pg1, "lr": lr_pg1, "weight_decay": weight_decay},
        {"params": pg2, "lr": lr_pg2},
    ]
    return group_params


def group_param_yolov8(
    params,
    weight_decay,
    start_factor,
    end_factor,
    lr_init,
    warmup_bias_lr,
    warmup_epochs,
    min_warmup_step,
    accumulate,
    epochs,
    steps_per_epoch,
    total_batch_size,
    **kwargs
):
    pg0, pg1, pg2 = _group_param_common3(params)  # bias/beta, weight, others

    lr_pg0, lr_pg1, lr_pg2 = [], [], []
    lrs = linear_lr(start_factor, end_factor, lr_init, steps_per_epoch, epochs)

    warmup_steps = max(round(warmup_epochs * steps_per_epoch), min_warmup_step)
    xi = [0, warmup_steps]
    for i in range(epochs * steps_per_epoch):
        _lr = lrs[i]
        if i < warmup_steps:
            lr_pg0.append(np.interp(i, xi, [warmup_bias_lr, _lr]))
            lr_pg1.append(np.interp(i, xi, [0.0, _lr]))
            lr_pg2.append(np.interp(i, xi, [0.0, _lr]))
        else:
            lr_pg0.append(_lr)
            lr_pg1.append(_lr)
            lr_pg2.append(_lr)

    nbs = 64
    weight_decay *= total_batch_size * accumulate / nbs  # scale weight_decay
    group_params = [
        {"params": pg0, "lr": lr_pg0},
        {"params": pg1, "lr": lr_pg1, "weight_decay": weight_decay},
        {"params": pg2, "lr": lr_pg2},
    ]
    return group_params


def _group_param_common2(params):
    pg0, pg1 = [], []  # optimizer parameter groups
    for p in params:
        if "bias" in p.name or "beta" in p.name or "gamma" in p.name:
            pg0.append(p)
        else:
            pg1.append(p)

    return pg0, pg1  # bias/beta/gamma, others


def _group_param_common3(params):
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for p in params:
        if "bias" in p.name or "beta" in p.name:
            pg0.append(p)
        elif "weight" in p.name:
            pg1.append(p)
        else:
            pg2.append(p)

    return pg0, pg1, pg2  # bias/beta, weight, others
