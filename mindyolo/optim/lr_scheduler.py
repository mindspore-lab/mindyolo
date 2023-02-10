import math
import numpy as np

__all__ = ['create_lr_scheduler']


def create_lr_scheduler():
    # TODO: Add lr_scheduler
    pass


def linear_lr(start_factor, end_factor, lr_init, steps_per_epoch, epochs, t_max=None, **kwargs):
    """
    Args:
        start_factor: Starting factor.
        end_factor: Ending factor.
        lr_init: Initial learning rate.
        steps_per_epoch: Total number of steps per epoch.
        epochs: Total number of epochs trained.
        t_max: The maximum number of epochs where lr changes. Default: None.

    Examples:
        >>> lrs = linear_lr(0.1, 0.01, 0.2, 100, 5)
        >>> print(f"lrs len: {len(lrs)}")
        >>> print(f"lrs per epoch: {[lrs[i] for i in range(len(lrs)) if ((i + 1) % 100 == 0)]}")
        lrs len: 500
        lrs: [0.02, 0.0155, 0.011, 0.0065, 0.002]
    """

    if t_max is None:
        t_max = epochs - 1 if epochs > 1 else 1
    lrs = []
    start_lr = lr_init * start_factor
    end_lr = lr_init * end_factor
    for i in range(steps_per_epoch * epochs):
        epoch_idx = i // steps_per_epoch
        multiplier = min(epoch_idx, t_max) / t_max
        lrs.append(start_lr + multiplier * (end_lr - start_lr))
    return lrs


def cosine_decay_lr(start_factor, end_factor, lr_init, steps_per_epoch, epochs, t_max=None, **kwargs):
    """
    Args:
        start_factor: Starting factor.
        end_factor: Ending factor.
        lr_init: Initial learning rate.
        steps_per_epoch: Total number of steps per epoch.
        epochs: Total number of epochs trained.
        t_max: The maximum number of epochs where lr changes. Default: None.

    Examples:
        >>> lrs = cosine_decay_lr(0.1, 0.01, 0.2, 100, 5)
        >>> print(f"lrs len: {len(lrs)}")
        >>> print(f"lrs: {[lrs[i] for i in range(len(lrs)) if ((i + 1) % 100 == 0)]}")
        lrs len: 500
        lrs: [0.02, 0.0173, 0.011, 0.0046, 0.002]
    """

    if t_max is None:
        t_max = epochs - 1 if epochs > 1 else 1
    lrs = []
    start_lr = lr_init * start_factor
    end_lr = lr_init * end_factor
    delta = 0.5 * (start_lr - end_lr)
    for i in range(steps_per_epoch * epochs):
        t_cur = i // steps_per_epoch
        t_cur = min(t_cur, t_max)
        lrs.append(end_lr + delta * (1.0 + math.cos(math.pi * t_cur / t_max)))
    return lrs


def cosine_decay_lr_with_linear_warmup(warmup_epochs, warmup_lrs,
                                       start_factor, end_factor, lr_init, steps_per_epoch, epochs,
                                       min_warmup_step=1000, t_max=None, **kwargs):
    """
    Args:
        warmup_epochs (Union[int, tuple[int]]): The warmup epochs of the lr scheduler.
            The data type is an integer or a tuple of integers. An integer represents the warmup epoch size.
            A tuple of integers represents the warmup epochs interpolation nodes. Like: [0, 12, 24] or 24.
        warmup_lrs (Union[int, tuple[float]]): The warmup lr of the lr scheduler.
            The data type is a float or a tuple of float(The last element can be None).
            A float represents the start warmup lr.
            A tuple of float represents the warmup lrs interpolation nodes. Like: [0.01, 0.1, 'None'] or [0.01, 0.1] or 0.01.
        start_factor: Starting factor.
        end_factor: Ending factor.
        lr_init: Initial learning rate.
        steps_per_epoch: Total number of steps per epoch.
        epochs: Total number of epochs trained.
        min_warmup_step (int): Minimum warm-up steps. Default: 1000.
        t_max: The maximum number of epochs where lr changes. Default: None.

    Examples:
        >>> lrs = cosine_decay_lr_with_linear_warmup([0, 3], [0.0001, None], 0.1, 0.01, 0.2, 100, 5, min_warmup_step=1)
        >>> print(f"lrs len: {len(lrs)}")
        >>> print(f"lrs every epoch: {[lrs[i] for i in range(len(lrs)) if ((i + 1) % 100 == 0)]}")
        lrs len: 500
        lrs every epoch: [0.0066, 0.0115, 0.0109, 0.0046, 0.002]
    """

    warmup_epochs = [0, warmup_epochs] if isinstance(warmup_epochs, int) else warmup_epochs
    if isinstance(warmup_epochs, (int, float)):
        warmup_epochs = [0, int(warmup_epochs)]
    elif isinstance(warmup_epochs, (list, tuple)):
        warmup_epochs = warmup_epochs
    else:
        raise ValueError

    if isinstance(warmup_lrs, float):
        warmup_lrs = [warmup_lrs,]
    elif isinstance(warmup_lrs, (list, tuple)):
        if warmup_lrs[-1] in ('None', 'none', None):
            warmup_lrs = warmup_lrs[:-1]
    else:
        raise ValueError

    assert len(warmup_epochs) == len(warmup_lrs) + 1, \
        "LRScheduler: The length of 'warmup_epochs' and 'warmup_lrs' is inconsistent"

    lrs = cosine_decay_lr(start_factor, end_factor, lr_init, steps_per_epoch, epochs, t_max)
    warmup_steps = [min(i * steps_per_epoch, len(lrs)) for i in warmup_epochs]
    warmup_steps[-1] = max(warmup_steps[-1], min(len(lrs), min_warmup_step))

    for i in range(warmup_steps[-1]):
        _lr = lrs[i]
        lrs[i] = np.interp(i, warmup_steps, warmup_lrs + [_lr,])

    return lrs
