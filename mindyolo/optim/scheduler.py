import math
import numpy as np

__all__ = ["create_lr_scheduler", "create_warmup_momentum_scheduler"]


def create_lr_scheduler(lr_init, lr_scheduler=None, by_epoch=True, **kwargs):
    """
    Create lr scheduler for optimizer.

    Args:
        lr_init: Initial learning rate
        lr_scheduler: LR scheduler name like 'linear', 'cos'.
        by_epoch: learning rate updated by epoch if true, else updated by iteration. Default true
        **kwargs: Others
    """

    if lr_scheduler:
        assert isinstance(lr_scheduler, str), f"lr_scheduler should be a string, but got {type(lr_scheduler)}"
        if lr_scheduler == "yolox":
            return create_yolox_lr_scheduler(lr_init=lr_init, by_epoch=by_epoch, **kwargs)
    else:
        return lr_init


def create_yolox_lr_scheduler(
    start_factor, end_factor, lr_init, steps_per_epoch, warmup_epochs, epochs, by_epoch, cooldown_epochs=0, **kwargs
):
    assert epochs - warmup_epochs - cooldown_epochs > 0, f"the sum of warmup({warmup_epochs}) and " \
                                                         f"cooldown{cooldown_epochs} epoch should " \
                                                         f"be less than total epoch{epochs}"
    # quadratic
    lrs_qua = quadratic_lr(0.01, start_factor, lr_init, steps_per_epoch, epochs=warmup_epochs, by_epoch=by_epoch)

    # cosine
    cosine_epochs = epochs - warmup_epochs - cooldown_epochs
    lrs_cos = cosine_decay_lr(
        start_factor, end_factor, lr_init, steps_per_epoch, epochs=cosine_epochs, by_epoch=by_epoch
    )

    # constant
    lrs_col = []
    if cooldown_epochs > 0:
        cool_down_lr = lr_init * end_factor
        lrs_col = [cool_down_lr] * cooldown_epochs * steps_per_epoch

    lrs = lrs_qua + lrs_cos + lrs_col
    return lrs


def quadratic_lr(start_factor, end_factor, lr_init, steps_per_epoch, epochs, by_epoch=True, t_max=None, **kwargs):
    if t_max is None:
        t_max = epochs if by_epoch else steps_per_epoch * epochs
    lrs = []
    start_lr = lr_init * start_factor
    end_lr = lr_init * end_factor
    for i in range(steps_per_epoch * epochs):
        epoch_idx = i // steps_per_epoch
        index = epoch_idx if by_epoch else i
        multiplier = min(index, t_max) / t_max
        multiplier = pow(multiplier, 2)
        lrs.append(start_lr + multiplier * (end_lr - start_lr))
    return lrs


def create_warmup_momentum_scheduler(
    steps_per_epoch, momentum=None, warmup_momentum=None, warmup_epochs=None, min_warmup_step=None, **kwargs
):
    """
    Create warmup momentum scheduler.

    Args:
        steps_per_epoch: Number of steps in each epoch.
        momentum (float, optional): Hyperparameter of type float, means momentum for the moving average.
            It must be at least 0.0. Default: None.
        warmup_momentum (float, optional): Hyperparameter of type float, means warmup momentum for the moving average.
            It must be at least 0.0. Default: None.
        warmup_epochs: Number of epochs for warmup.
        min_warmup_step: Minimum number of steps for warmup.
        **kwargs: Others
    """

    if warmup_momentum:
        warmup_steps = max(round(warmup_epochs * steps_per_epoch), min_warmup_step)
        return linear_momentum(warmup_momentum, momentum, warmup_steps)
    else:
        return None


def linear_momentum(start, end, total_steps):
    """
    Args:
        start: Starting value.
        end: Ending value.
        total_steps: Number of total step.

    Returns:
        momentum_list: A list with length total_steps.
    """

    momentum_list = []
    for i in range(total_steps):
        momentum_list.append(np.interp(i, [0, total_steps], [start, end]))

    return momentum_list


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
        t_max = epochs
    lrs = []
    start_lr = lr_init * start_factor
    end_lr = lr_init * end_factor
    for i in range(steps_per_epoch * epochs):
        epoch_idx = i // steps_per_epoch
        multiplier = min(epoch_idx, t_max) / t_max
        lrs.append(start_lr + multiplier * (end_lr - start_lr))
    return lrs


def cosine_decay_lr(start_factor, end_factor, lr_init, steps_per_epoch, epochs, by_epoch=True, t_max=None, **kwargs):
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
        t_max = epochs if by_epoch else steps_per_epoch * epochs
    lrs = []
    start_lr = lr_init * start_factor
    end_lr = lr_init * end_factor
    delta = 0.5 * (start_lr - end_lr)
    for i in range(steps_per_epoch * epochs):
        epoch_idx = i // steps_per_epoch
        index = epoch_idx if by_epoch else i
        multiplier = min(index, t_max) / t_max
        lrs.append(end_lr + delta * (1.0 + math.cos(math.pi * multiplier)))
    return lrs


def cosine_decay_lr_with_linear_warmup(
    warmup_epochs,
    warmup_lrs,
    start_factor,
    end_factor,
    lr_init,
    steps_per_epoch,
    epochs,
    min_warmup_step=1000,
    t_max=None,
    **kwargs,
):
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
        warmup_lrs = [
            warmup_lrs,
        ]
    elif isinstance(warmup_lrs, (list, tuple)):
        if warmup_lrs[-1] in ("None", "none", None):
            warmup_lrs = warmup_lrs[:-1]
    else:
        raise ValueError

    assert (
        len(warmup_epochs) == len(warmup_lrs) + 1
    ), "LRScheduler: The length of 'warmup_epochs' and 'warmup_lrs' is inconsistent"

    lrs = cosine_decay_lr(start_factor, end_factor, lr_init, steps_per_epoch, epochs, t_max)
    warmup_steps = [min(i * steps_per_epoch, len(lrs)) for i in warmup_epochs]
    warmup_steps[-1] = max(warmup_steps[-1], min(len(lrs), min_warmup_step))

    for i in range(warmup_steps[-1]):
        _lr = lrs[i]
        lrs[i] = np.interp(i, warmup_steps, warmup_lrs + [_lr,])

    return lrs
