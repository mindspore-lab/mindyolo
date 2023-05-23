""" optim factory """
import os
from typing import Optional

from mindspore import load_checkpoint, load_param_into_net, nn

__all__ = ["create_optimizer"]


def create_optimizer(
    params,
    optimizer: str = "momentum",
    lr: Optional[float] = 1e-3,
    weight_decay: float = 0,
    momentum: float = 0.9,
    nesterov: bool = False,
    loss_scale: float = 1.0,
    checkpoint_path: str = "",
    **kwargs,
):
    r"""Creates optimizer by name.

    Args:
        params: network parameters.
        optim: optimizer name like 'sgd', 'nesterov', 'momentum'.
        lr: learning rate, float or lr scheduler. Fixed and dynamic learning rate are supported. Default: 1e-3.
        weight_decay: weight decay factor. Default: 0.
        momentum: momentum if the optimizer supports. Default: 0.9.
        nesterov: Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients. Default: False.
        loss_scale: A floating point value for the loss scale, which must be larger than 0.0. Default: 1.0.
        checkpoint_path: Optimizer weight path. Default: ''.

    Returns:
        Optimizer object
    """

    optim = optimizer.lower()

    if optim == "sgd":
        optimizer = nn.SGD(
            params=params,
            learning_rate=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            loss_scale=loss_scale,
        )
    elif optim in ["momentum", "nesterov"]:
        optimizer = nn.Momentum(
            params=params,
            learning_rate=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            use_nesterov=nesterov,
            loss_scale=loss_scale,
        )
    else:
        raise ValueError(f"Invalid optimizer: {optim}")

    if checkpoint_path.endswith(".ckpt") and os.path.isfile(checkpoint_path):
        param_dict = load_checkpoint(checkpoint_path, filter_prefix="learning_rate")
        load_param_into_net(optimizer, param_dict)

    return optimizer
