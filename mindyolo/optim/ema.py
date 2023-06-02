import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops

__all__ = ["EMA"]


class EMA(nn.Cell):
    """Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weight is necessary for some training schemes to perform well.
    """

    def __init__(self, model, ema_model, decay=0.9999, updates=0):
        super(EMA, self).__init__()
        # Create EMA
        self.ema = ema_model
        self.ema.set_train(False)
        self.weight = ms.ParameterTuple(list(model.get_parameters()))
        self.ema_weight = ms.ParameterTuple(list(ema_model.get_parameters()))
        self.updates = Parameter(Tensor(updates, ms.float32), requires_grad=False)  # number of EMA updates
        self.decay_value = decay
        self.assign = ops.Assign()
        self.hyper_map = ops.HyperMap()

    def decay(self, x):
        # decay exponential ramp (to help early epochs)
        return self.decay_value * (1 - ops.exp(ops.neg(x) / 2000))

    @ms.jit
    def update(self):
        # Update EMA parameters
        def update_param(d, ema_v, weight):
            if weight.dtype == ms.int32:
                return self.assign(ema_v, weight)
            else:
                tep_v = ema_v * d
                return self.assign(ema_v, weight * (1.0 - d) + tep_v)

        ops.assign_add(self.updates, 1)
        d = self.decay(self.updates)
        success = self.hyper_map(ops.partial(update_param, d), self.ema_weight, self.weight)

        return success

    @ms.jit
    def clone_from_model(self):
        ops.assign_add(self.updates, 1)
        success = self.hyper_map(ops.assign, self.ema_weight, self.weight)
        return success
