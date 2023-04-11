import mindspore as ms
from mindspore import context, nn, ops

__all__ = [
    "create_train_step_fn",
    "get_gradreducer", "get_loss_scaler"
]



def get_gradreducer(is_parallel, parameters):
    if is_parallel:
        mean = context.get_auto_parallel_context("gradients_mean")
        degree = context.get_auto_parallel_context("device_num")
        grad_reducer = nn.DistributedGradReducer(parameters, mean, degree)
    else:
        grad_reducer = ops.functional.identity

    return grad_reducer


def get_loss_scaler(ms_loss_scaler='static', scale_value=1024, scale_factor=2, scale_window=2000):
    if ms_loss_scaler == 'dynamic':
        from mindspore.amp import DynamicLossScaler
        loss_scaler = DynamicLossScaler(scale_value=scale_value,
                                        scale_factor=scale_factor,
                                        scale_window=scale_window)
    elif ms_loss_scaler == 'static':
        from mindspore.amp import StaticLossScaler
        loss_scaler = StaticLossScaler(scale_value=scale_value)
    elif ms_loss_scaler in ('none', 'None'):
        from mindspore.amp import StaticLossScaler
        loss_scaler = StaticLossScaler(1.0)
    else:
        raise NotImplementedError(f"Not support ms_loss_scaler: {ms_loss_scaler}")

    return loss_scaler


def create_train_step_fn(network, loss_fn, optimizer, loss_ratio, scaler, reducer, overflow_still_update=False, ms_jit=False):
    from mindyolo.utils.all_finite import all_finite

    def forward_func(x, label):
        pred = network(x)
        loss, loss_items = loss_fn(pred, label, x)
        loss *= loss_ratio
        return scaler.scale(loss), ops.stop_gradient(loss_items)

    grad_fn = ops.value_and_grad(forward_func, grad_position=None, weights=optimizer.parameters, has_aux=True)

    def train_step_func(x, label, optimizer_update=True):
        (loss, loss_items), grads = grad_fn(x, label)
        grads = reducer(grads)
        unscaled_grads = scaler.unscale(grads)
        grads_finite = all_finite(unscaled_grads)

        if optimizer_update:
            if grads_finite:
                loss = ops.depend(loss, optimizer(unscaled_grads))
            else:
                if overflow_still_update:
                    loss = ops.depend(loss, optimizer(unscaled_grads))

        return scaler.unscale(loss), loss_items, unscaled_grads, grads_finite

    @ms.ms_function
    def jit_warpper(*args):
        return train_step_func(*args)

    return train_step_func if not ms_jit else jit_warpper
