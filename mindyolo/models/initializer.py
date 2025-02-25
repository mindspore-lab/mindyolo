import math

from mindspore import nn, Parameter
from mindspore.common import initializer as init

__all__ = ["initialize_defult", "trunc_normal_"]


def initialize_defult(model):
    for _, cell in model.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(
                init.initializer(init.HeUniform(negative_slope=math.sqrt(5)), cell.weight.shape, cell.weight.dtype)
            )
            if cell.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight.shape)
                bound = 1 / math.sqrt(fan_in)
                cell.bias.set_data(init.initializer(init.Uniform(bound), cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Dense):
            cell.weight.set_data(
                init.initializer(init.HeUniform(negative_slope=math.sqrt(5)), cell.weight.shape, cell.weight.dtype)
            )
            if cell.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight.shape)
                bound = 1 / math.sqrt(fan_in)
                cell.bias.set_data(init.initializer(init.Uniform(bound), cell.bias.shape, cell.bias.dtype))


def _calculate_fan_in_and_fan_out(shape):
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def trunc_normal_(tensor: Parameter, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
    tensor.set_data(
        init.initializer(init.TruncatedNormal(std, mean, a, b), tensor.shape, tensor.dtype)
    ) 