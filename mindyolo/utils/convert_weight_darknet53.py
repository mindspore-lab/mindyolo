import os
import sys

import numpy as np

import mindspore as ms
from config import parse_args

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from mindyolo.models import create_model


def _load_weight(weights_file):
    """Loads pre-trained weights."""
    if not os.path.isfile(weights_file):
        raise ValueError(f'"{weights_file}" is not a valid weight file.')
    with open(weights_file, "rb") as fp:
        np.fromfile(fp, dtype=np.int32, count=5)
        return np.fromfile(fp, dtype=np.float32)


def convert_weight(cfg, weights_file="./darknet53.conv.74", output_file="./yolov3_backbone.ckpt"):
    """Convert weight to mindspore ckpt."""
    net = create_model(model_name=cfg.network.model_name, model_cfg=cfg.network)
    params = net.get_parameters()
    params = [p for p in params]
    weights = _load_weight(weights_file)
    index = 0
    param_list = []
    weights_num = len(weights)
    for i in range(0, len(params), 5):
        weight = params[i]
        mean = params[i + 1]
        var = params[i + 2]
        gamma = params[i + 3]
        beta = params[i + 4]
        beta_data = weights[index : index + beta.size].reshape(beta.shape)
        index += beta.size
        gamma_data = weights[index : index + gamma.size].reshape(gamma.shape)
        index += gamma.size
        mean_data = weights[index : index + mean.size].reshape(mean.shape)
        index += mean.size
        var_data = weights[index : index + var.size].reshape(var.shape)
        index += var.size
        weight_data = weights[index : index + weight.size].reshape(weight.shape)
        index += weight.size

        param_list.append(
            {"name": weight.name, "type": weight.dtype, "shape": weight.shape, "data": ms.Tensor(weight_data)}
        )
        param_list.append({"name": mean.name, "type": mean.dtype, "shape": mean.shape, "data": ms.Tensor(mean_data)})
        param_list.append({"name": var.name, "type": var.dtype, "shape": var.shape, "data": ms.Tensor(var_data)})
        param_list.append(
            {"name": gamma.name, "type": gamma.dtype, "shape": gamma.shape, "data": ms.Tensor(gamma_data)}
        )
        param_list.append({"name": beta.name, "type": beta.dtype, "shape": beta.shape, "data": ms.Tensor(beta_data)})

        if index >= weights_num:
            break

    ms.save_checkpoint(param_list, output_file)


if __name__ == "__main__":
    args = parse_args()
    convert_weight(args)
