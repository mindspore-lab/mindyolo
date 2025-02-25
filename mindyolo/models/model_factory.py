import math
import os
from copy import deepcopy

import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net, nn, ops

from mindyolo.utils import logger
from .heads import *
from .layers import *
from .registry import is_model, model_entrypoint
from .initializer import initialize_defult

__all__ = ["create_model", "build_model_from_cfg"]


def create_model(
    model_name: str,
    model_cfg: dict = None,
    in_channels: int = 3,
    num_classes: int = 80,
    checkpoint_path: str = "",
    **kwargs,
):
    model_args = dict(cfg=model_cfg, num_classes=num_classes, in_channels=in_channels)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if not is_model(model_name):
        raise RuntimeError(f"Unknown model {model_name}")

    create_fn = model_entrypoint(model_name)
    model = create_fn(**model_args, **kwargs)

    if checkpoint_path:
        assert os.path.isfile(checkpoint_path) and checkpoint_path.endswith(
            ".ckpt"
        ), f"[{checkpoint_path}] not a ckpt file."
        checkpoint_param = load_checkpoint(checkpoint_path)
        load_param_into_net(model, checkpoint_param)
        logger.info(f"Load checkpoint from [{checkpoint_path}] success.")

    return model


# Tools: build model from yaml cfg
def build_model_from_cfg(**kwargs):
    return Model(**kwargs)


class Model(nn.Cell):
    def __init__(self, model_cfg, in_channels=3, num_classes=80, sync_bn=False):
        super(Model, self).__init__()
        self.model, self.save, self.layers_param = parse_model(
            deepcopy(model_cfg), ch=[in_channels], nc=num_classes, sync_bn=sync_bn
        )
        # Recompute
        if hasattr(model_cfg, "recompute") and model_cfg.recompute and model_cfg.recompute_layers > 0:
            for i in range(model_cfg.recompute_layers):
                self.model[i].recompute()
            logger.info(
                f"Turn on recompute, and the results of the first {model_cfg.recompute_layers} layers "
                f"will be recomputed."
            )
        initialize_defult(self)

    def construct(self, x):
        y, dt = (), ()  # outputs
        for i in range(len(self.model)):
            m = self.model[i]
            iol, f, _, _ = self.layers_param[i]  # iol: index of layers

            if not (isinstance(f, int) and f == -1):  # if not from previous layer
                if isinstance(f, int):
                    x = y[f]
                else:
                    _x = ()
                    for j in f:
                        if j == -1:
                            _x += (x,)
                        else:
                            _x += (y[j],)
                    x = _x

            x = m(x)  # run

            y += (x if iol in self.save else None,)  # save output
        return x

    @staticmethod
    @ops.constexpr
    def _get_h_w_list(ratio, gs, hw):
        return tuple([math.ceil(x * ratio / gs) * gs for x in hw])


def parse_model(d, ch, nc, sync_bn=False):  # model_dict, input_channels(3)
    _SYNC_BN = sync_bn
    if _SYNC_BN:
        logger.info("Parse model with Sync BN.")
    verbose = d.get("verbose_log", False)
    if verbose:
        logger.info("")
        logger.info("network structure are as follows")
        logger.info("%3s%18s%3s%10s  %-60s%-40s" % ("", "from", "n", "params", "module", "arguments"))
    anchors, reg_max, max_channels = d.get("anchors", None), d.get("reg_max", None), d.get("max_channels", None)
    stride, gd, gw = d.stride, d.depth_multiple, d.width_multiple

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    layers_param = []
    num_total_param, num_train_param = 0, 0
    for i, (f, n, m, args) in enumerate(d.backbone + d.head):  # from, number, module, args
        kwargs = {}
        m = eval(m) if isinstance(m, str) else m  # eval strings

        _args = []
        for j, a in enumerate(args):
            if isinstance(a, str) and "=" in a:
                _index = a.find("=")
                k, v = a[:_index], a[_index + 1 :]
                try:
                    v = eval(v)
                except:
                    logger.warning(f"Parse Model, args: {k}={v}, keep str type")
                kwargs[k] = v
            else:
                try:
                    a = eval(a) if isinstance(a, str) else a
                except:
                    logger.warning(f"Parse Model, args: {a}, keep str type")
                _args += [
                    a,
                ]
        args = _args

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in (
            nn.Conv2d,
            ConvNormAct,
            AConv,
            RepConv,
            DownC,
            SPPCSPC,
            ADown,
            SPPF,
            C3,
            C2f,
            Bottleneck,
            Residualblock,
            Focus,
            DWConvNormAct,
            DWBottleneck,
            DWC3,
            ELAN1,
            RepNCSPELAN4,
            SPPELAN,
            SCDown,
            PSA,
            C2fCIB,
            C3k2,
            A2C2f,
        ):
            c1, c2 = ch[f], args[0]
            if max_channels:
                c2 = min(c2, max_channels)
            c2 = math.ceil(c2 * gw / 8) * 8

            args = [c1, c2, *args[1:]]
            if m in (
                ConvNormAct,
                RepConv,
                DownC,
                SPPCSPC,
                SPPF,
                C3,
                C2f,
                Bottleneck,
                Residualblock,
                DWConvNormAct,
                DWBottleneck,
                DWC3,
                RepNCSPELAN4,
                AConv,
                ELAN1,
                SPPELAN,
                ADown
            ):
                kwargs["sync_bn"] = sync_bn
            if m in (DownC, SPPCSPC, C3, C2f, DWC3, C2fCIB, C3k2, A2C2f):
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2 and d.get("scale") in "mlx":
                args[3] = True
            if m is A2C2f and d.get("scale") in "lx":
                args.extend((True, 1.2))
        elif m in (nn.BatchNorm2d, nn.SyncBatchNorm):
            args = [ch[f]]
        elif m in (Concat,):
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m in (YOLOv7Head, YOLOv7AuxHead, YOLOv5Head, YOLOv4Head, YOLOv3Head):
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m in (YOLOv12Head, YOLOv10Head, YOLOv9Head, YOLOv8Head, YOLOv8SegHead, YOLOXHead):  # head of anchor free
            args.append([ch[x] for x in f])
            if m in (YOLOv8SegHead,):
                args[3] = math.ceil(min(args[3], max_channels) * gw / 8) * 8
            if m in (YOLOv9Head,):
                kwargs["sync_bn"] = sync_bn
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        else:
            c2 = ch[f]

        m_ = nn.SequentialCell([m(*args, **kwargs) for _ in range(n)]) if n > 1 else m(*args, **kwargs)

        t = str(m)  # module type
        np = sum([x.size for x in m_.get_parameters()])  # number params
        np_trainable = sum([x.size for x in m_.trainable_params()])  # number trainable params
        num_total_param += np
        num_train_param += np_trainable
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        layers_param.append((i, f, t, np))
        if verbose:
            logger.info("%3s%18s%3s%10.0f  %-60s%-40s" % (i, f, n, np, t, args + [kwargs] if kwargs else args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    logger.info(f"number of network params, total: {num_total_param / 1e6}M, trainable: {num_train_param / 1e6}M")
    return nn.CellList(layers), sorted(save), layers_param
