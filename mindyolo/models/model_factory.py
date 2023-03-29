import os
import math
from copy import deepcopy
from mindspore import nn, ops, load_checkpoint, load_param_into_net
from .registry import is_model, model_entrypoint
from .layers import *
from .heads import *

__all__ = [
    'create_model',
    'build_model_from_cfg'
]


def create_model(
        model_name: str,
        model_cfg: dict = None,
        in_channels: int = 3,
        num_classes: int = 80,
        checkpoint_path: str = '',
        **kwargs):
    model_args = dict(cfg=model_cfg, num_classes=num_classes, in_channels=in_channels)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if not is_model(model_name):
        raise RuntimeError(f'Unknown model {model_name}')

    create_fn = model_entrypoint(model_name)
    model = create_fn(**model_args, **kwargs)

    if os.path.exists(checkpoint_path):
        checkpoint_param = load_checkpoint(checkpoint_path)
        load_param_into_net(model, checkpoint_param)

    return model


# Tools: build model from yaml cfg
def build_model_from_cfg(**kwargs):
    return Model(**kwargs)


class Model(nn.Cell):
    def __init__(self, model_cfg, in_channels=3, num_classes=80, sync_bn=False):
        super(Model, self).__init__()
        self.model, self.save, self.layers_param = parse_model(deepcopy(model_cfg),
                                                               ch=[in_channels],
                                                               nc=num_classes,
                                                               sync_bn=sync_bn)
        # Recompute
        if hasattr(model_cfg, 'recompute') and model_cfg.recompute and model_cfg.recompute_layers > 0:
            for i in range(model_cfg.recompute_layers):
                self.model[i].recompute()
            print(f"Turn on recompute, and the results of the first {model_cfg.recompute_layers} layers "
                  f"will be recomputed.")

    def construct(self, x):
        y, dt = (), ()  # outputs
        for i in range(len(self.model)):
            m = self.model[i]
            iol, f, _, _ = self.layers_param[i]  # iol: index of layers

            if not (isinstance(f, int) and f == -1):  # if not from previous layer
                # x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
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

            # print("index m: ", iol) # print if debug on pynative mode, not available on graph mode.
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
        print('Parse model with Sync BN.')
    print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, stride, gd, gw = d.anchors, d.stride, d.depth_multiple, d.width_multiple
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    layers_param = []
    for i, (f, n, m, args) in enumerate(d.backbone + d.head):  # from, number, module, args
        kwargs = {}
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in (nn.Conv2d, ConvNormAct, RepConv, DownC, SPPCSPC, SPPF, C3, C2f, Bottleneck):
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = math.ceil(c2 * gw / 8) * 8

            args = [c1, c2, *args[1:]]
            if m in (ConvNormAct, RepConv, DownC, SPPCSPC, SPPF, C3, C2f, Bottleneck):
                kwargs["sync_bn"] = sync_bn
            if m in (DownC, SPPCSPC, C3, C2f):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m in (nn.BatchNorm2d, nn.SyncBatchNorm):
            args = [ch[f]]
        elif m in (Concat,):
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m in (YOLOv7Head, YOLOv7AuxHead, YOLOv5Head):
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            c2 = ch[f] * 4
        else:
            c2 = ch[f]

        m_ = nn.SequentialCell([m(*args, **kwargs) for _ in range(n)]) if n > 1 else m(*args, **kwargs)

        t = str(m) # module type
        np = sum([x.size for x in m_.get_parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        layers_param.append((i, f, t, np))
        print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.CellList(layers), sorted(save), layers_param
