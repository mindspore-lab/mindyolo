from mindspore import nn, ops, Parameter
from mindspore.common.initializer import Constant, initializer

from .conv import ConvNormAct, DWConvNormAct, RepConv
from ..initializer import trunc_normal_

class Bottleneck(nn.Cell):
    # Standard bottleneck
    def __init__(
        self, c1, c2, shortcut=True, k=(1, 3), g=(1, 1), e=0.5, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, k[0], 1, g=g[0], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c_, c2, k[1], 1, g=g[1], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        if self.add:
            out = x + self.conv2(self.conv1(x))
        else:
            out = self.conv2(self.conv1(x))
        return out

    
class Residualblock(nn.Cell):
    def __init__(
        self, c1, c2, k=(1, 3), g=(1, 1), act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, kernels, groups, expand
        super().__init__()
        self.conv1 = ConvNormAct(c1, c2, k[0], 1, g=g[0], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c2, c2, k[1], 1, g=g[1], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)

    def construct(self, x):
        out = x + self.conv2(self.conv1(x))
        return out


class C3(nn.Cell):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False):
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv3 = ConvNormAct(2 * c_, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)  # act=FReLU(c2)
        self.m = nn.SequentialCell(
            [
                Bottleneck(c_, c_, shortcut, k=(1, 3), g=(1, g), e=1.0, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.m(c1)
        c3 = self.conv2(x)
        c4 = self.concat((c2, c3))
        c5 = self.conv3(c4)

        return c5


class C2f(nn.Cell):
    # CSP Bottleneck with 2 convolutions
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvNormAct(c1, 2 * self.c, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(
            (2 + n) * self.c, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn
        )  # optional act=FReLU(c2)
        self.m = nn.CellList(
            [
                Bottleneck(self.c, self.c, shortcut, k=(3, 3), g=(1, g), e=1.0, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )

    def construct(self, x):
        y = ()
        x = self.cv1(x)
        _c = x.shape[1] // 2
        x_tuple = ops.split(x, axis=1, split_size_or_sections=_c)
        y += x_tuple
        for i in range(len(self.m)):
            m = self.m[i]
            out = m(y[-1])
            y += (out,)

        return self.cv2(ops.concat(y, axis=1))


class DWBottleneck(nn.Cell):
    # depthwise bottleneck used in yolox nano scale
    def __init__(
        self, c1, c2, shortcut=True, k=(1, 3), e=0.5, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, k[0], 1, act=True, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = DWConvNormAct(c_, c2, k[1], 1, act=True, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        if self.add:
            out = x + self.conv2(self.conv1(x))
        else:
            out = self.conv2(self.conv1(x))
        return out


class DWC3(nn.Cell):
    # depthwise DwC3 used in yolox nano scale, similar as C3
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False):
        super(DWC3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv3 = ConvNormAct(2 * c_, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)  # act=FReLU(c2)
        self.m = nn.SequentialCell(
            [
                DWBottleneck(c_, c_, shortcut, k=(1, 3), e=1.0, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.m(c1)
        c3 = self.conv2(x)
        c4 = self.concat((c2, c3))
        c5 = self.conv3(c4)

        return c5


class RepNBottleneck(nn.Cell):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, sync_bn=False):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1, bn=False, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(c_, c2, k[1], 1, g=g, sync_bn=sync_bn)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepNCSP(nn.Cell):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, sync_bn=False):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(RepNCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvNormAct(c1, c_, 1, 1, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(c1, c_, 1, 1, sync_bn=sync_bn)
        self.cv3 = ConvNormAct(2 * c_, c2, 1, sync_bn=sync_bn)  # optional act=FReLU(c2)
        self.m = nn.SequentialCell(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0, sync_bn=sync_bn) for _ in range(n)))

    def construct(self, x):
        return self.cv3(ops.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNCSPELAN4(nn.Cell):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1, sync_bn=False):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(RepNCSPELAN4, self).__init__()
        self.c = c3//2
        self.cv1 = ConvNormAct(c1, c3, 1, 1, sync_bn=sync_bn)
        self.cv2 = nn.SequentialCell(RepNCSP(c3//2, c4, c5, sync_bn=sync_bn), ConvNormAct(c4, c4, 3, 1, sync_bn=sync_bn))
        self.cv3 = nn.SequentialCell(RepNCSP(c4, c4, c5, sync_bn=sync_bn), ConvNormAct(c4, c4, 3, 1, sync_bn=sync_bn))
        self.cv4 = ConvNormAct(c3+(2*c4), c2, 1, 1, sync_bn=sync_bn)

    def construct(self, x):
        y = ()
        x = self.cv1(x)
        _c = x.shape[1] // 2
        x_tuple = ops.split(x, axis=1, split_size_or_sections=_c)
        y += x_tuple
        for m in [self.cv2, self.cv3]:
            out = m(y[-1])
            y += (out,)
        return self.cv4(ops.cat(y, 1))

class SCDown(nn.Cell):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvNormAct(c1, c2, k=1, s=1)
        self.cv2 = ConvNormAct(c2, c2, k=k, s=s, g=c2, act=False)
    
    def construct(self, x):
        return self.cv2(self.cv1(x))

class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = ConvNormAct(c1=dim, c2=h, k=1, act=False)
        self.proj = ConvNormAct(c1=dim, c2=dim, k=1, act=False)
        self.pe = ConvNormAct(c1=dim, c2=dim, k=3, s=1, g=dim, act=False)

    def construct(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], axis=2)
        
        attn = (
            (ops.transpose(q, (0, 1, 3, 2)) @ k) * self.scale
        )
        attn = ops.softmax(attn)
        x = (v @ ops.transpose(attn, (0, 1, 3, 2))).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
    
class PSA(nn.Cell):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert(c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = ConvNormAct(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvNormAct(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.SequentialCell(
            [
                ConvNormAct(self.c, self.c*2, 1),
                ConvNormAct(self.c*2, self.c, 1, act=False)
            ]
        )
    
    def construct(self, x):
        a, b = self.cv1(x).split((self.c, self.c), axis=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(ops.concat((a, b), 1))

class RepVGGDW(nn.Cell):
    def __init__(self, ed):
        super().__init__()
        self.conv = ConvNormAct(ed, ed, k=7, s=1, p=3, g=ed, act=False)
        self.conv1 = ConvNormAct(ed, ed, k=3, s=1, p=1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()
    
    def construct(self, x):
        return self.act(self.conv(x) + self.conv1(x))

class CIB(nn.Cell):
    # Standard bottleneck
    def __init__(
        self, c1, c2, shortcut=True, e=0.5, lk=False, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.SequentialCell(
            [
                ConvNormAct(c1, c1, 3, g=c1, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn),
                ConvNormAct(c1, 2 * c_, 1, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn),
                ConvNormAct(2 * c_, 2 * c_, 3, g=2 * c_, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn) if not lk else RepVGGDW(2 * c_),
                ConvNormAct(2 * c_, c2, 1, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn),
                ConvNormAct(c2, c2, 3, g=c2, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn),
            ]
        )
        self.add = shortcut and c1 == c2
    
    def construct(self, x):
        if self.add:
            out = x + self.cv1(x)
        else:
            out = self.cv1(x)
        return out

class C2fCIB(C2f):
    # CSP Bottleneck with 2 convolutions
    def __init__(
        self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e, momentum, eps, sync_bn)
        self.m = nn.CellList(
            [
                CIB(self.c, self.c, shortcut, e=1.0, lk=lk, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )

class C3k2(C2f):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, sync_bn=False):  
        # ch_in, ch_out, number, c3k, expansion, groups, shortcut, sync_bn
        super().__init__(c1, c2, n, shortcut, g, e, sync_bn=sync_bn)
        self.m = nn.CellList(
            [
                C3k(self.c, self.c, 2, shortcut, g, sync_bn=sync_bn) if c3k else Bottleneck(self.c, self.c, shortcut, k=(3, 3), g=(1, g), sync_bn=sync_bn)
                for _ in range(n)
            ]
        )
    
class C3k(C3):
    # C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks.
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, sync_bn=False):
        super().__init__(c1, c2, n, shortcut, g, e, sync_bn=sync_bn)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.SequentialCell(
            [
                Bottleneck(c_, c_, shortcut, k=(k, k), g=(1, g), e=1.0, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )

class AAttn(nn.Cell):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.
    """
    def __init__(self, dim, num_heads, area=1):
        """
        Initializes an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels;
            num_heads (int): Number of heads into which the attention mechanism is divided;
            area (int, optional): Number of areas the feature map is divided. Defaults to 1.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = ConvNormAct(dim, all_head_dim * 3, k=1, act=False)
        self.proj = ConvNormAct(all_head_dim, dim, k=1, act=False)
        self.pe = ConvNormAct(all_head_dim, dim, k=7, s=1, p=3, g=dim, act=False, has_bias=True)

    def construct(self, x):
        B, C, H, W = x.shape
        N = H * W

        qkv = ops.transpose(ops.flatten(self.qkv(x), start_dim=2), (0, 2, 1))
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        
        q, k, v = ops.transpose(qkv.view(B, N, self.num_heads, self.head_dim * 3), (0, 2, 3, 1)).split([self.head_dim, self.head_dim, self.head_dim], axis=2)
        attn = (
            (ops.transpose(q, (0, 1, 3, 2)) @ k) * (self.head_dim**-0.5)
        )
        attn = ops.softmax(attn, -1)
        x = v @ ops.transpose(attn, (0, 1, 3, 2))
        x = ops.transpose(x, (0, 3, 1, 2))
        v = ops.transpose(v, (0, 3, 1, 2))

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        
        x = ops.transpose(x.reshape(B, H, W, C), (0, 3, 1, 2))
        v = ops.transpose(v.reshape(B, H, W, C), (0, 3, 1, 2))

        x = x + self.pe(v)
        return self.proj(x)
    
class ABlock(nn.Cell):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.
    """
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.SequentialCell(ConvNormAct(dim, mlp_hidden_dim, k=1), ConvNormAct(mlp_hidden_dim, dim, k=1, act=False))

    def _init_weights(self, m):
        # Initialize weights using a truncated normal distribution
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                m.bias.set_data(initializer(Constant(0), shape=m.bias.shape, dtype=m.bias.dtype))

    def construct(self, x):
        # Performs a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor.
        x = x + self.attn(x)
        return x + self.mlp(x)

class A2C2f(nn.Cell):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.
    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.    
    """
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        num_heads = c_ // 32

        self.cv1 = ConvNormAct(c1, c_, k=1, s=1)
        self.cv2 = ConvNormAct((1 + n) * c_, c2, k=1)

        init_values = 0.01
        self.gamma = Parameter(init_values * ops.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.CellList(
        [
            nn.SequentialCell(
                [
                    ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2)
                ]
            ) if a2 else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        ])

    def construct(self, x):
        # Performs a forward pass through R-ELAN layer.
        x1 = self.cv1(x)
        y = (x1, )
        for i in range(len(self.m)):
            m = self.m[i]
            out = m(y[-1])
            y += (out,)
        y = self.cv2(ops.concat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * y
        return y