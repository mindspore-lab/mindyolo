from mindspore import nn, ops

from .conv import ConvNormAct
from .pool import MaxPool2d

class SPPCSPC(nn.Cell):
    """
    CSPNet, https://arxiv.org/pdf/1911.11929v1.pdf. The main purpose of designing CSPNet is to enable
        this architecture to achieve a richer gradient combination while reducing the amount of computation. This aim
        is achieved by partitioning feature map of the base layer into two parts and then merging them through a proposed
        cross-stage hierarchy. Our main concept is to make the gradient flow propagate through different network paths
        by splitting the gradient flow. In this way, we have confirmed that the propagated gradient information can
        have a large correlation difference by switching concatenation and transition steps.
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = ConvNormAct(c1, c_, 1, 1)
        self.cv2 = ConvNormAct(c1, c_, 1, 1)
        self.cv3 = ConvNormAct(c_, c_, 3, 1)
        self.cv4 = ConvNormAct(c_, c_, 1, 1)
        self.m = nn.CellList([MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = ConvNormAct(4 * c_, c_, 1, 1)
        self.cv6 = ConvNormAct(c_, c_, 3, 1)
        self.cv7 = ConvNormAct(2 * c_, c2, 1, 1)

    def construct(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        m_tuple = (x1,)
        for i in range(len(self.m)):
            m_tuple += (self.m[i](x1),)
        y1 = self.cv6(self.cv5(ops.Concat(axis=1)(m_tuple)))
        y2 = self.cv2(x)
        return self.cv7(ops.Concat(axis=1)((y1, y2)))
