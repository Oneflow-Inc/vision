import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import math

from flowvision.layers.helpers import make_divisible
from flowvision.layers.activation import hard_swish


class CoordAttModule(nn.Module):
    def __init__(
        self,
        channels,
        out_channels=None,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=1,
        act_layer=hard_swish,
        gate_layer=nn.Sigmoid,
        mlp_bias=False,
    ):
        super(CoordAttModule, self).__init__()
        if not rd_channels:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, min_value=8, round_limit=0.0
            )
        out_channels = channels if out_channels is None else out_channels

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(
            channels, rd_channels, kernel_size=1, stride=1, padding=0, bias=mlp_bias
        )
        self.bn1 = nn.BatchNorm2d(rd_channels)
        self.act = act_layer()

        self.conv_h = nn.Conv2d(
            rd_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=mlp_bias
        )
        self.conv_w = nn.Conv2d(
            rd_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=mlp_bias
        )
        self.gate = gate_layer()

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = flow.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = flow.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.gate(self.conv_h(x_h))
        a_w = self.gate(self.conv_w(x_w))

        out = identity * a_w * a_h

        return out
