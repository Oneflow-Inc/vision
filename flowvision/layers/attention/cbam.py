import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import math

from ..helpers import make_divisible
from ..blocks import ConvBnAct


class ChannelAttn(nn.Module):
    """ Original CBAM channel attention module, currently avg + max pool variant only.
    """

    def __init__(
        self,
        channels,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=1,
        act_layer=nn.ReLU,
        gate_layer=nn.Sigmoid,
        mlp_bias=False,
    ):
        super(ChannelAttn, self).__init__()
        if not rd_channels:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        self.fc1 = nn.Conv2d(channels, rd_channels, 1, bias=mlp_bias)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, 1, bias=mlp_bias)
        self.gate = gate_layer()

    def forward(self, x):
        x_avg = self.fc2(self.act(self.fc1(x.mean((2, 3), keepdim=True))))
        # TODO: switch F.max_pool2d to amax
        x_max = self.fc2(
            self.act(
                self.fc1(
                    F.max_pool2d(
                        x,
                        kernel_size=(x.size(2), x.size(3)),
                        stride=(x.size(2), x.size(3)),
                    )
                )
            )
        )
        return x * self.gate(x_avg + x_max)


class SpatialAttn(nn.Module):
    """ Original CBAM spatial attention module
    """

    def __init__(self, kernel_size=7, gate_layer=nn.Sigmoid):
        super(SpatialAttn, self).__init__()
        # TODO: update ConvBnAct
        self.conv = ConvBnAct(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, act_layer=None
        )
        self.gate = gate_layer()

    def forward(self, x):
        # TODO: switch flow.max to tensor.amax
        x_attn = flow.cat(
            [x.mean(dim=1, keepdim=True), flow.max(x, dim=1, keepdim=True)[0]], dim=1
        )
        x_attn = self.conv(x_attn)
        return x * self.gate(x_attn)


class CbamModule(nn.Module):
    def __init__(
        self,
        channels,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=1,
        spatial_kernel_size=7,
        act_layer=nn.ReLU,
        gate_layer=nn.Sigmoid,
        mlp_bias=False,
    ):
        super(CbamModule, self).__init__()
        self.channel = ChannelAttn(
            channels,
            rd_ratio=rd_ratio,
            rd_channels=rd_channels,
            rd_divisor=rd_divisor,
            act_layer=act_layer,
            gate_layer=gate_layer,
            mlp_bias=mlp_bias,
        )
        self.spatial = SpatialAttn(spatial_kernel_size, gate_layer=gate_layer)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x
