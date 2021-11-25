import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

from flowvision.layers.blocks import ConvBnAct
from flowvision.models.helpers import make_divisible


class LinearBnAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_layer: nn.Module = nn.ReLU,
        fc: nn.Module = nn.Linear,
        normalization: nn.Module = nn.BatchNorm1d,
        bias: bool = False,
    ):
        super().__init__()
        self.add_module("fc", fc(in_features, out_features, bias=bias))
        if normalization:
            self.add_module("bn", normalization(out_features))
        if act_layer:
            self.add_module("act", act_layer())


class BamChannelAttn(nn.Module):
    def __init__(
        self,
        channels,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=1,
        act_layer=nn.ReLU,
        gate_layer=nn.Sigmoid,
        num_layers=2,
        mlp_bias=False,
    ):
        super(BamChannelAttn, self).__init__()
        if not rd_channels:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        self.fc_layers = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.fc_layers.add_module(
                    "fc_bn_act_%d" % i,
                    LinearBnAct(channels, rd_channels, act_layer, bias=mlp_bias),
                )
            else:
                self.fc_layers.add_module(
                    "fc_bn_act_%d" % i,
                    LinearBnAct(rd_channels, rd_channels, act_layer, bias=mlp_bias),
                )
        self.fc_layers.add_module(
            "fc_out", nn.Linear(rd_channels, channels, bias=mlp_bias)
        )
        self.gate = gate_layer()

    def forward(self, x):
        b, c, _, _ = x.shape
        x_attn = self.gate(self.fc_layers(x.mean((2, 3)))).view(b, c, 1, 1)
        return x * x_attn.expand_as(x)


class BamSpatialAttn(nn.Module):
    def __init__(
        self,
        channels,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=1,
        act_layer=nn.ReLU,
        gate_layer=nn.Sigmoid,
        num_layers=1,
        dilation=4,
        mlp_bias=False,
    ):
        super(BamSpatialAttn, self).__init__()
        if not rd_channels:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        self.conv_layers = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.conv_layers.add_module(
                    "conv_bn_act_%d" % i,
                    ConvBnAct(
                        channels,
                        rd_channels,
                        act_layer,
                        kernel_size=3,
                        padding=4,
                        dilation=4,
                        bias=mlp_bias,
                    ),
                )
            else:
                self.conv_layers.add_module(
                    "conv_bn_act_%d" % i,
                    ConvBnAct(
                        rd_channels,
                        rd_channels,
                        act_layer,
                        kernel_size=3,
                        padding=4,
                        dilation=4,
                        bias=mlp_bias,
                    ),
                )
        self.conv_layers.add_module(
            "conv_final", nn.Conv2d(rd_channels, 1, kernel_size=1, bias=mlp_bias)
        )
        self.gate = gate_layer()

    def forward(self, x):
        b, c, _, _ = x.shape
        x_attn = self.gate(self.conv_layers(x))
        return x * x_attn.expand_as(x)


class BAMModule(nn.Module):
    def __init__(
        self,
        channels,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=1,
        act_layer=nn.ReLU,
        gate_layer=nn.Sigmoid,
        num_channel_attn_layers=2,
        num_spatial_attn_layers=2,
        mlp_bias=False,
    ):
        super(BAMModule, self).__init__()
        self.channel_att = BamChannelAttn(
            channels,
            rd_ratio,
            rd_channels,
            rd_divisor,
            act_layer,
            gate_layer,
            num_channel_attn_layers,
            mlp_bias,
        )
        self.spatial_att = BamSpatialAttn(
            channels,
            rd_ratio,
            rd_channels,
            rd_divisor,
            act_layer,
            gate_layer,
            num_spatial_attn_layers,
            mlp_bias,
        )

    def forward(self, x):
        x_attn = 1 + F.sigmoid(self.channel_att(x) * self.spatial_att(x))
        return x * x_attn
