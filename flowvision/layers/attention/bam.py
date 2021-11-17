import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import math

from ..helpers import make_divisible
from flowvision.layers.build import LAYER_REGISTRY


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
            self, channels, rd_ratio=1./16, rd_channels=None, rd_divisor=1, act_layer=nn.ReLU, gate_layer=nn.Sigmoid, num_layers=2, mlp_bias=False):
        super(BamChannelAttn, self).__init__()
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc_layers = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.fc_layers.add_module("fc_bn_act_%d" % i, LinearBnAct(channels, rd_channels, act_layer, bias=mlp_bias))
            else:
                self.fc_layers.add_module("fc_bn_act_%d" % i, LinearBnAct(rd_channels, rd_channels, act_layer, bias=mlp_bias))
        self.fc_layers.add_module("fc_out", nn.Linear(rd_channels, channels, bias=mlp_bias))
        self.gate = gate_layer()
    
    def forward(self, x):
        b, c, _, _ = x.shape
        x_attn = self.gate(self.fc_layers(x.mean((2, 3)))).view(b, c, 1, 1)
        return x * x_attn.expand_as(x)