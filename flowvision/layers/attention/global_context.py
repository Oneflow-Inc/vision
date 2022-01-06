"""
Modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/global_context.py
"""

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import oneflow.nn.init as init

from flowvision.layers.helpers import make_divisible
from flowvision.layers.blocks import ConvMlp
from flowvision.layers.regularization import LayerNorm2d


class GlobalContext(nn.Module):
    def __init__(
        self,
        channels,
        use_attn=True,
        fuse_add=False,
        fuse_scale=True,
        init_last_zero=False,
        rd_ratio=1.0 / 8,
        rd_channels=None,
        rd_divisor=1,
        act_layer=nn.ReLU,
        gate_layer=nn.Sigmoid,
    ):
        super(GlobalContext, self).__init__()

        self.conv_attn = (
            nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None
        )

        if rd_channels is None:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        if fuse_add:
            self.mlp_add = ConvMlp(
                channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d
            )
        else:
            self.mlp_add = None
        if fuse_scale:
            self.mlp_scale = ConvMlp(
                channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d
            )
        else:
            self.mlp_scale = None

        self.gate = gate_layer()
        self.init_last_zero = init_last_zero
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_attn is not None:
            init.kaiming_normal_(
                self.conv_attn.weight, mode="fan_in", nonlinearity="relu"
            )
        if self.mlp_add is not None:
            init.zeros_(self.mlp_add.fc2.weight)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
            attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W * 1)
            context = flow.matmul(x.reshape(B, C, H * W).unsqueeze(1), attn)
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean((2, 3), keepdim=True)

        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x

        return x
