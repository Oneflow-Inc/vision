"""
Modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/non_local_attn.py
"""

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import oneflow.nn.init as init

from flowvision.layers.helpers import make_divisible


class NonLocalAttn(nn.Module):
    """Spatial non-local block for image classification
    """

    def __init__(
        self,
        in_channels,
        use_scale=True,
        rd_ratio=1 / 8,
        rd_channels=None,
        rd_divisor=8,
        **kwargs
    ):
        super(NonLocalAttn, self).__init__()
        if rd_channels is None:
            rd_channels = make_divisible(in_channels * rd_ratio, divisor=rd_divisor)
        self.scale = in_channels ** -0.5 if use_scale else 1.0
        self.t = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.p = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.g = nn.Conv2d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.z = nn.Conv2d(rd_channels, in_channels, kernel_size=1, stride=1, bias=True)
        self.norm = nn.BatchNorm2d(in_channels)
        self.reset_parameters()

    def forward(self, x):
        shortcut = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        B, C, H, W = t.size()
        t = t.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        p = p.view(B, C, -1)  # (B, C, N)
        g = g.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)

        att = flow.bmm(t, p) * self.scale
        att = F.softmax(att, dim=2)
        x = flow.bmm(att, g)

        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.z(x)
        x = self.norm(x) + shortcut

        return x

    def reset_parameters(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if len(list(m.parameters())) > 1:
                    init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 0)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                init.constant_(m.weight, 0)
                init.constant_(m.bias, 0)
