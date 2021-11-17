"""Normalization layers warpped by OneFlow borrowed from timm
"""

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial BCHW tensors """
    def __init__(self, num_channels):
        super().__init__(num_channels)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)