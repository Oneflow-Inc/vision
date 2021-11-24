"""Normalization layers warpped by OneFlow borrowed from timm
"""

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F


# TODO: switch self.norm to F.layer_norm
class LayerNorm2d(nn.Module):
    """ LayerNorm for channels of '2D' spatial BCHW tensors """

    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
