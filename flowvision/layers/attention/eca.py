import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import math

from ..helpers import make_divisible


class EcaModule(nn.Module):
    """ECA module
    """

    # TODO: add docstring
    def __init__(
        self, channels=None, kernel_size=3, gamma=2, beta=1, gate_layer=nn.Sigmoid,
    ):
        super(EcaModule, self).__init__()
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.gate = gate_layer()

    def forward(self, x):
        y = x.mean((2, 3)).view(x.shape[0], 1, -1)
        y = self.conv(y)
        y = self.gate(y).view(x.shape[0], -1, 1, 1)
        return x * y.expand_as(x)
