"""Gated Channel Transformation <https://arxiv.org/abs/1909.11519>
"""

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import math


class GCTModule(nn.Module):
    def __init__(self, channels, epsilon=1e-5, mode="l2", after_relu=False):
        super(GCTModule, self).__init__()

        self.alpha = nn.Parameter(flow.ones(1, channels, 1, 1))
        self.gamma = nn.Parameter(flow.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(flow.zeros(1, channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        assert self.mode in ["l1", "l2"], "Unknown mode type in GCTModule"

        if self.mode == "l2":
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(
                0.5
            ) * self.alpha
            norm = self.gamma / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)
        elif self.mode == "l1":
            if not self.after_relu:
                _x = flow.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (
                flow.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon
            )

        gate = 1.0 + flow.tanh(embedding * norm + self.beta)
        return x * gate
