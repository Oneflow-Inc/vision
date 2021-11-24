import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import math


class hard_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(hard_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class hard_swish(nn.Module):
    def __init__(self, inplace=True):
        super(hard_swish, self).__init__()
        self.sigmoid = hard_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
