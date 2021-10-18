import oneflow as flow
import oneflow.nn as nn
from functools import partial


class ConvBnAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module = nn.ReLU,
        conv: nn.Module = nn.Conv2d,
        normalization: nn.Module = nn.BatchNorm2d,
        bias: bool = False,
        **kwargs
    ):
        super().__init__()
        self.add_module("conv", conv(in_features, out_features, **kwargs, bias=bias))
        if normalization:
            self.add_module("bn", normalization(out_features))
        if activation:
            self.add_module("act", activation())


ConvBn = partial(ConvBnAct, activation=None)
ConvAct = partial(ConvBnAct, normalization=None, bias=True)
